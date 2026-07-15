"""Panoramic depth estimation using Depth Anything 3 (DA3).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.sparse import csr_array, vstack
from scipy.sparse.linalg import lsmr

# ---------------------------------------------------------------------------
# Geometry utilities (self-contained, no utils3d dependency)
# ---------------------------------------------------------------------------


def create_icosahedron_vertices() -> np.ndarray:
    """Return the 12 vertices of a regular icosahedron on the unit sphere."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float32)
    vertices = vertices / np.linalg.norm(vertices, axis=-1, keepdims=True)
    return vertices


def look_at_extrinsics(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute a 4x4 world-to-camera extrinsic matrix (OpenCV convention).

    Camera looks from `eye` toward `target`, with `up` as the world up hint.
    Returns a (4, 4) w2c matrix.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    down = np.cross(forward, right)  # OpenCV: y-axis points down

    rotation = np.stack([right, down, forward], axis=0)  # (3, 3)
    translation = -rotation @ eye

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = translation
    return extrinsic


def intrinsics_from_fov(fov_rad: float, resolution: int) -> np.ndarray:
    """Create a 3x3 intrinsic matrix from field-of-view and resolution."""
    focal = resolution / (2.0 * np.tan(fov_rad / 2.0))
    cx = cy = (resolution - 1) / 2.0
    intrinsic = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1],
    ], dtype=np.float32)
    return intrinsic


def get_panorama_cameras(fov_deg: float = 90.0, resolution: int = 512):
    """Generate icosahedron-based cameras for panorama splitting (12 views with overlap).

    Returns:
        extrinsics: (12, 4, 4) w2c matrices
        intrinsics: (12, 3, 3) intrinsic matrices
    """
    vertices = create_icosahedron_vertices()
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    fov_rad = np.deg2rad(fov_deg)
    intrinsic = intrinsics_from_fov(fov_rad, resolution)

    extrinsics = []
    for vertex in vertices:
        # Camera at origin, looking toward vertex direction
        extrinsic = look_at_extrinsics(
            eye=np.zeros(3, dtype=np.float32),
            target=vertex,
            up=up,
        )
        extrinsics.append(extrinsic)

    extrinsics = np.stack(extrinsics, axis=0)
    intrinsics_all = np.tile(intrinsic[None], (len(vertices), 1, 1))
    return extrinsics, intrinsics_all


def get_cubemap_cameras(fov_deg: float = 90.0, resolution: int = 512):
    """Generate 6-face cubemap cameras for panorama splitting.

    The 6 faces correspond to +X, -X, +Y, -Y, +Z, -Z directions.
    With fov=90° faces have no overlap; with fov>90° adjacent faces overlap.

    Returns:
        extrinsics: (6, 4, 4) w2c matrices
        intrinsics: (6, 3, 3) intrinsic matrices
    """
    # 6 cubemap face directions and their up vectors
    # For ±Z faces, use a different up to avoid degeneracy
    face_configs = [
        ([1, 0, 0], [0, 0, 1]),   # +X, up=Z
        ([-1, 0, 0], [0, 0, 1]),  # -X, up=Z
        ([0, 1, 0], [0, 0, 1]),   # +Y, up=Z
        ([0, -1, 0], [0, 0, 1]),  # -Y, up=Z
        ([0, 0, 1], [0, -1, 0]),  # +Z (top), up=-Y
        ([0, 0, -1], [0, 1, 0]),  # -Z (bottom), up=+Y
    ]

    fov_rad = np.deg2rad(fov_deg)
    intrinsic = intrinsics_from_fov(fov_rad, resolution)

    extrinsics = []
    for target, up in face_configs:
        extrinsic = look_at_extrinsics(
            eye=np.zeros(3, dtype=np.float32),
            target=np.array(target, dtype=np.float32),
            up=np.array(up, dtype=np.float32),
        )
        extrinsics.append(extrinsic)

    extrinsics = np.stack(extrinsics, axis=0)
    intrinsics_all = np.tile(intrinsic[None], (6, 1, 1))
    return extrinsics, intrinsics_all


def uv_map(height: int, width: int) -> np.ndarray:
    """Create a UV coordinate map with shape (H, W, 2), values in [0, 1]."""
    u = np.linspace(0.5 / width, 1 - 0.5 / width, width, dtype=np.float32)
    v = np.linspace(0.5 / height, 1 - 0.5 / height, height, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(u, v)
    return np.stack([grid_u, grid_v], axis=-1)


def spherical_uv_to_directions(uv: np.ndarray) -> np.ndarray:
    """Convert equirectangular UV coords to unit 3D directions.

    UV convention: u in [0,1] maps to longitude [2pi, 0], v in [0,1] maps to colatitude [0, pi].
    """
    theta = (1 - uv[..., 0]) * (2 * np.pi)
    phi = uv[..., 1] * np.pi
    directions = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], axis=-1)
    return directions


def directions_to_spherical_uv(directions: np.ndarray) -> np.ndarray:
    """Convert 3D directions to equirectangular UV coords."""
    directions = directions / (np.linalg.norm(directions, axis=-1, keepdims=True) + 1e-8)
    u = 1 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(np.clip(directions[..., 2], -1, 1)) / np.pi
    return np.stack([u, v], axis=-1)


def unproject_pixels(uv: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Unproject UV pixel coords to camera-space rays (z=1 plane).

    Args:
        uv: (H, W, 2) UV coords in [0, 1]
        intrinsics: (3, 3) camera intrinsic matrix

    Returns:
        rays: (H, W, 3) rays in camera space
    """
    height, width = uv.shape[:2]
    # Convert UV to pixel coordinates
    px = uv[..., 0] * (width - 1)  # note: width of the perspective view
    py = uv[..., 1] * (height - 1)

    # Actually, for a general resolution perspective view:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    rays_x = (px - cx) / fx
    rays_y = (py - cy) / fy
    rays_z = np.ones_like(rays_x)

    return np.stack([rays_x, rays_y, rays_z], axis=-1)


def project_points(points_3d: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray):
    """Project 3D world points to 2D UV coords in a camera view.

    Args:
        points_3d: (..., 3) points in world space
        extrinsics: (4, 4) w2c matrix
        intrinsics: (3, 3) intrinsic matrix

    Returns:
        uv: (..., 2) UV coords in [0, 1]
        depth: (...,) z-depth in camera space
    """
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]

    # Transform to camera space
    points_cam = (points_3d @ rotation.T) + translation
    depth = points_cam[..., 2]

    # Project to pixel
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    px = points_cam[..., 0] / (depth + 1e-8) * fx + cx
    py = points_cam[..., 1] / (depth + 1e-8) * fy + cy

    # Get resolution from intrinsics (cx ~ (res-1)/2)
    res_w = int(round(cx * 2 + 1))
    res_h = int(round(cy * 2 + 1))

    u = px / (res_w - 1)
    v = py / (res_h - 1)

    return np.stack([u, v], axis=-1), depth


def split_panorama_image(
    image: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    resolution: int,
) -> list[np.ndarray]:
    """Split equirectangular panorama into perspective views.

    Args:
        image: (H, W, 3) equirectangular image
        extrinsics: (N, 4, 4) w2c matrices
        intrinsics: (N, 3, 3) intrinsic matrices
        resolution: output resolution for each perspective view

    Returns:
        List of N perspective images (resolution, resolution, 3)
    """
    height, width = image.shape[:2]
    uv_grid = uv_map(resolution, resolution)

    splitted_images = []
    for i in range(len(extrinsics)):
        # Unproject perspective pixels to camera-space rays
        rays_cam = unproject_pixels(uv_grid, intrinsics[i])

        # Transform rays to world space (inverse of w2c rotation)
        rotation = extrinsics[i, :3, :3]
        rays_world = rays_cam @ rotation  # R^T @ ray_cam = ray_world (since R is orthogonal)

        # Convert world rays to equirectangular UV
        spherical_uv = directions_to_spherical_uv(rays_world)

        # Convert UV to pixel coordinates for remapping
        pixel_x = spherical_uv[..., 0] * (width - 1)
        pixel_y = spherical_uv[..., 1] * (height - 1)

        # Remap
        splitted_image = cv2.remap(
            image,
            pixel_x.astype(np.float32),
            pixel_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        splitted_images.append(splitted_image)

    return splitted_images


def zdepth_to_distance(depth_map: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Convert z-depth map to radial distance (Euclidean distance from camera center).

    Args:
        depth_map: (H, W) z-depth values
        intrinsics: (3, 3) camera intrinsic matrix

    Returns:
        distance_map: (H, W) radial distances
    """
    height, width = depth_map.shape
    uv_grid = uv_map(height, width)
    rays = unproject_pixels(uv_grid, intrinsics)
    ray_lengths = np.linalg.norm(rays, axis=-1)
    return depth_map * ray_lengths


def merge_cubemap_blended_to_panorama(
    width: int,
    height: int,
    distance_maps: list[np.ndarray],
    pred_masks: list[np.ndarray],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge cubemap depth maps with cosine-weighted blending in overlap regions.

    Uses slightly expanded FOV (>90°) so adjacent faces overlap. In the overlap
    regions, depth values are blended using cosine weights based on the angle
    between the ray and each face's optical axis. Pixels near the center of a
    face get higher weight than pixels near the edge.

    This achieves seamless transitions without expensive Poisson solving.

    Args:
        width, height: output panorama resolution
        distance_maps: list of N (H, W) radial distance maps per face
        pred_masks: list of N (H, W) boolean masks per face
        extrinsics: (N, 4, 4) w2c matrices
        intrinsics: (N, 3, 3) intrinsic matrices

    Returns:
        panorama_depth: (H, W) merged panoramic depth
        panorama_mask: (H, W) valid region mask
    """
    pano_uv = uv_map(height, width)
    spherical_dirs = spherical_uv_to_directions(pano_uv)

    # Accumulate weighted depth and total weight
    weighted_depth_sum = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)

    for i in range(len(distance_maps)):
        view_res = distance_maps[i].shape[0]

        # Project panorama directions into this face
        projected_uv, projected_depth = project_points(
            spherical_dirs, extrinsics[i], intrinsics[i]
        )

        # Valid: in front of camera and within image bounds
        valid = (
            (projected_depth > 0)
            & (projected_uv[..., 0] >= 0) & (projected_uv[..., 0] <= 1)
            & (projected_uv[..., 1] >= 0) & (projected_uv[..., 1] <= 1)
        )

        # Sample distance from this face
        pixel_x = (np.clip(projected_uv[..., 0], 0, 1) * (view_res - 1)).astype(np.float32)
        pixel_y = (np.clip(projected_uv[..., 1], 0, 1) * (view_res - 1)).astype(np.float32)

        sampled_distance = cv2.remap(
            distance_maps[i].astype(np.float32),
            pixel_x, pixel_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        sampled_mask = cv2.remap(
            pred_masks[i].astype(np.uint8),
            pixel_x, pixel_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE,
        ) > 0

        # Compute cosine weight: based on angle from face optical axis.
        # In camera space, optical axis = [0, 0, 1].
        # For projected UV (u, v), the ray direction in camera space is:
        #   ray = [(u*res - cx)/fx, (v*res - cy)/fy, 1]
        # cos(angle) = 1 / ||ray||
        # We use cos²(angle) as weight for smooth falloff.
        fx = intrinsics[i, 0, 0]
        fy = intrinsics[i, 1, 1]
        cx = intrinsics[i, 0, 2]
        cy = intrinsics[i, 1, 2]

        ray_x = (projected_uv[..., 0] * (view_res - 1) - cx) / fx
        ray_y = (projected_uv[..., 1] * (view_res - 1) - cy) / fy
        cos_angle_sq = 1.0 / (ray_x ** 2 + ray_y ** 2 + 1.0)
        # Use cos^4 for sharper center weighting (smoother blending at edges)
        weight = cos_angle_sq ** 2  # cos^4(angle)

        # Apply validity mask
        effective_mask = valid & sampled_mask
        weight = np.where(effective_mask, weight, 0.0)

        weighted_depth_sum += sampled_distance.astype(np.float64) * weight
        weight_sum += weight

    # Compute final depth
    panorama_mask = weight_sum > 0
    panorama_depth = np.zeros((height, width), dtype=np.float32)
    panorama_depth[panorama_mask] = (
        weighted_depth_sum[panorama_mask] / weight_sum[panorama_mask]
    ).astype(np.float32)

    return panorama_depth, panorama_mask


# ---------------------------------------------------------------------------
# Poisson-based panorama depth merging (adapted from MoGe)
# ---------------------------------------------------------------------------


def grad_equation(width: int, height: int, wrap_x: bool = False) -> csr_array:
    """Build gradient operator matrix for solving depth in log-space."""
    grid_index = np.arange(width * height).reshape(height, width)
    if wrap_x:
        grid_index_padded = np.pad(grid_index, ((0, 0), (0, 1)), mode='wrap')
    else:
        grid_index_padded = np.pad(grid_index, ((0, 0), (0, 1)), mode='edge')

    # Horizontal gradients
    data_x = np.concatenate([
        np.ones((height, width), dtype=np.float32).reshape(-1, 1),
        -np.ones((height, width), dtype=np.float32).reshape(-1, 1),
    ], axis=1).reshape(-1)
    indices_x = np.concatenate([
        grid_index_padded[:, :-1].reshape(-1, 1),
        grid_index_padded[:, 1:].reshape(-1, 1),
    ], axis=1).reshape(-1)

    # Vertical gradients
    grid_index_v = np.pad(grid_index, ((0, 1), (0, 0)), mode='edge')
    data_y = np.concatenate([
        np.ones((height, width), dtype=np.float32).reshape(-1, 1),
        -np.ones((height, width), dtype=np.float32).reshape(-1, 1),
    ], axis=1).reshape(-1)
    indices_y = np.concatenate([
        grid_index_v[:-1, :].reshape(-1, 1),
        grid_index_v[1:, :].reshape(-1, 1),
    ], axis=1).reshape(-1)

    data = np.concatenate([data_x, data_y])
    indices = np.concatenate([indices_x, indices_y])
    num_rows = height * width + height * width
    indptr = np.arange(0, num_rows * 2 + 1, 2)

    return csr_array((data, indices, indptr), shape=(num_rows, height * width))


def poisson_equation(width: int, height: int, wrap_x: bool = False) -> csr_array:
    """Build Laplacian operator matrix."""
    grid_index = np.arange(height * width).reshape(height, width)
    grid_index_padded = np.pad(grid_index, ((0, 0), (1, 1)), mode='wrap' if wrap_x else 'edge')
    grid_index_padded = np.pad(grid_index_padded, ((1, 1), (0, 0)), mode='edge')

    data = np.array([[-4, 1, 1, 1, 1]], dtype=np.float32).repeat(height * width, axis=0).reshape(-1)
    indices = np.stack([
        grid_index_padded[1:-1, 1:-1],
        grid_index_padded[:-2, 1:-1],   # up
        grid_index_padded[2:, 1:-1],    # down
        grid_index_padded[1:-1, :-2],   # left
        grid_index_padded[1:-1, 2:],    # right
    ], axis=-1).reshape(-1)
    indptr = np.arange(0, height * width * 5 + 1, 5)

    return csr_array((data, indices, indptr), shape=(height * width, height * width))


def merge_panorama_depth(
    width: int,
    height: int,
    distance_maps: list[np.ndarray],
    pred_masks: list[np.ndarray],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge multiple perspective depth maps into a panoramic depth map.

    Uses Poisson-based merging in log-space for smooth results.
    """
    # Multi-resolution: solve coarse first for initialization
    if max(width, height) > 256:
        panorama_depth_init, _ = merge_panorama_depth(
            width // 2, height // 2, distance_maps, pred_masks, extrinsics, intrinsics
        )
        panorama_depth_init = cv2.resize(panorama_depth_init, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        panorama_depth_init = None

    pano_uv = uv_map(height, width)
    spherical_dirs = spherical_uv_to_directions(pano_uv)

    panorama_log_distance_laplacian_maps = []
    panorama_laplacian_masks = []
    panorama_log_distance_grad_maps = []
    panorama_grad_masks = []
    panorama_pred_masks = []

    for i in range(len(distance_maps)):
        projected_uv, projected_depth = project_points(spherical_dirs, extrinsics[i], intrinsics[i])
        view_res = distance_maps[i].shape[0]

        projection_valid_mask = (
            (projected_depth > 0)
            & (projected_uv[..., 0] > 0) & (projected_uv[..., 0] < 1)
            & (projected_uv[..., 1] > 0) & (projected_uv[..., 1] < 1)
        )

        pixel_x = (np.clip(projected_uv[..., 0], 0, 1) * (view_res - 1)).astype(np.float32)
        pixel_y = (np.clip(projected_uv[..., 1], 0, 1) * (view_res - 1)).astype(np.float32)

        log_distance = np.log(np.clip(distance_maps[i], 1e-6, None))
        panorama_log_dist = np.where(
            projection_valid_mask,
            cv2.remap(log_distance, pixel_x, pixel_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
            0,
        )

        mask_remapped = cv2.remap(
            pred_masks[i].astype(np.uint8), pixel_x, pixel_y,
            cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE,
        ) > 0
        panorama_pred_mask = projection_valid_mask & mask_remapped

        # Gradient
        padded = np.pad(panorama_log_dist, ((0, 0), (0, 1)), mode='wrap')
        grad_x = padded[:, :-1] - padded[:, 1:]
        padded_y = np.pad(panorama_log_dist, ((0, 1), (0, 0)), mode='edge')
        grad_y = padded_y[:-1, :] - padded_y[1:, :]

        padded_mask = np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode='wrap')
        mask_x = padded_mask[:, :-1] & padded_mask[:, 1:]
        padded_mask_y = np.pad(panorama_pred_mask, ((0, 1), (0, 0)), mode='edge')
        mask_y = padded_mask_y[:-1, :] & padded_mask_y[1:, :]

        panorama_log_distance_grad_maps.append((grad_x, grad_y))
        panorama_grad_masks.append((mask_x, mask_y))

        # Laplacian
        padded_lap = np.pad(panorama_log_dist, ((1, 1), (0, 0)), mode='edge')
        padded_lap = np.pad(padded_lap, ((0, 0), (1, 1)), mode='wrap')
        laplacian = convolve(padded_lap, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32))[1:-1, 1:-1]

        padded_mask_lap = np.pad(panorama_pred_mask, ((1, 1), (0, 0)), mode='edge')
        padded_mask_lap = np.pad(padded_mask_lap, ((0, 0), (1, 1)), mode='wrap')
        lap_mask = convolve(padded_mask_lap.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))[1:-1, 1:-1] == 5

        panorama_log_distance_laplacian_maps.append(laplacian)
        panorama_laplacian_masks.append(lap_mask)
        panorama_pred_masks.append(panorama_pred_mask)

    # Average gradients across views
    grad_x_stack = np.stack([g[0] for g in panorama_log_distance_grad_maps], axis=0)
    grad_y_stack = np.stack([g[1] for g in panorama_log_distance_grad_maps], axis=0)
    mask_x_stack = np.stack([m[0] for m in panorama_grad_masks], axis=0)
    mask_y_stack = np.stack([m[1] for m in panorama_grad_masks], axis=0)

    avg_grad_x = np.sum(grad_x_stack * mask_x_stack, axis=0) / np.sum(mask_x_stack, axis=0).clip(1e-3)
    avg_grad_y = np.sum(grad_y_stack * mask_y_stack, axis=0) / np.sum(mask_y_stack, axis=0).clip(1e-3)

    laplacian_stack = np.stack(panorama_log_distance_laplacian_maps, axis=0)
    lap_mask_stack = np.stack(panorama_laplacian_masks, axis=0)
    avg_laplacian = np.sum(laplacian_stack * lap_mask_stack, axis=0) / np.sum(lap_mask_stack, axis=0).clip(1e-3)

    grad_x_valid = np.any(mask_x_stack, axis=0).reshape(-1)
    grad_y_valid = np.any(mask_y_stack, axis=0).reshape(-1)
    grad_valid = np.concatenate([grad_x_valid, grad_y_valid])
    laplacian_valid = np.any(lap_mask_stack, axis=0).reshape(-1)

    # Solve
    grad_op = grad_equation(width, height, wrap_x=True)
    lap_op = poisson_equation(width, height, wrap_x=True)

    system_matrix = vstack([
        grad_op[grad_valid],
        lap_op[laplacian_valid],
    ])
    rhs = np.concatenate([
        avg_grad_x.reshape(-1)[grad_x_valid],
        avg_grad_y.reshape(-1)[grad_y_valid],
        avg_laplacian.reshape(-1)[laplacian_valid],
    ])

    initial_guess = np.log(panorama_depth_init).reshape(-1) if panorama_depth_init is not None else None
    solution, *_ = lsmr(system_matrix, rhs, atol=1e-5, btol=1e-5, x0=initial_guess, show=False)

    panorama_depth = np.exp(solution).reshape(height, width).astype(np.float32)
    panorama_mask = np.any(panorama_pred_masks, axis=0)

    return panorama_depth, panorama_mask


# ---------------------------------------------------------------------------
# DA3 inference pipeline for panorama
# ---------------------------------------------------------------------------


def _get_gpu_mem(reserved=False) -> float:
    """Return current GPU memory usage in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return (torch.cuda.memory_reserved() if reserved
                    else torch.cuda.memory_allocated()) / 1e9
        return 0.0
    except Exception:
        return 0.0


def save_perspective_views_as_images(images: list[np.ndarray], temp_dir: str) -> list[str]:
    """Save perspective view images to disk for DA3 input."""
    paths = []
    for idx, img in enumerate(images):
        path = os.path.join(temp_dir, f"view_{idx:02d}.png")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


def load_da3_model(model_dir: str):
    """Load DA3 model, handling both safetensors and pytorch_model.bin formats.

    Args:
        model_dir: HuggingFace model name or local path

    Returns:
        Loaded DepthAnything3 model on GPU
    """
    import torch
    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading DA3 model from: {model_dir}")
    try:
        model = DepthAnything3.from_pretrained(model_dir)
    except (FileNotFoundError, OSError):
        # Fallback: try loading with explicit safetensors filename
        # This handles older huggingface_hub versions that default to pytorch_model.bin
        try:
            model = DepthAnything3.from_pretrained(
                model_dir, weights_name="model.safetensors"
            )
        except TypeError:
            # Even older versions may not support weights_name kwarg
            # Try creating a symlink as last resort
            import os
            from pathlib import Path

            model_path = Path(model_dir)
            safetensors_file = model_path / "model.safetensors"
            bin_file = model_path / "pytorch_model.bin"

            if safetensors_file.exists() and not bin_file.exists():
                # Load safetensors manually
                from safetensors.torch import load_file

                model = DepthAnything3(model_name=_detect_model_name(model_path))
                state_dict = load_file(str(safetensors_file))
                model.load_state_dict(state_dict, strict=False)
            else:
                raise

    model = model.to(device=device)
    print("DA3 model loaded.")
    return model


def _detect_model_name(model_path) -> str:
    """Detect DA3 model name from config.json in model directory."""
    import json
    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        return config.get("model_name", "da3nested-giant-large-1.1")
    return "da3nested-giant-large-1.1"


def run_da3_inference(
    image_paths: list[str],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    model_dir: str,
    process_res: int = 504,
) -> np.ndarray:
    """Run DA3 multi-view pose-conditioned inference.

    Args:
        image_paths: list of perspective view image paths
        extrinsics: (N, 4, 4) w2c matrices
        intrinsics: (N, 3, 3) intrinsic matrices
        model_dir: HuggingFace model name or local path
        process_res: DA3 processing resolution

    Returns:
        depth_maps: (N, H, W) z-depth maps
    """
    model = load_da3_model(model_dir)

    # Monkey-patch: skip pose alignment post-processing.
    # All panorama cameras are at the origin (only rotations differ),
    # so Umeyama Sim3 alignment is degenerate. The forward pass still uses
    # extrinsics for pose-conditioned inference; we just skip the post-hoc alignment.
    model._align_to_input_extrinsics_intrinsics = lambda *args, **kwargs: args[2]  # return prediction as-is

    import time as _time
    print(f"Running DA3 inference on {len(image_paths)} views (pose-conditioned)...")
    print(f"  GPU memory before inference: {_get_gpu_mem():.1f}GB allocated, {_get_gpu_mem(reserved=True):.1f}GB reserved")
    _t0 = _time.perf_counter()
    prediction = model.inference(
        image_paths,
        extrinsics=extrinsics.astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        process_res=process_res,
        process_res_method="upper_bound_resize",
    )
    _t1 = _time.perf_counter()
    print(f"  DA3 inference completed in {_t1 - _t0:.1f}s")
    print(f"  GPU memory after inference: {_get_gpu_mem():.1f}GB allocated, {_get_gpu_mem(reserved=True):.1f}GB reserved")

    # prediction.depth is (N, H, W) float32 numpy array
    return prediction.depth


def infer_panorama_depth(
    image_path: str,
    output_dir: str,
    model_dir: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    view_resolution: int = 512,
    fov_deg: float = 90.0,
    process_res: int = 504,
    split_mode: str = "cubemap_overlap",
):
    """Full pipeline: panorama → split → DA3 inference → merge → save.

    Args:
        image_path: path to equirectangular panorama image
        output_dir: output directory for depth.npy
        model_dir: DA3 model (HuggingFace name or local path)
        view_resolution: resolution of each perspective view
        fov_deg: field of view for perspective views (cubemap_overlap uses max(fov, 100°))
        process_res: DA3 internal processing resolution
        split_mode: "cubemap_overlap" (6 faces, cosine blending, default) or
                    "icosahedron" (12 faces, Poisson merge)
    """
    # Load panorama
    print(f"Loading panorama: {image_path}")
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"Cannot read image: {image_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pano_height, pano_width = img_rgb.shape[:2]
    print(f"Panorama size: {pano_width} x {pano_height}")

    # Generate cameras based on split mode
    if split_mode == "icosahedron":
        extrinsics, intrinsics = get_panorama_cameras(fov_deg=90, resolution=view_resolution)
    else:
        # cubemap_overlap: use wider FOV for overlap blending
        effective_fov = max(fov_deg, 100.0)
        extrinsics, intrinsics = get_cubemap_cameras(fov_deg=effective_fov, resolution=view_resolution)

    num_views = len(extrinsics)
    actual_fov = max(fov_deg, 100.0) if split_mode == "cubemap_overlap" else 90
    print(f"Split mode: {split_mode} ({num_views} views, {view_resolution}x{view_resolution}, {actual_fov}° FOV)")

    # Split panorama into perspective views
    splitted_images = split_panorama_image(img_rgb, extrinsics, intrinsics, view_resolution)

    # Save perspective views to temp directory for DA3 input
    temp_dir = tempfile.mkdtemp(prefix="da3_pano_")
    image_paths = save_perspective_views_as_images(splitted_images, temp_dir)

    # Run DA3 inference (all views at once, pose-conditioned)
    depth_maps = run_da3_inference(
        image_paths, extrinsics, intrinsics, model_dir, process_res=process_res
    )

    # Convert z-depth to radial distance for each view
    print("Converting z-depth to radial distance...")
    distance_maps = []
    pred_masks = []
    for i in range(num_views):
        # DA3 output resolution may differ from view_resolution
        da3_height, da3_width = depth_maps[i].shape
        # Create intrinsics matching DA3's output resolution
        scale_h = da3_height / view_resolution
        scale_w = da3_width / view_resolution
        scaled_intrinsics = intrinsics[i].copy()
        scaled_intrinsics[0, 0] *= scale_w  # fx
        scaled_intrinsics[0, 2] *= scale_w  # cx
        scaled_intrinsics[1, 1] *= scale_h  # fy
        scaled_intrinsics[1, 2] *= scale_h  # cy

        distance = zdepth_to_distance(depth_maps[i], scaled_intrinsics)
        mask = depth_maps[i] > 0  # valid depth mask

        distance_maps.append(distance)
        pred_masks.append(mask)

    # Merge into panoramic depth
    merge_height = min(1024, pano_height)
    merge_width = merge_height * 2
    print(f"Merging depth maps to panorama ({merge_width} x {merge_height}, mode={split_mode})...")

    if split_mode == "cubemap_overlap":
        # Cosine-weighted blending in overlap regions (fast + seamless)
        panorama_depth, panorama_mask = merge_cubemap_blended_to_panorama(
            merge_width, merge_height, distance_maps, pred_masks, extrinsics, intrinsics
        )
    else:
        # Poisson-based merge (handles overlap, higher quality but slow)
        panorama_depth, panorama_mask = merge_panorama_depth(
            merge_width, merge_height, distance_maps, pred_masks, extrinsics, intrinsics
        )

    # Resize to original panorama resolution if needed
    if panorama_depth.shape[0] != pano_height or panorama_depth.shape[1] != pano_width:
        panorama_depth = cv2.resize(panorama_depth, (pano_width, pano_height), interpolation=cv2.INTER_LINEAR)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "depth.npy")
    np.save(output_path, panorama_depth)

    print(f"\nDepth saved to: {output_path}")
    print(f"Depth shape: {panorama_depth.shape}, range: [{panorama_depth.min():.4f}, {panorama_depth.max():.4f}]")

    # Cleanup temp files
    for path in image_paths:
        os.remove(path)
    os.rmdir(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="Panoramic depth estimation using DA3")
    parser.add_argument("--path", type=str, required=True, help="Path to equirectangular panorama image")
    parser.add_argument("--output_dir", type=str, default="./infer_results", help="Output directory")
    parser.add_argument("--model_dir", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                        help="DA3 pretrained model (HuggingFace name or local path)")
    parser.add_argument("--view_resolution", type=int, default=512,
                        help="Resolution of each perspective view")
    parser.add_argument("--fov", type=float, default=120.0,
                        help="Field of view (degrees) for perspective views")
    parser.add_argument("--process_res", type=int, default=504,
                        help="DA3 internal processing resolution")
    parser.add_argument("--split_mode", type=str, default="cubemap_overlap",
                        choices=["cubemap_overlap", "icosahedron"],
                        help="Panorama split strategy: "
                             "'cubemap_overlap' (6 faces, 100° FOV, cosine blending, fast+seamless, default), "
                             "'icosahedron' (12 faces, Poisson merge, highest quality but slow)")
    parser.add_argument("--gpu", default="0", help="GPU device id")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    infer_panorama_depth(
        image_path=args.path,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        view_resolution=args.view_resolution,
        fov_deg=args.fov,
        process_res=args.process_res,
        split_mode=args.split_mode,
    )


if __name__ == "__main__":
    main()
