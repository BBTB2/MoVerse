import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from autolevel.geometry import persp_to_erp_grid
from autolevel.modules import MSCAN, LightHamHead, ConvModule


class LowLevelEncoder(nn.Module):
    """Two-conv full-resolution feature encoder."""

    def __init__(self):
        super().__init__()
        self.conv1 = ConvModule(3, 64, kernel_size=3, padding=1)
        self.conv2 = ConvModule(64, 64, kernel_size=3, padding=1)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = data["image"]
        assert x.shape[-1] % 32 == 0 and x.shape[-2] % 32 == 0
        return {"features": self.conv2(self.conv1(x))}


def _ray_pe(H: int, W: int, device: torch.device) -> torch.Tensor:
    """2-channel positional encoding [gx * AR, gy], returned as (1, 2, H, W)."""
    ar = W / float(H)
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device),
        torch.linspace(-1.0, 1.0, W, device=device),
        indexing="ij",
    )
    return torch.stack([gx * ar, gy], dim=0).unsqueeze(0)


class FlowDecoder(nn.Module):
    """Predicts a dense per-pixel ERP forward warp residual from multi-scale
    backbone features, injecting a ray positional encoding at two stages."""

    def __init__(self):
        super().__init__()
        self.ham_head = LightHamHead()
        ham_ch = self.ham_head.ham_channels  # 512
        out_ch = self.ham_head.out_channels  # 64

        self.ray_pe_proj    = nn.Conv2d(ham_ch + 2, ham_ch, kernel_size=1)
        self.linear_pred_flow = nn.Conv2d(out_ch + 2, 2, kernel_size=1)

    def forward(
        self, features: Dict[str, torch.Tensor], ray_pe: torch.Tensor
    ) -> torch.Tensor:
        head = self.ham_head
        inputs = [features["hl"][i] for i in head.in_index]
        inputs = [
            F.interpolate(level, size=inputs[0].shape[2:], mode="bilinear", align_corners=head.align_corners)
            for level in inputs
        ]
        x = head.squeeze(torch.cat(inputs, dim=1))

        pe_small = F.interpolate(ray_pe, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = self.ray_pe_proj(torch.cat([x, pe_small], dim=1))

        x = head.hamburger(x)
        feats = head.align(x)

        feats = F.interpolate(feats, scale_factor=2, mode="bilinear", align_corners=False)
        feats = head.out_conv(feats)
        feats = F.interpolate(feats, scale_factor=2, mode="bilinear", align_corners=False)
        feats = head.ll_fusion(feats, features["ll"].clone())

        feats_pe = torch.cat([feats, ray_pe], dim=1)
        return self.linear_pred_flow(feats_pe)


def _camera_pixel_uv(p: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """Per-pixel forward model: camera params → ERP (u, v) for one pixel."""
    log_fov, pitch, roll = p[0], p[1], p[2]
    f = 1.0 / torch.tan(torch.exp(log_fov) * (math.pi / 180.0) / 2.0)

    gxf, gyf = gx.to(dtype=f.dtype), gy.to(dtype=f.dtype)
    ray = torch.stack([gxf, gyf, f])
    ray = ray / ray.norm()

    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll),  torch.sin(roll)
    z, o = cp * 0.0, cp * 0.0 + 1.0

    Rx = torch.stack([torch.stack([o, z, z]), torch.stack([z, cp, -sp]), torch.stack([z, sp, cp])])
    Rz = torch.stack([torch.stack([cr, -sr, z]), torch.stack([sr, cr, z]), torch.stack([z, z, o])])
    d = (Rz @ Rx) @ ray

    u = torch.atan2(d[0], d[2]) / math.pi
    v = torch.asin(torch.clamp(d[1], -1.0 + 1e-6, 1.0 - 1e-6)) / (math.pi / 2.0)
    return torch.stack([u, v])


class RigidFilter(nn.Module):
    """Fits a 3-DoF pinhole camera (vFOV, pitch, roll) to the dense predicted
    warp field, then reconstructs a geometrically
    consistent warp grid from the fitted parameters."""

    def __init__(self, lm_steps: int = 10, lm_lambda: float = 0.1, fit_res: int = 128, **_):
        super().__init__()
        self.lm_steps  = lm_steps
        self.lm_lambda = lm_lambda
        self.fit_res   = fit_res

    @staticmethod
    @torch.no_grad()
    def _init_params(flow_small: torch.Tensor, aspect: float, device: torch.device) -> torch.Tensor:
        """SVD-based closed-form initialisation of (log_fov, pitch_rad, roll_rad)."""
        sh, sw = flow_small.shape[1], flow_small.shape[2]

        u_flat, v_flat = flow_small[0].flatten(), flow_small[1].flatten()
        phi, theta = u_flat * math.pi, v_flat * (math.pi / 2)
        cos_t = torch.cos(theta)
        d_world = torch.stack([torch.sin(phi) * cos_t, torch.sin(theta), torch.cos(phi) * cos_t], dim=-1)

        v_span = float((flow_small[1].max() - flow_small[1].min()).clamp(0.05, 1.98))
        f_est  = 1.0 / math.tan(math.radians(max(10.0, min(v_span * 90.0, 150.0)) / 2))

        y_cam = torch.linspace(-1.0, 1.0, sh, device=device, dtype=torch.float32)
        x_cam = torch.linspace(-aspect, aspect, sw, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(y_cam, x_cam, indexing="ij")
        d_cam = F.normalize(torch.stack([gx, gy, torch.full_like(gx, f_est)], dim=-1).reshape(-1, 3), dim=-1)

        M = d_cam.T @ d_world
        try:
            U, _, Vt = torch.linalg.svd(M)
            D = torch.diag(torch.tensor([1.0, 1.0, torch.linalg.det(Vt.T @ U.T).item()], device=device))
            R = Vt.T @ D @ U.T
        except Exception:
            R = torch.eye(3, device=device)

        pitch_rad = torch.asin(torch.clamp(R[2, 1], -1.0 + 1e-6, 1.0 - 1e-6))
        roll_rad  = torch.atan2(R[1, 0], R[0, 0])

        fov_cands = torch.linspace(10.0, 150.0, 1401, device=device)
        f_cands   = 1.0 / torch.tan(torch.deg2rad(fov_cands / 2.0))
        gx_f, gy_f = gx.reshape(-1), gy.reshape(-1)
        d_cam_all = F.normalize(torch.stack([
            gx_f.unsqueeze(0).expand(1401, -1),
            gy_f.unsqueeze(0).expand(1401, -1),
            f_cands.unsqueeze(1).expand(-1, gx_f.shape[0]),
        ], dim=-1), dim=-1)
        errs = ((d_cam_all @ R.T - d_world.unsqueeze(0)) ** 2).sum(dim=(-1, -2))
        log_fov = torch.log(fov_cands[errs.argmin()])

        return torch.stack([log_fov, pitch_rad, roll_rad])

    @staticmethod
    def _lm_step(
        p: torch.Tensor, target: torch.Tensor,
        gx: torch.Tensor, gy: torch.Tensor, lamb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from torch.func import jacrev, vmap
        N = gx.numel()
        gx_f, gy_f = gx.reshape(N), gy.reshape(N)
        J = vmap(lambda xi, yi: jacrev(lambda pp: _camera_pixel_uv(pp, xi, yi))(p))(gx_f, gy_f)
        r = vmap(lambda xi, yi: _camera_pixel_uv(p, xi, yi))(gx_f, gy_f) - target.permute(1, 2, 0).reshape(N, 2)
        H  = torch.einsum("Nji,Njk->ik", J, J)
        g  = torch.einsum("Nji,Nj->i",  J, r)
        Hd = H + (H.diagonal() * lamb).clamp(min=1e-6).diag_embed()
        try:
            delta = torch.cholesky_solve(g.cpu().unsqueeze(-1), torch.linalg.cholesky(Hd.cpu())).squeeze(-1).to(p.device)
        except RuntimeError:
            delta = torch.zeros_like(g)
        p_new = torch.stack([
            (p - delta)[0].clamp(math.log(10.0), math.log(150.0)),
            (p - delta)[1], (p - delta)[2],
        ])
        return p_new, (r ** 2).sum()

    def _fit_one(self, flow_small: torch.Tensor, aspect: float, device: torch.device) -> torch.Tensor:
        sh, sw = flow_small.shape[1], flow_small.shape[2]
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, sh, device=device, dtype=torch.float32),
            torch.linspace(-aspect, aspect, sw, device=device, dtype=torch.float32),
            indexing="ij",
        )
        p = self._init_params(flow_small, aspect, device)
        lamb, prev_cost = flow_small.new_tensor(self.lm_lambda), None
        for _ in range(self.lm_steps):
            p, cost = self._lm_step(p, flow_small, gx, gy, lamb)
            if prev_cost is not None:
                lamb = torch.where(cost > prev_cost, lamb * 10.0, lamb * 0.1).clamp(1e-6, 1e2)
            prev_cost = cost
        return p

    def forward(self, dense_grid: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device, B, dtype = dense_grid.device, dense_grid.shape[0], dense_grid.dtype
        aspect = W / float(H)
        s = self.fit_res
        sw = s if aspect >= 1.0 else max(int(s * aspect), 1)
        sh = s if aspect < 1.0  else max(int(s / aspect), 1)
        small = F.interpolate(dense_grid.detach().float(), size=(sh, sw), mode="bilinear", align_corners=False)

        fovs, pitches, rolls = [], [], []
        for b in range(B):
            p = self._fit_one(small[b], aspect, device)
            fovs.append(torch.exp(p[0]))
            pitches.append(torch.rad2deg(p[1]))
            rolls.append(torch.rad2deg(p[2]))

        fovs    = torch.stack(fovs)
        pitches = torch.stack(pitches)
        rolls   = torch.stack(rolls)
        params  = torch.stack([fovs, pitches, rolls], dim=1)
        rigid_grid = persp_to_erp_grid(fovs, pitches, rolls, H, W, device=str(device)).to(dtype)
        return rigid_grid, params.to(dtype)


class FlowEstimator(nn.Module):
    """Single-image camera leveling model.

    Input:  RGB image (B, 3, H, W) in [0, 1], H and W multiples of 32.
    Output: dict with
        'dense_grid'  (B, 2, H, W) — raw per-pixel warp prediction.
        'rigid_grid'  (B, 2, H, W) — warp rebuilt from the fitted camera.
        'params'      (B, 3)       — estimated (vFOV, pitch, roll) in degrees.
    """

    _BASE_FOV = 60.0

    def __init__(self, freeze_backbone: bool = True, rigid_filter_cfg: dict | None = None):
        super().__init__()
        self.backbone     = MSCAN()
        self.ll_enc       = LowLevelEncoder()
        self.decoder      = FlowDecoder()
        self.rigid_filter = RigidFilter(**(rigid_filter_cfg or {}))

        if freeze_backbone:
            for module in (self.backbone, self.ll_enc):
                for p in module.parameters():
                    p.requires_grad = False
                module.eval()

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = image.shape

        hl_feats = self.backbone({"image": image})["features"]
        ll_feats = self.ll_enc({"image": image})["features"]

        ray_pe = _ray_pe(H, W, image.device).expand(B, -1, -1, -1)
        if image.dtype != torch.float32:
            ray_pe = ray_pe.to(image.dtype)

        delta = self.decoder({"hl": hl_feats, "ll": ll_feats}, ray_pe)

        fov_base = torch.full((B,), self._BASE_FOV, device=image.device)
        zero     = torch.zeros(B, device=image.device)
        base     = persp_to_erp_grid(fov_base, zero, zero, H, W, device=str(image.device))
        dense_grid = torch.clamp(base + torch.tanh(delta) * 2.0, -1.0, 1.0)

        with torch.autocast(device_type=image.device.type, enabled=False):
            rigid_grid, params = self.rigid_filter(dense_grid.float(), H, W)

        return {
            "dense_grid": dense_grid,
            "rigid_grid": rigid_grid.to(dense_grid.dtype),
            "params":     params.to(dense_grid.dtype),
        }
