# MoVerse: Real-Time Video World Modeling with Panoramic Gaussian Scaffold

<div align="center">
  <img src="asset/moverse_icon2.png" alt="MoVerse" width="80%"/>
</div>

<div align="center">
  <a href="https://orange-3dv-team.github.io/MoVerse"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://arxiv.org/abs/2606.13376"><img src="https://img.shields.io/static/v1?label=paper&message=arxiv&color=red"></a> &ensp;
</div>

## 📢 About

**TL;DR:** We propose MoVerse, a method that builds a navigable 3D world from a single narrow-field-of-view image and renders interactive video in real time on a single RTX 4090. MoVerse separates world construction from observation rendering through a three-stage pipeline, producing coherent and photorealistic free-roaming video walkthroughs.

**Features:** MoVerse generates high-quality real-time navigable worlds from a single input image, supporting:
- **Panoramic Generation**: Expands a single NFOV image into a gravity-aligned, horizontally periodic 360° ERP panorama via topology-aware latent diffusion
- **3D Gaussian Scaffold**: Lifts the panorama into a panoramic 3D Gaussian scaffold using feed-forward residual prediction in angular–inverse-depth space
- **Autoregressive Video Refinement**: Translates scaffold renderings along user-specified camera trajectories into photorealistic video at **8 FPS on a single RTX 4090**
- **Free-Roaming Walkthrough**: Supports user-controlled camera trajectories with temporally coherent real-time rendering
- **Diverse Scenes**: Works across indoor and outdoor scenes — from ancient ruins to cyberpunk streets, anime landscapes to realistic interiors

## 📅 Open Source Plan

The code and pretrained models are currently under corporate compliance and security review. We anticipate the review will take about a month, and we will release all materials promptly upon approval.

## 🔗 Links

- 🌐 **Project Page**: [https://orange-3dv-team.github.io/MoVerse](https://orange-3dv-team.github.io/MoVerse)
- 📄 **Paper**: [arXiv:2606.13376](https://arxiv.org/abs/2606.13376)
- 💻 **Code**: [GitHub](https://github.com/Orange-3DV-Team/MoVerse)
- 🤗 **Model**: Coming Soon

## 🏗️ Pipeline

Given a single narrow-field-of-view image, MoVerse separates world construction from observation rendering:

```
INPUT (NFOV image)
  → STAGE I:   Panoramic Generation (single image → 360° ERP)
  → STAGE II:  3D Gaussian Scaffold (panorama → splattable 3D asset)
  → STAGE III: Autoregressive Video Refinement (scaffold → real-time roaming video)
OUTPUT (navigable world)
```

<div align="center">
  <img src="asset/pipeline_overview.png" alt="MoVerse Pipeline Overview" width="100%"/>
</div>

## 🎥 Preview

Visit our [Project Page](https://orange-3dv-team.github.io/MoVerse) to view complete demo videos, interactive 360° panorama viewers, and 3D Gaussian scaffold visualizations.

## 📖 Citation

```bibtex
@article{moverse2026,
  title   = {MoVerse: Real-Time Video World Modeling with Panoramic Gaussian Scaffold},
  author  = {Yang Zhou and Ziheng Wang and Yuqin Lu and Haofeng Liu and Jun Liang and Shengfeng He and Jing Li},
  journal = {arXiv preprint arXiv:2606.13376},
  year    = {2026}
}
```

## ⭐ Star

If you're interested in this project, please give us a Star ⭐ to receive timely open-source notifications!

---

<div align="center">
  <sub>Built with ❤️ by Orange Team, Youku Moku-Lab, HUJING Digital Media & Entertainment Group</sub>
</div>