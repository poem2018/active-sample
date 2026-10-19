# Reproducibility and provenance

## What is preserved

`configs/release_provenance.json` records SHA256 values for the original experiment implementation, sampler, prior adapter, weights, metrics, audit and report modules. Those files are copied unchanged and checked by the test suite. The original results and frozen protocols in the research workspace are not modified by this release.

New packaging consists of a small CLI, a generic Slurm launcher, dependency instructions, and fixed-roster data preparation. The original data preparation scanned historical pilot directories to choose unused images. That selection is already complete: the portable helper replays exactly the five recorded filenames and verifies both JPEG and task-array hashes. It never repeats selection against a different filesystem.

The reference settings are in `configs/experiment_reference.json`. Each clone generates its own runtime `runs/celeba128_ads_blocks_v1/protocol.json` with hashes of its actual source files. That runtime protocol therefore differs from the original in packaging/provenance fields, while the candidate generation, geometry, sensor seeds, weighting, and evaluation definitions are preserved. Do not copy an old run's protocol into a changed checkout and bypass its hash checks.

## Prior and references

- Rectified Flow: https://github.com/gnobitab/RectifiedFlow
  - Source revision: `5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6`.
  - Source ZIP SHA256: `070dc68c7f903802f6f6e9db0a9ef9f3f7c61668b69dcdb0c2af4d1e3a6eade8`.
  - [Official CelebA-HQ checkpoint](https://drive.google.com/file/d/1ryhuJGz75S35GEdWDLiq4XFrsbwPdHnF/view).
  - Checkpoint SHA256: `890198b78860ca1eb679d01485231f6fcb47161a215aab4a94de7fd7156d6d00`.
  - The loader installs the saved EMA parameters exactly. Native time runs from 0.001 to 1, with labels `999*t`.
- [FlowDPS paper](https://arxiv.org/abs/2503.08136): inspiration for clean/noise decomposition and clean-image corrections. This code is a pixel-space RF adaptation, not an official SD3 reproduction.
- [ADS official code](https://github.com/active-diffusion-subsampling/ads), revision `64f2889b192cee2280ec7bf1405752b81ec02bc1`: reference for our separately named ADS-alg/RF-prior adaptation. RF schedule, noisy observations, initial measurements, and aligned action grid differ from the published CelebA experiment.
- CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html ; pinned mirror https://huggingface.co/datasets/Yuehao/celeba/tree/35ebb31405e9da028409199a494f2e77eb10f253 .

Third-party source and weights are obtained from their upstream locations, with their original notices/terms. They are not redistributed in this repository. No new license grant for the original research code is implied by this private collaboration release.

## Numerical environment

Original environment: Python 3.12, PyTorch `2.7.0.dev20250224+cu126`, torchvision `0.22.0.dev20250224`, NumPy `2.2.3`, Pillow `11.1.0`. The current validation environment additionally has SciPy `1.17.1`, scikit-image `0.26.0`, requests `2.32.3`, ninja `1.13.0`, pytest `8.4.2`. Requirements pin the data-sensitive NumPy/Pillow versions and allow compatible scientific-library releases; this is not a bitwise environment lock.

The frozen loader enables TF32 matrix multiplication and cuDNN benchmarking, as in the original runs. GPU/library differences can alter small numerical errors, and these can change later argmax acquisition choices. Identical seeds are necessary but do not guarantee identical closed-loop trajectories across hardware.

Keep an explicit record of `python -m pip freeze`, GPU model, CUDA toolkit and driver when reproducing. Actual execution metadata is also saved with each trajectory. The upstream CUDA extensions need `nvcc`, a compatible C++ toolchain, and writable build space. Set `CUDA_HOME`/`TORCH_CUDA_ARCH_LIST` only as appropriate for your machine; no GH200-specific architecture is forced in the portable launcher.

CPU checks validate the algebra, selection, metrics and data preparation; they do not validate CUDA extension compilation or actual model output on another machine. The release validation record distinguishes these checks from new GPU inference. The uploaded PDF summarizes the original completed experiments, not newly rerun trajectories from this packaging task.

## Outputs and restart semantics

`runs/celeba128_ads_blocks_v1/inputs/` contains shared truth arrays and fixed sensor banks. Each group/case has one initialization and separate U/W/SCRATCH trajectories. Every committed round contains task candidates, measurement history, scoring weights/temperature, metrics, and numerical diagnostics. Native candidates are retained at initialization, selected checkpoint budgets, the final budget, and the latest rolling state.

The previous rolling native state is deleted only after the next state is committed. An interrupted uncommitted directory is moved to `incomplete_attempts` on restart. Source and input hashes are checked. Do not manually delete checkpoints needed for resume, share writable run directories across modified code versions, or use the same output directory for multiple independent experiments.

Use a separate checkout for a different scientific configuration. The convenience CLI intentionally does not expose K, steps, noise or weighting hyperparameters, so the documented commands preserve the reported design.

## Interpretation

- UW vs UU and WW vs WU isolate reconstruction weighting on identical candidates.
- WU vs UU and WW vs UW compare acquisition policies with subsequent candidate-update feedback.
- UU vs From-scratch compares complete active pipelines; it is not a fixed-history causal ablation of old-candidate guidance alone.
- Initial blocks count toward the measurement budget. The prior's native 256 resolution does not turn this direct-block task into standard whole-image super-resolution.
- The five test images were excluded from this project's tuning. The upstream prior's training-image IDs are unavailable.
- Adaptive temperature prevents severe weight collapse; this alone does not establish posterior accuracy or a reconstruction benefit.
