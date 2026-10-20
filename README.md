# Active Sample: sequential guidance for active image measurements

Reproducible code for the five-image **CelebA128 active block-measurement** study. The main comparison separates candidate weighting for **acquisition** from weighting for **reconstruction**, and compares old-candidate guidance with From-scratch Flow.

[Experiment report (English PDF)](docs/experiment_report.pdf) · [Mathematical note](docs/Active_Sample_Math_Note.pdf) · [Implementation and reference details](docs/reproducibility.md)

This is a curated release of the current experiment, not the directory of all earlier pilots. No training is required. Data, pretrained weights, environments, and generated trajectories are downloaded or generated locally and are excluded from Git.

## What is compared?

| View | Next-block acquisition | Reconstruction | Old-candidate guidance |
|---|---|---|---|
| UU | Uniform-weight maximum disagreement | Uniform mean | Yes |
| UW | Same U trajectory as UU | Recursively weighted mean | Yes |
| WU | Recursively weighted maximum disagreement | Uniform mean | Yes |
| WW | Same W trajectory as WU | Recursively weighted mean | Yes |
| From-scratch | Uniform-weight maximum disagreement | Uniform mean | No |

**U means uniform candidate weights, not random block selection.** Only three trajectories are generated: `U`, `W`, and `SCRATCH`. U maintains recursive weights for the UW reconstruction view, but those weights never affect UU. WU needs updated weights for acquisition even though reconstruction is uniform.

All updates use the full acquired measurement history. U/W additionally guide each new candidate toward its corresponding old candidate. Every update starts from fresh Gaussian noise; no warm starts, resampling, weight-dependent guidance, or model training are used.

The recursive score multiplies the previous weight by a newest-block Gaussian factor and a full-native-image proximity Gaussian factor. Adaptive inverse temperature scales the **entire** log score, including the previous log weight, choosing the largest value in `[0,1]` with ESS at least 8. This is heuristic scoring, not an exact Bayesian/SMC correction. Generation uses all measurements; measurement scoring uses only the newest block.

## Frozen experiment

- Images: ordinary CelebA official-test filenames `187000`, `183258`, `200793`, `189467`, `193037`; seed `6101`.
- Task: 128×128 RGB, direct nonoverlapping 4×4 pixel blocks, one new block per round. No full low-resolution base image.
- Independent initial→final budgets: **50→100**, **100→200**, **250→500**. Initial blocks are spatially distributed and count toward the budget.
- Prior: frozen official CelebA-HQ **256×256** Rectified Flow EMA checkpoint. A 2×2 average pool maps native images to task128, so the native measurement operator is `A = M D2`.
- `K=16`, 128 sampling steps, sensor noise `sigma=0.05` in `[-1,1]`, native proximity `tau=1`, weighting coefficient `lambda_w=1`, target ESS `8`.
- Five images × three budgets × three policies = **15 shared initializations, 45 trajectories, 6,000 updates**.

GT is used only for simulated sensor readings and evaluation. Acquisition finishes before the selected block's reading is exposed. Model-scale arrays remain unclipped for acquisition, weighting, guidance and metrics.

## 1. Install

Use Linux, Python **3.12**, an NVIDIA CUDA GPU, a CUDA toolkit with `nvcc`, and a C++ compiler. The upstream prior builds its original CUDA FIR operators on first use. CPU-only machines can run unit tests and the numerical preflight, but cannot run image generation.

```bash
git clone https://github.com/poem2018/active-sample.git
cd active-sample
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Example stable installation for a compatible CUDA 12.6 machine:
python -m pip install torch==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The stable install example follows the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v270); choose a torch build for your hardware/toolkit if needed. Use PyTorch **2.6 or newer**: checkpoint loading uses `weights_only=True`, `mmap=True`, and an explicit safe-global mapping. The original runs used Python 3.12, PyTorch `2.7.0.dev20250224+cu126`, NumPy `2.2.3`, Pillow `11.1.0`, and NVIDIA GH200. **torchvision is not required** by this release. Other PyTorch/CUDA/GPU combinations, including the stable install example, are not claimed to be bitwise identical or fully validated. See [reproducibility notes](docs/reproducibility.md).

Run all commands from this checkout's root. No editable/package install is required.

## 2. Obtain data and the frozen prior

Prepare exactly the five recorded images (a pinned public mirror, about 3 MB transferred rather than the full archive):

```bash
python -m experiments.prepare_ads_celeba128
```

Alternatively, use your own original **aligned** CelebA files and official split file:

```bash
python -m experiments.prepare_ads_celeba128 \
  --image-dir /path/to/img_align_celeba \
  --partition-file /path/to/list_eval_partition.txt
```

The helper verifies original-image and preprocessed-array hashes against `configs/dataset_reference.json`. It crops `(0,20,178,198)`, resizes to 128 with Pillow bilinear interpolation, and converts to float32 `[-1,1]`. It does **not** select a new random roster or require older project data. Obtain/use CelebA under its dataset terms.

Download the pinned official Rectified Flow source and approximately **1.05 GB** checkpoint:

```bash
python -m experiments.fetch_official_face_prior
```

It verifies SHA256 and never silently replaces different artifacts. If automatic Google Drive download fails, download the [official checkpoint](https://drive.google.com/file/d/1ryhuJGz75S35GEdWDLiq4XFrsbwPdHnF/view) into `artifacts/prior/celebahq256_rectified_official/checkpoint_10.pth`, then rerun the helper. It also installs the pinned upstream source under `artifacts/external/rectified_flow_official/`.

## 3. Validate and prepare

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m pytest -q
python -m active_sample preflight
python -m active_sample prepare
python -m active_sample tasks
```

Tests cover pooling-adjoint gradients, clean-image correction, exact no-old equivalence to the measurement-only sampler, ESS tempering of the whole score, acquisition, and source provenance (including retained-function equivalence after dead-code cleanup). `prepare` freezes the protocol, initial layouts, paired seeds and sensor banks under `runs/celeba128_ads_blocks_v1/`. Repeating it verifies the existing protocol rather than overwriting it.

## 4. Start with one image and one update

```bash
CUDA_VISIBLE_DEVICES=0 python -m active_sample run \
  --group 50-100 --image 187000 --policy all --max-updates 1
python -m active_sample status
python -m active_sample report --frames 3
```

This generates the shared 50-block initialization once, followed by **one additional update for each** of U, W and From-scratch. It keeps the full K=16/128-step configuration; this is a short execution check, not a cheaper sampler. The initial CUDA extension compilation can take extra time.

Open/download the resulting offline report:

```text
runs/celeba128_ads_blocks_v1/report/ads_celeba128.html
```

Images are embedded; no web server is required. The existing interactive layout includes GT, actual noisy measurements, the five reconstruction views, all 16 candidates, trajectories, acquisition paths and metrics. Unfinished cases stay explicitly incomplete. The viewer labels are currently in Chinese; the README and PDF are English.

## 5. Resume or run the full experiment

Complete the same image/budget:

```bash
CUDA_VISIBLE_DEVICES=0 python -m active_sample run --group 50-100 --image 187000
```

Run all budgets, all five images, all three policies:

```bash
CUDA_VISIBLE_DEVICES=0 python -m active_sample run --group all
python -m experiments.audit_ads_celeba128
python -m active_sample report
```

`--max-updates N` always means **N additional updates per selected trajectory per invocation**. Omit it to finish. Committed initializations/rounds are reused; interrupted uncommitted attempts are retained separately. Do not modify source files after freezing a protocol: source hashes are checked on resume. A modified study should use a separate checkout/output directory.

For available partial results, use `python -m experiments.audit_ads_celeba128 --partial`. Full audit success requires all 15 initializations and 6,000 updates.

### Parallel GPUs / Slurm

Assign disjoint image/budget/policy combinations to independent GPUs. Each process uses **one GPU** and still generates 16 candidates. On the same image/budget, initialize once (the one-update example above is sufficient), then launch separate `--policy U`, `--policy W`, and `--policy SCRATCH` jobs. Duplicate jobs fail a nonblocking file lock rather than sharing writes.

```bash
mkdir -p logs
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  experiments/ads_celeba128.slurm \
  --group 250-500 --image 187000 --policy U --max-updates 25
```

Submit from the repository root with the Python environment activated; resubmit the same command to continue. No site-specific account, node, or user path is embedded in this script.

## Optional baselines: 250→500 only

**Random + From-scratch** uses the same 250-block initial candidates and sensor noise. Initialize the 250-block group for all five images before preparing this baseline (running that group normally does so).

```bash
python -m experiments.ads128_random_baseline prepare
python -m experiments.ads128_random_baseline preflight
# case-index follows the five-image order listed above; repeat for 0 through 4.
CUDA_VISIBLE_DEVICES=0 python -m experiments.ads128_random_baseline infer --case-index 0 --limit 25
python -m experiments.audit_ads128_random_baseline --case-index 0
```

Omit `--limit` to finish; rerun `infer` to resume. `prepare` is a one-time operation.

**ADS-alg/RF-prior** is our separate adaptation, not the unpublished original CelebA128 ADS checkpoint. It uses entropy acquisition and a 600-step guided trajectory with the same frozen RF prior. It does not use our old-candidate guidance or recursive weights.

```bash
python scripts/fetch_ads_reference.py
python -m experiments.ads250_rf prepare
CUDA_VISIBLE_DEVICES=0 python -m experiments.ads250_rf worker \
  --cases 0,1,2,3,4 --chunk 16 --minutes 105
python -m experiments.ads250_rf evaluate --cases 0,1,2,3,4
python -m experiments.ads250_report
```

Rerun the worker until it prints `"done": true`. It saves its main trajectory and deterministic completion branches at 35 budgets; completing those branches costs much more than the final ADS result alone. `--chunk 4` reduces ADS gradient-batch memory without changing K; numerical equality across chunk sizes is not bitwise guaranteed. For **final-only** ADS output, append `--budgets 500`; intermediate curves will remain incomplete.

The combined baseline viewer is `runs/celeba128_ads_blocks_v1/extensions/ads250_v1/report/ads250_comparison.html`. Missing results are reported rather than filled in.

## Metrics and cost

Primary MSE evaluates the **mean reconstruction**, not the average single-candidate error, on unclipped `[-1,1]` arrays. PSNR uses data range 2; SSIM uses a fixed linear conversion to `[0,1]` without clipping. Averages treat each image equally; update-trajectory averages exclude initialization. Different active paths leave different unobserved regions.

The full U/W/From-scratch study took roughly **25 GPU-hours** on GH200, plus setup/evaluation, and needs about **25 GB** of trajectory storage. At 250→500, one policy/image took about **62 minutes**. These are reference measurements, not hardware-independent promises. UW/WW reuse trajectories; this does not imply a sampling speedup.

The five-image results support the complete old-candidate-guided active pipeline over From-scratch; differing acquisition paths prevent attributing the entire improvement solely to guidance. Weighting gains depend on budget. See the PDF for exact comparisons and limitations.

## Code map

| Location | Purpose |
|---|---|
| `active_sample/__main__.py` | Small portable orchestration CLI |
| `experiments/ads_celeba128.py` | Frozen initialization, active loop, metrics, checkpoint/resume |
| `experiments/sequential_flowdps.py` | Full-history + corresponding-old clean-image guidance |
| `experiments/adaptive_candidate_weights.py` | Stable recursive scores and ESS-controlled temperature |
| `experiments/active_blocks.py` | Block geometry and disagreement selection |
| `experiments/official_face_prior.py` | Exact official EMA loading and native time convention |
| `experiments/ads_celeba128_report.py` | Offline HTML and numerical summaries |
| `experiments/ads128_random_baseline.py`, `ads250_rf.py` | Optional random and ADS-RF controls |
| `configs/` | Fixed roster, geometry/settings, source provenance |
| `tests/` | CPU algorithm and packaging checks |

Unused MNIST, R25, new-only-guidance and pixel-ridge helpers have been removed. The active sampler, acquisition, weighting, evaluation, baselines and independent audits are preserved. The current update loop passes all acquired measurements directly to `sequential_flowdps.sample`. See [cleanup provenance](docs/reproducibility.md#dead-code-cleanup) before resuming a run initialized with the earlier release.
