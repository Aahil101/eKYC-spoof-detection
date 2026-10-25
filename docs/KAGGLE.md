# Kaggle workflow

All training runs on Kaggle (T4/P100, about 30 GPU-h/week, 12 h per session). Datasets are **attached to notebooks online**, so nothing large is downloaded to the laptop. The laptop only runs CPU smoke tests and the demo app.

## 0. Status of account `aahil1011` (done 2026-10-26)

| Item | Kaggle | State |
|---|---|---|
| Source dataset | https://www.kaggle.com/datasets/aahil1011/codmav-src | v2 (git `27ff19c`) |
| Selftest notebook | https://www.kaggle.com/code/aahil1011/codmav-selftest | COMPLETE; output imported as bundle `0.0.1+8b7f7c43b6`, round trip verified (Kaggle image: Python 3.13) |
| SIDTD fetch | https://www.kaggle.com/code/aahil1011/codmav-fetch-sidtd-templates | COMPLETE; 1.27 GB from original host, 3,454 files |
| Dataset audit | https://www.kaggle.com/code/aahil1011/codmav-dataset-audit | see `docs/DATASET_AUDIT.md` |

The CLI authenticates from `%USERPROFILE%\.kaggle\access_token`, outside the repo. On Windows, pass **absolute** paths to `kaggle datasets create/version -p` and `kaggle kernels push -p`. Relative paths with `/` fail in kaggle 2.2.5.

Routine loop after code changes:
```
git commit ...
python -m training.package_src --owner aahil1011
kaggle datasets version -p <ABS>\dist\codmav-src -m "<git short sha>"
python -m training.make_notebook --config training/configs/<job>.yaml --owner aahil1011
kaggle kernels push -p <ABS>\kaggle\build\<run_name>
kaggle kernels output aahil1011/codmav-<run-slug> -p downloads\<x> --file-pattern "runs/.*/(artifacts/.*|run_manifest.json)"
python -m training.import_weights --run-dir downloads\<x>\runs\<run_name> --module Mx --bundle <name>
```

## 0b. First run walkthrough (kept for reference)

The laptop has already prepared `dist/codmav-src/` (source dataset) and `kaggle/build/selftest/` (notebook plus metadata). The selftest notebook was executed in a Kaggle-like Docker container (`scripts/simulate_kaggle.py`) and finished with `status: completed`.

1. Put `kaggle.json` in `%USERPROFILE%\.kaggle\` (step 1.1 below) and run `pip install kaggle`.
2. Upload the code as a private dataset: `kaggle datasets create -p dist/codmav-src`. It appears as `aahil1011/codmav-src`. Later updates: `python -m training.package_src --owner aahil1011` then `kaggle datasets version -p dist/codmav-src -m "update"`.
3. Push the selftest notebook: `kaggle kernels push -p kaggle/build/selftest`. It runs on CPU with internet on. Watch it at https://www.kaggle.com/code/aahil1011/codmav-selftest.
4. When it finishes: `kaggle kernels output aahil1011/codmav-selftest -p downloads/selftest`, then `python -m training.import_weights --run-dir downloads/selftest/runs/selftest --module M2 --bundle selftest-kaggle --semver 0.0.1`. That is the same round trip already verified locally. Afterwards delete `models/bundles/selftest-kaggle`; it is only a plumbing test.

Without the CLI: create a dataset in the Kaggle UI and upload `dist/codmav-src/codmav_src.zip`. Then create a notebook, *File → Import Notebook* `kaggle/build/selftest/selftest.ipynb`, *Add Input* `codmav-src`, and choose *Save Version → Save & Run All*.

To test any generated notebook locally before spending quota: `python scripts/simulate_kaggle.py kaggle/build/<run> --src-zip dist/codmav-src/codmav_src.zip`.

## 1. One-time setup

1. Create an API token at kaggle.com → Settings → API → *Create New Token*, and save `kaggle.json` to `%USERPROFILE%\.kaggle\kaggle.json`. **Never commit it**; it is listed in `.gitignore`.
2. `pip install kaggle` (laptop, optional; it is only needed to push notebooks or fetch small subsets).
3. Accept the rules of the DFDC competition on its Kaggle page if you will use `dfdc`.
4. Verify the Kaggle phone number. GPU and internet in notebooks require it.

## 2. Two run modes, same code

| | smoke (laptop CPU) | full (Kaggle notebook) |
|---|---|---|
| command | `python -m training.run --config training/configs/<job>.yaml --mode smoke` | generated notebook runs `--mode full --resume auto` |
| data | tiny synthetic data or a capped subset in `data/raw/` | attached datasets under `/kaggle/input/` |
| outputs | `./runs/<run_name>/` | `/kaggle/working/runs/<run_name>/` |

Each job YAML has `params` plus `modes.smoke` / `modes.full` overrides, and also `checkpoint`, `budget.max_wall_minutes` (default 660 = 11 h, so the job stops cleanly before Kaggle kills it), `data.datasets` (logical names from `training/datasets.yaml`) and a `kaggle` block (GPU, internet, pip packages, weight datasets).

Only `selftest.yaml` is implemented in milestone 2. It is a numpy logistic regression on synthetic data that exists to test the plumbing. The M1/M2/M3 and SBI configs are present with `implemented: false` and exit with a clear message until milestone 3.

## 3. Attaching datasets (no laptop storage needed)

`training/datasets.yaml` maps logical names to Kaggle sources:

| name | Kaggle source | status |
|---|---|---|
| `ffpp_c23` | https://www.kaggle.com/datasets/xdxd003/ff-c23 | page found; check contents and size against the original before relying on it |
| `dfdc` | https://www.kaggle.com/c/deepfake-detection-challenge (competition) | official host |
| `celebdf_v2`, `celeba_spoof`, `df40`, `sidtd` | **not yet verified** | pick a mirror (below) |

To verify a mirror: search Kaggle and open the dataset page. Check that the file layout matches `expected_layout` in `datasets.yaml` and that the size and file counts match the original paper. Then set `kaggle_ref: owner/slug` and `verified: true`. If there is no good mirror for **SIDTD** (CC-BY-4.0), you may upload it yourself as a *private* Kaggle dataset with attribution. Do this from a Kaggle notebook with internet on by downloading from the original URL into `/kaggle/working` and saving the output as a dataset, so it never touches the laptop.

**License rule:** a Kaggle mirror does not change the original terms. The original license is listed in `datasets.yaml` and `RESEARCH_NOTES.md`, and the paper cites the original dataset paper.

## 4. Internet-off notebooks: assets to create once

Some jobs set `internet: false` (more reproducible and required for some competitions). Packages and pretrained weights must then come from Kaggle datasets that you own. Create each one from a helper notebook with **internet ON**, then *Save Version → Output → New Dataset*:

| dataset slug | how to build (in a Kaggle notebook, internet on) | used by |
|---|---|---|
| `codmav-src` | on laptop: `python -m training.package_src --owner <you>` then `kaggle datasets create -p dist/codmav-src` (later: `kaggle datasets version -p dist/codmav-src -m "msg"`) | all offline jobs |
| `codmav-wheels` | `pip download -d /kaggle/working/wheels -r requirements/kaggle-extra.txt` | all offline jobs |
| `codmav-weights-openclip-vitb16` | `import open_clip; open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', cache_dir='/kaggle/working/openclip')` | M2 |
| `codmav-weights-mediapipe` | download `face_landmarker.task` from the MediaPipe model page (Apache-2.0) into `/kaggle/working` | M4, M5, quality |
| `codmav-weights-insightface` | `insightface.app.FaceAnalysis(name='buffalo_l', root='/kaggle/working/insightface').prepare(ctx_id=0)` — **non-commercial research only** | M5 |
| `codmav-weights-facenet` | `facenet_pytorch.InceptionResnetV1(pretrained='vggface2')`, then copy `~/.cache/torch/checkpoints/*` | M5 alternative |
| `codmav-weights-vosk` | download `vosk-model-small-en-us-0.15.zip` from alphacephei.com/vosk/models and unzip | M4 read-digits |
| `codmav-weights-timm-effnetb4` | `timm.create_model('efficientnet_b4', pretrained=True)`, then copy the HF cache | SBI plug-in only |

`requirements/kaggle-extra.txt` lists the pip packages that are not preinstalled on the Kaggle image. torch, torchvision, numpy, scikit-learn and opencv are preinstalled. Versions get pinned in milestone 3, after the jobs run on Kaggle.

## 5. Generate and push a notebook

```
python -m training.make_notebook --config training/configs/m2_clip_probe.yaml --owner <you> \
       --repo-url https://github.com/<you>/codmav.git --ref main
kaggle kernels push -p kaggle/build/m2_clip_probe_ffpp_c23
```

This writes `<run_name>.ipynb` and `kernel-metadata.json`, which set GPU, internet, `dataset_sources` and `competition_sources` from the config. Unverified datasets are reported as warnings; add them in the Kaggle UI with *Add Input*. You can also upload the `.ipynb` manually and attach inputs in the UI.

The notebook gets the code from the attached `codmav-src` zip, or from `git clone` if internet is on. It installs the extras, exposes the weight dirs through `CODMAV_WEIGHTS_DIRS`, and runs the job in full mode.

## 6. Checkpoints and resuming after a dead session

- Checkpoints are written atomically every `every_steps` steps (on step multiples) and every `every_minutes` minutes, to `/kaggle/working/runs/<run>/checkpoints/`. The last `keep_last` are kept, along with `latest.json`. They include the RNG state, so a resumed run reproduces an uninterrupted one exactly; this is tested.
- If the session dies, `/kaggle/working` of a *committed* version survives. Open the notebook, use **Add Input → Your Work → (this notebook) → previous version**, and Run All. `--resume auto` searches `/kaggle/working` and every attached input for `runs/<run_name>/checkpoints` and continues from the highest step. Corrupt or truncated checkpoint files are skipped.
- Use *Save Version → Save & Run All (Commit)* for long jobs. Interactive sessions lose `/kaggle/working` when they stop.
- If the wall-clock budget runs out, `run_manifest.json` has `status: incomplete_budget`. Commit again with the previous output attached to continue.

## 7. Bringing trained weights back into `models/`

Download only the small artifacts, not the checkpoints:

```
kaggle kernels output <you>/codmav-m2-clip-probe-ffpp-c23 -p downloads/m2
python -m training.import_weights --run-dir downloads/m2/runs/m2_clip_probe_ffpp_c23 \
       --module M2 --bundle kyc-v0.3 --semver 0.3.0
```

`import_weights` checks that the run status is `completed` and that every artifact hash matches `run_manifest.json`, so it catches corrupted downloads. It then copies the artifacts to `models/bundles/<bundle>/<module>/`, records the training run in `manifest.yaml` (git sha, config sha256, data, metrics, Kaggle kernel) and recomputes `bundle_version = <semver>+<hash>`. Any change to weights, calibrators, fusion or thresholds therefore gives a new version. Serve a bundle with `MODEL_BUNDLE=kyc-v0.3` in `.env`. Every session, drift report and alert stores the version.

## 8. Optional: small local subset

```
python -m training.fetch_subset --dataset ffpp_c23 --pattern "*.mp4" --max-files 10
```

This needs `kaggle.json`. Downloads are capped by `max_local_mb` per dataset (0 disables them) and go into `data/raw/` (git-ignored). Competition data is never fetched locally.
