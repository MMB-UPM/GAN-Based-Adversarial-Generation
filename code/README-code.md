# `code/` — AdvGAN / MalGAN training pipeline (entrypoint: `run_experiment_alb_v2.sh` → `AdvGAN-FINAL_alb.py`)

This folder contains the **executable pipeline** to train and evaluate an adversarial generator (AdvGAN / MalGAN) on pre-scaled tabular datasets (`data2_scaled.npy`), measuring success against a **Black-Box (BB)** classifier (RF or NN) and saving models/metrics/plots.

---

## 1) Files in this folder

- `run_experiment_alb_v2.sh`  
  Bash launcher: activates conda env and runs the main Python script while tee-ing stdout/stderr into log files.

- `AdvGAN-FINAL_alb.py`  
  **Main entrypoint**: parses CLI args, sets GPU visibility, loads dataset + YAML hyperparameter combination, builds **Generator/Discriminator**, instantiates `AdvGAN`, trains, and saves artifacts.

- `advgan_alb_solo_noise.py`  
  Core training implementation: `class AdvGAN(tf.keras.Model)` with a custom training loop (`train()`), BB-hit evaluation, distances/plots, and saving artifacts.

- `model_constructor_alb.py`  
  Model builders + BB inference helpers:
  - `build_discriminator_from_model_complexity(...)`
  - `build_generator_from_model_complexity(...)`
  - `predict(...)` (RF/NN BB prediction)
  - `compute_metrics(...)` (distances, etc.)

- `muestras.py`  
  Dataset loader + sampler: `class Muestras` loads `data2_scaled.npy` / `labels2.npy`, supports class-based sampling, and `inverse_normalise(...)` using a stored scaler (`scaler_*.pkl`).

- `distancias.py`  
  Distance functions (Euclidean + OT/Wasserstein/Sinkhorn via POT + `ot_tf`), plus utilities.

- `smirnov_activation.py`  
  `class SmirnovActivation`: builds per-feature spline-based transforms from a sample of real data and provides `custom_fs` used as generator output activations (when enabled).

---

## 2) How the code is called (call graph + runtime flow)

### 2.1 Call graph (modules)

```text
run_experiment_alb_v2.sh
└── python AdvGAN-FINAL_alb.py
    ├── import muestras as mu
    │   └── mu.Muestras(...) loads:
    │       ./dataset/<dataset>/data2_scaled.npy
    │       ./dataset/<dataset>/labels2.npy
    │       ./dataset/<dataset>/scaler_<dataset>_<MaxMin|Standard>.pkl
    ├── import yaml  → loads: ./Combinations/Combination_<combid>.yaml
    ├── import smirnov_activation as sa
    │   └── (optional) sa.SmirnovActivation(...).create(XX_train) → builds sa.custom_fs
    ├── import model_constructor_alb as mc
    │   ├── mc.build_discriminator_from_model_complexity(...)
    │   ├── mc.build_generator_from_model_complexity(..., custom_activation=sa.custom_fs, solo_noise=True)
    │   ├── mc.predict(...)  → loads BB model from ./bb_models/...
    │   └── mc.compute_metrics(...) → uses distancias.py
    ├── import distancias as dist
    └── import advgan_alb_solo_noise as ag
        └── ag.AdvGAN(...) uses:
            - mu.muestras.sample_examples(...)
            - mu.muestras.inverse_normalise(...)
            - mc.predict(...) (BB inference)
            - distancias.py (distances / OT)
            - saves plots (incl. plots_hists_debug)
````

### 2.2 Runtime flow (high level)

1. `run_experiment_alb_v2.sh` activates conda env and launches `AdvGAN-FINAL_alb.py` with the provided arguments.
2. `AdvGAN-FINAL_alb.py`:

   * sets GPU visibility (via `--gpu_id`)
   * selects a feature subset depending on `--dataset`
   * instantiates `mu.muestras = mu.Muestras(...)`
   * loads `Combinations/Combination_<combid>.yaml`
   * (optional) builds Smirnov activation functions from real malicious samples
   * builds discriminator/generator with `model_constructor_alb.py`
   * instantiates `ag.AdvGAN(...)`, sets weights/ratios from YAML, trains, and saves artifacts
3. `advgan_alb_solo_noise.py` performs the training loop, computes BB hits and distance metrics, and saves arrays + plots.

---

## 3) How to run

From inside `code/`:

```bash
cd code
bash run_experiment_alb_v2.sh <server_id> <exp> <dataset> <modeltype> <modelsize> <combid> <epochs> <archNN> <reescritura>
```

Example:

```bash
cd code
bash run_experiment_alb_v2.sh go1 0 ctu nn small 000 500 y n
```

Notes:

* The launcher currently **does not pass** `--archNN` to Python (the line is commented). `AdvGAN-FINAL_alb.py` will therefore use its default `--archNN y`.
* Training is started with `AdvGAN-FINAL_alb.py`, which internally calls `model.train(epochs=<EPOCHS>, batch_size=512, train_gen=True)`.

---

## 4) `run_experiment_alb_v2.sh` parameters + logs

Arguments (positional):

1. `server_id` : `go1|go2|...` (sets `CUDA_VISIBLE_DEVICES` locally in the script)
2. `exp` : `0|1|2|3` (experiment id forwarded to Python)
3. `dataset` : `ctu|crypto|syn|adult` (forwarded to Python)
4. `modeltype` : `rf|nn` (forwarded to Python; Python uses `RF|NN`)
5. `modelsize` : `small|large` (forwarded to Python)
6. `combid` : combination id (e.g., `000`) → loads `Combinations/Combination_<combid>.yaml`
7. `epochs` : number of epochs (forwarded to Python)
8. `archNN` : currently **not forwarded** (Python default is used)
9. `reescritura` : controls moving previous logs/outputs to `old/` or `borrar/`

Logs produced by the launcher:

* `./<dataset>_output_logs/EXP_<exp>_<dataset>_<modeltype>_<modelsize>/..._output.log`
* `./<dataset>_output_logs/EXP_<exp>_<dataset>_<modeltype>_<modelsize>/..._error.log`

The script also creates legacy folders:

* `./<dataset>_output/...`
* `./plots_hists_debug/...`

(See Section 8 for the Python-side output layout, which uses a different base directory.)

---

## 5) `AdvGAN-FINAL_alb.py` CLI arguments

Direct invocation (without the `.sh` launcher):

```bash
cd code
python AdvGAN-FINAL_alb.py \
  --exp 0 \
  --dataset ctu \
  --modeltype nn \
  --modelsize small \
  --combid 000 \
  --epochs 500 \
  --archNN y \
  --model_gan advgan \
  --gpu_id 0 \
  --es_wgan gan
```

Arguments (defaults are those in the script):

* `--exp` (str, default `"0"`)
* `--dataset` (str, default `"ctu"`)
* `--modeltype` (str, default `"rf"`) → converted to `RF` or `NN` internally
* `--modelsize` (str, default `"small"`)
* `--combid` (str, default `"000"`)
* `--epochs` (int, default `500`)
* `--archNN` (str, default `"y"`)
* `--model_gan` (str, default `"advgan"`) → `advgan|malgan`
* `--gpu_id` (str, default `"0"`)
* `--es_wgan` (str, default `"gan"`) → `gan|wgan` (WGAN enables critic loss + gradient penalty)

Feature selection by dataset (hardcoded in `AdvGAN-FINAL_alb.py`):

* `syn`   → `[0, 1, 2, 3]`
* `ctu`   → `[0, 3, 4, 5, 6, 7, 8]`
* `crypto`→ `[0, 1, 2, 3]`
* `adult` → `[0, 1, 2, 3, 4, 5, 7, 8, 9]`

Scaler type by dataset (hardcoded):

* `syn|ctu|crypto` → `MaxMin`
* `adult` → `Standard`

---

## 6) Experiments (`--exp`: 0..3)

`AdvGAN-FINAL_alb.py` defines 4 variants:

* `0`: `ACT_SMIRNOV_G = False`, `DIST_G = False`
* `1`: `ACT_SMIRNOV_G = True`,  `DIST_G = False`
* `2`: `ACT_SMIRNOV_G = False`, `DIST_G = True`
* `3`: `ACT_SMIRNOV_G = True`,  `DIST_G = True`

Meaning:

* **Smirnov activation** (GEN output transform) is enabled when `ACT_SMIRNOV_G=True`:

  * Samples real malicious data via `mu.muestras.sample_examples(batch_size=0, class_label=mu.MALIGN)`
  * Builds `sa.SmirnovActivation(...).custom_fs`
  * Passes it into `mc.build_generator_from_model_complexity(..., custom_activation=custom_fs, solo_noise=True)`
* **Distance loss** is only allowed when `DIST_G=True`:

  * The script enforces `RATIO_DIST_G == 0` when `DIST_G=False`
  * `UMBRAL_DIST_ALB > 0` implies “ALB distance mode” is active and must match `(RATIO_DIST_G > 0)`

---

## 7) Required inputs (expected folder layout)

All paths are relative to the **current working directory** (usually `code/` when running from here). The following folders/files are expected to exist:

### 7.1 Dataset

`./dataset/<dataset>/`

* `data2_scaled.npy`  (scaled features)
* `labels2.npy`       (labels; benign=0, malign=1)
* `scaler_<dataset>_<MaxMin|Standard>.pkl` (loaded by `muestras.py`)

### 7.2 Hyperparameter combinations (YAML)

`./Combinations/Combination_<combid>.yaml`

Keys used by `AdvGAN-FINAL_alb.py`:

* `DISC_NN` (list[int])  → discriminator dense-layer sizes
* `GEN_NN` (list[int])   → generator dense-layer sizes
* `GP_WGAN` (float)      → gradient penalty weight (set to `0.0` automatically when `--es_wgan gan`)
* `RATIO_LOSS_G`, `RATIO_REG_G`, `RATIO_DIST_G`
* `RATIO_LOSS_D`, `RATIO_REG_D`
* `ALPHA_DISTILLED_LOSS`, `BETA_SAMPLE_DISTANCE`
* `UMBRAL_DIST_ALB` (float)
* optional: `STOCHASTIC`, `TIPO_DISTANCIA`

### 7.3 Black-Box (BB) models and BB data

`AdvGAN-FINAL_alb.py` expects (based on `--modeltype`, `--modelsize`, `--dataset`):

* BB train/test arrays:

  * `./bb_data/<BB_MODEL_TYPE>/<BB_MODEL_TYPE>_BB_<dataset>_<modelsize>_train_dataset.npy`
  * `./bb_data/<BB_MODEL_TYPE>/<BB_MODEL_TYPE>_BB_<dataset>_<modelsize>_train_labels.npy`
  * `./bb_data/<BB_MODEL_TYPE>/<BB_MODEL_TYPE>_BB_<dataset>_<modelsize>_test_dataset.npy`
  * `./bb_data/<BB_MODEL_TYPE>/<BB_MODEL_TYPE>_BB_<dataset>_<modelsize>_test_labels.npy`

* BB model:

  * `./bb_models/<BB_MODEL_TYPE>/<BB_MODEL_TYPE>_BB_<modelsize>_<dataset>.<pkl|h5>`

    * `NN` → `.h5`
    * `RF` → `.pkl`

* Distilled BB model (optional but referenced by path):

  * `./bb_models/<BB_MODEL_TYPE>/<BB_MODEL_TYPE>_BB_<modelsize>_<dataset>_distilled.h5`

* Test results JSON (referenced by path):

  * `./bb_test_results/<BB_MODEL_TYPE>/<BB_MODEL_TYPE>_BB_<dataset>_<modelsize>_test_results.json`

Additionally, the script defines:

* `./bb_models/NN/NN_BB_<modelsize>_<dataset>.h5` (referenced as `NN_BB_MODEL_PATH`)

### 7.4 Other folders referenced by path

* `./dataset_info/` (path is defined in the main script)

---

## 8) Outputs (what is written and where)

### 8.1 Python outputs (main training artifacts)

`AdvGAN-FINAL_alb.py` sets:

* `EXP_NAME = EXP_<exp>_<model_gan>_<dataset>_<BB_MODEL_TYPE>_<modelsize>`
* `OUT_DIR  = ./{model_gan}_{dataset}_output/`

Artifacts are written under:

`./{model_gan}_{dataset}_output/{EXP_NAME}/{combid}/`

Inside, `advgan_alb_solo_noise.py` (`class AdvGAN`) creates (among others):

* `GAN_models/` (saved generator/discriminator models)
* `metrics/`, `distances/`, `samples/`, `preds/`
* `plots/bb_hits/` and other plot subfolders

Additionally, debug histogram plots are saved to:

`./plots_hists_debug/{EXP_NAME}/{combid}/{it}/...png`

### 8.2 Launcher logs

As in Section 4:

* `./<dataset>_output_logs/EXP_<exp>_<dataset>_<modeltype>_<modelsize>/..._output.log`
* `./<dataset>_output_logs/EXP_<exp>_<dataset>_<modeltype>_<modelsize>/..._error.log`

### 8.3 Legacy folders created by the launcher

The launcher also creates/moves:

* `./<dataset>_output/...`
* `./plots_hists_debug/...`

(These paths do not match the Python-side `{model_gan}_{dataset}_output/...` base directory; the Python code writes to its own structure described above.)

---

## 9) Dependencies (Python packages + local modules)

### 9.1 Python packages imported by this pipeline

Minimum packages required by the code in this folder:

* `tensorflow` (GPU optional)
* `numpy`
* `scikit-learn`
* `matplotlib`
* `pandas`
* `seaborn`
* `pyyaml`
* `scipy`
* `tensorflow_probability`
* `POT` (`ot`)
* `ot_tf` (imported by `distancias.py` as `from ot_tf import dmat as dmat_tf, sink as sink_tf`)

### 9.2 Local modules referenced

* `clustering` (required by `advgan_alb_solo_noise.py`: `from clustering import KMeansHelper`)
* `ot_tf` (required by `distancias.py`)

### 9.3 Short runtime notes (directly implied by the code)

* `run_experiment_alb_v2.sh` sets `CUDA_VISIBLE_DEVICES=...` but does not `export` it; GPU selection is reliably controlled in Python via `--gpu_id`.
* `muestras.py` and other modules reference `pause()` in some error branches; a successful run typically avoids those branches.
* In `model_constructor_alb.py`, the RF branch of `predict(...)` assigns `last_bb_model_path=bb_mode_path` (typo). If you run with `--modeltype rf` and hit that branch, it will raise a `NameError` unless corrected.

