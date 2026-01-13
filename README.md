
# MaleVis Latent Perturbation Classification (VAE → Latent Classifier Robustness)

> **TL;DR** Train a **VAE** on **MaleVis** images, encode samples into **latent space** (`z_mean`), then train a **latent classifier** (e.g., Transformer/RNN/CNN-on-latents) and evaluate robustness under:

> - **Latent noise perturbations** (Gaussian / Uniform / Dropout / Salt–Pepper)

> - **Adversarial attacks via ART** (FGSM / PGD / HopSkipJump / Boundary)


This repository is centered around the notebook:
- `malevis_latent_pertubation_classification.ipynb`

---

## Table of Contents
- [Project Overview](#project-overview)
- [Method Summary](#method-summary)
- [Repository Contents](#repository-contents)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Project Overview
This project investigates **robust malware-family classification** by learning compact representations using a **Variational Autoencoder (VAE)** and performing classification in the **latent space**. The focus is on how **latent-space perturbations** (random and adversarial) affect downstream classifier performance.

---

## Method Summary
**Pipeline**
1. Load MaleVis images from `train/` and `val/` directories (stable label mapping based on train classes).

2. Train an image **VAE** with **KL warmup**.

3. Encode images → deterministic latent vectors (`z_mean`).

4. Train latent classifier (e.g., **Latent Transformer**) on `z_mean`.

5. Evaluate:

   - Clean latent performance

   - Robustness under **latent noise** (Gaussian/Uniform/Dropout/Salt–Pepper)

   - Robustness under **ART adversarial attacks** (FGSM/PGD/HSJ/Boundary)

---

## Repository Contents



.
├── malevis_latent_pertubation_classification.ipynb

├── outputs/

│   ├── latent_dim_sweep_results.csv

│   └── malevis_latent_transformer_noise_adv_results.csv

├── data/                     

├── requirements.txt          

└── README.md

```

---

## Dataset
This notebook expects a directory with `train/` and `val/` subfolders (Keras-style):
```

DATA_ROOT/
├── train/

│   ├── class_1/

│   ├── class_2/

│   └── ...

└── val/

├── class_1/

├── class_2/

└── ...

````

In the notebook, update:

DATA_ROOT = r"/path/to/malevis_train_val_300x300"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")


> Note: The notebook resizes images to a configurable `IMG_H x IMG_W` (default used in the code is often `128x128` for efficiency).

---

## Installation

### Option A — pip (recommended)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -U pip
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
pip install adversarial-robustness-toolbox
```

### Option B — requirements.txt

Create `requirements.txt`:

```txt
tensorflow
numpy
pandas
matplotlib
scikit-learn
jupyter
adversarial-robustness-toolbox
```

Then:

```bash
pip install -r requirements.txt
```

---

## Quick Start

1. **Open the notebook**

```bash
jupyter notebook
```

2. Open: `malevis_latent_pertubation_classification.ipynb`
3. Set `DATA_ROOT` to your dataset path
4. Run cells top-to-bottom

### Optional: Run headless

```bash
jupyter nbconvert --to notebook --execute --inplace malevis_latent_pertubation_classification.ipynb
```

---

## Outputs

The notebook writes CSV summaries (names may vary depending on which sections you execute):

* **Latent dimension sweep**

  * `latent_dim_sweep_results.csv`
  * Includes clean/robust averages, reconstruction loss, KL, runtime, etc.

* **Noise + adversarial robustness**

  * `malevis_latent_transformer_noise_adv_results.csv`
  * Includes Accuracy/Macro-F1 under:

    * Clean
    * Gaussian / Uniform / Dropout / Salt–Pepper (latent noise)
    * FGSM / PGD / HopSkipJump / Boundary (ART attacks)

---

## Reproducibility

* Fixed seeds (`SEED = 42`) for NumPy + TensorFlow
* Deterministic evaluation uses **`z_mean`** (not sampled `z`) for stability
* Validation label mapping follows training `class_names` to prevent label mismatch

---

## Troubleshooting

### ART import errors

If you hit ART errors during adversarial evaluation:

```bash
pip install adversarial-robustness-toolbox
```

Then **restart your kernel/runtime** (required by the notebook).

### GPU memory issues

* Reduce `BATCH_SIZE`
* Reduce `IMG_H, IMG_W`
* Ensure TF GPU memory growth is enabled (the notebook includes a helper for this)

---

## Citation

If you use this code in academic work, cite the repository and the dataset source(s).

Suggested BibTeX (edit the fields):

```bibtex
@misc{malevis_latent_robustness,
  title        = {MaleVis Latent Perturbation Classification},
  author       = {Bamidele Ajayi},
  year         = {2026},
  }
```

---

## License

Choose a license and add `LICENSE` (e.g., MIT, Apache-2.0). If unsure, MIT is a common default for research code.

---

## Contributing

PRs welcome. For major changes, open an issue first to discuss the proposal.

---

**Maintainer:** <abamidele>


```
```
