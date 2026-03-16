# Datasets for *Balancing Realism and Evasion in GAN-Based Adversarial Generation*

This repository contains the datasets used in the paper:

**Balancing Realism and Evasion in GAN-Based Adversarial Generation**

The experiments in the paper use two complementary benchmarks:

1. **`datasets/syn/`**: a controlled **simulated parametric dataset**
2. **`datasets/crypto/`**: a **real cryptomining traffic dataset**

The goal of releasing these datasets is to support **transparency, reproducibility, and further research** on adversarial generative modeling, realism--evasion trade-offs, and evaluation methodologies for synthetic adversarial data.

---

## General format

Across both benchmarks:

- the task is **binary classification**
- labels follow the convention:
  - **0** = benign
  - **1** = malicious
- each sample contains **4 features**
- datasets are provided both in:
  - **raw form** (`data*.npy`)
  - **scaled form** (`data*_scaled.npy`), where each feature has been independently rescaled to the interval **[0,1]**
- labels are stored separately in `labels*.npy`

The scaled versions are the ones used for GAN training in the paper.

---

## Repository structure

```text
datasets/
├── crypto/
│   ├── labels1.npy
│   ├── labels2.npy
│   ├── data1.npy
│   ├── data2.npy
│   ├── data1_scaled.npy
│   ├── data2_scaled.npy
│   └── scaler_crypto_MaxMin.pkl
└── syn/
    ├── data1.npy
    ├── data2.npy
    ├── data1_scaled.npy
    ├── data2_scaled.npy
    ├── labels1.npy
    ├── labels2.npy
    ├── scaler_syn2_MaxMin.pkl
    ├── data1.csv
    └── data2.csv
```

### File naming convention

- **`data1` / `labels1`**: first subset / first trace / first generated block
- **`data2` / `labels2`**: second subset / second trace / second generated block
- **`*_scaled.npy`**: normalized version in **[0,1]**
- **`scaler_*.pkl`**: fitted scaler used for normalization
- **`.csv` files**: human-readable export (provided for the synthetic benchmark)

---

## 1. Simulated parametric benchmark (`datasets/syn/`)

This benchmark is a **fully rendered parametric dataset** designed to provide a controlled testbed for adversarial generation. In the original experimental design, two statistically identical datasets were generated independently (denoted **DS1-r** and **DS2-r**). In this repository, they are stored separately as `data1` / `labels1` and `data2` / `labels2`.

### Main properties

- **2 balanced classes**
- **4 features per sample**
- features are sampled **independently given the class**
- the benchmark combines **continuous and discrete non-Gaussian marginals**
- all features are also provided in a normalized version in **[0,1]**

This dataset was designed so that the classification problem is non-trivial: some features are shared across classes, while others differ in shape more than in mean. This makes the benchmark suitable for evaluating whether a generator captures the **distributional structure** of the data rather than only simple shifts.

### Feature distributions

#### Class 0 (benign)

- **Feature 0**: Normal distribution, mean 0, standard deviation 1
- **Feature 1**: Binomial distribution with `n = 15` and `p = 0.3`
- **Feature 2**: Exponential distribution with scale 3
- **Feature 3**: Poisson distribution with `lambda = 1.0`

#### Class 1 (malicious)

- **Feature 0**: Normal distribution, mean 0, standard deviation 1
- **Feature 1**: Discrete uniform distribution on `[0, 15]`
- **Feature 2**: Snedecor F distribution with `df1 = 3`, `df2 = 3`
- **Feature 3**: Poisson distribution with `lambda = 2.0`

### Files

- `data1.npy`, `data2.npy`: raw feature matrices
- `labels1.npy`, `labels2.npy`: binary labels
- `data1_scaled.npy`, `data2_scaled.npy`: normalized versions in **[0,1]**
- `data1.csv`, `data2.csv`: CSV export of the same benchmark
- `scaler_syn2_MaxMin.pkl`: scaler used to obtain the normalized representation

---

## 2. Real cryptomining benchmark (`datasets/crypto/`)

This benchmark is a **real flow-based network traffic dataset** built from benign traffic and malicious cryptomining traffic.

The traffic was collected in the **Mouseworld** network digital twin, an NFV/SDN-based environment used to emulate realistic Internet conditions. Regular traffic (such as web and video) was generated together with traffic produced by hosts running **cryptomining clients** connected to public mining pools. The result is a binary dataset containing benign and malicious flows.

Two independent traces were captured and are stored separately in this repository as `data1` / `labels1` and `data2` / `labels2`.

### Main properties

- **2 classes**: benign vs. malicious cryptomining flows
- **4 retained features** per sample
- features were selected from a larger set of **59 flow-level statistics**
- the selected marginals are **non-Gaussian**, often **heavy-tailed**, and suitable for evaluating realism-preserving adversarial generation
- normalized versions in **[0,1]** are provided for direct use in GAN training

### Retained features

Only four features were used in the experiments reported in the paper:

1. **Number of bytes sent from the client**
2. **Average round-trip time as observed from the server**
3. **Outbound bytes per packet**
4. **Ratio of inbound packets to outbound packets**

These variables were selected because they combine discriminative power with complex marginal behavior and do not reduce the task to trivial mean-based separation.

### Files

- `data1.npy`, `data2.npy`: raw feature matrices
- `labels1.npy`, `labels2.npy`: binary labels
- `data1_scaled.npy`, `data2_scaled.npy`: normalized versions in **[0,1]**
- `scaler_crypto_MaxMin.pkl`: scaler used to obtain the normalized representation

---

## How to load the data in Python

```python
import numpy as np

# Synthetic benchmark
X_syn_1 = np.load("datasets/syn/data1.npy")
y_syn_1 = np.load("datasets/syn/labels1.npy")
X_syn_1_scaled = np.load("datasets/syn/data1_scaled.npy")

# Cryptomining benchmark
X_crypto_1 = np.load("datasets/crypto/data1.npy")
y_crypto_1 = np.load("datasets/crypto/labels1.npy")
X_crypto_1_scaled = np.load("datasets/crypto/data1_scaled.npy")
```

---

## Recommended usage

- Use **`*_scaled.npy`** to reproduce the training setting described in the paper.
- Use raw **`data*.npy`** files to inspect the original feature distributions.
- Use the `.csv` files in `datasets/syn/` for quick inspection without Python.
- Depending on your protocol, `data1` and `data2` can be:
  - kept separate as independent subsets, or
  - merged into a single dataset.

---

## Reproducibility note

The repository is intended to facilitate the reproducibility of the experiments reported in the paper and to support future work on:

- adversarial generative modeling
- realism-aware synthetic data generation
- optimal-transport-guided GAN training
- Pareto-front-based evaluation of realism vs. evasion

---

## Related paper

If you use this repository, please cite the associated paper:

**Balancing Realism and Evasion in GAN-Based Adversarial Generation**

```bibtex
@article{mozo2026balancing,
  title={Balancing Realism and Evasion in GAN-Based Adversarial Generation},
  author={Mozo, Alberto and Karamchandani, Amit and de la Cal, Luis},
  journal={PeerJ Computer Science},
  year={2026}
}
```

---

## Contact

For questions related to the datasets or the experiments, please contact the authors through the repository or through the paper correspondence details.

