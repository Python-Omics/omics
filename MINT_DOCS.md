# MINT Documentation

## Overview

**MINT** is a pure Python implementation of the **MINT (Multi-group INtegrative Tensor)** algorithm from the mixOmics R package. It performs multi-omics integration while correcting for batch/study effects, making it ideal for multi-cohort studies.

Unlike DIABLO (which assumes a single homogeneous cohort), MINT explicitly models study/batch structure by computing **per-study loadings** that are then aggregated. This prevents batch effects from contaminating the integrative components.

### Key Features
- **Two algorithmic paths**:
  - **MINT path** (`tau=None`): Per-study loading aggregation (true batch correction)
  - **RGCCA path** (`tau="optimal"` or fixed): Ridge-regularized generalized CCA
- **Sparsity** via `keepA` (exact variable selection) or `penalty` (L1 regularization)
- **Scikit-learn compatible** API (`fit`, `transform`, `fit_transform`)
- **Full visualization integration** with existing pythomics plotting functions

---

## Feature/Implementation Table

| Feature | Implementation |
|---------|----------------|
| **Core algorithm** | Pure Python (`internal_mint_block`) - no R/rpy2 |
| **MINT path** | `tau=None` - per-study loading aggregation (batch correction) |
| **RGCCA path** | `tau="optimal"` or fixed - ridge regularization |
| **Sparsity** | `keepA` dict `{block: [c1, c2]}` or `penalty` dict |
| **Schemes** | horst, factorial, centroid |
| **Modes** | canonical, classic, invariant, regression |
| **API** | `fit()`, `transform()`, `fit_transform()`, `get_biomarkers()`, `get_expression_means()` |

---

## Installation

```bash
# Core dependencies
pip install numpy pandas scikit-learn scipy matplotlib seaborn

# For MuData support (optional)
pip install anndata mudata

# For visualization dependencies
pip install pycirclize networkx adjustText fpdf2

# For AI annotation/reporting (optional)
pip install biothings_client openai python-dotenv
```

---

## Quick Start

```python
import numpy as np
import pandas as pd
from pythomics.models import MINT

# Prepare multi-omics data (samples x features per block)
X_dict = {
    'mRNA': pd.DataFrame(np.random.randn(100, 500), 
                         columns=[f'Gene_{i}' for i in range(500)]),
    'Proteins': pd.DataFrame(np.random.randn(100, 200), 
                             columns=[f'Prot_{i}' for i in range(200)]),
    'Metabolites': pd.DataFrame(np.random.randn(100, 150), 
                                columns=[f'Met_{i}' for i in range(150)])
}

# Study/batch labels (same length as samples)
study = np.array([0]*50 + [1]*50)  # Two cohorts

# Optional: outcome variable for supervised analysis
Y = np.array([0]*25 + [1]*25 + [2]*25 + [3]*25)  # 4 classes

# Initialize and fit
model = MINT(
    n_components=2,
    keepA={'mRNA': [20, 20], 'Proteins': [10, 10], 'Metabolites': [8, 8]},
    tau=None,  # MINT path (batch correction)
    scheme='horst',
    max_iter=200
)

model.fit(X_dict=X_dict, Y=Y, study=study)

# Get selected biomarkers
biomarkers = model.get_biomarkers('mRNA', comp=1)
print(biomarkers.head(10))
```

---

## Constructor Parameters

```python
MINT(
    n_components=2,           # int or list[int]: components per block
    design=None,              # array (J,J): block connection matrix
    keepA=None,               # dict or list: sparsity (exact variable count)
    penalty=None,             # dict or list: L1 penalty (alternative to keepA)
    tau=None,                 # None, "optimal", list, or list of list
    scheme="horst",           # "horst", "factorial", "centroid"
    mode="canonical",         # "canonical", "classic", "invariant", "regression"
    scale=True,               # bool: scale per study
    max_iter=100,             # int: max iterations per component
    tol=1e-6,                 # float: convergence tolerance
    init="svd.single"         # "svd" or "svd.single"
)
```

### Parameter Details

#### `n_components`
Number of components to compute.
- **int**: Same number for all blocks (e.g., `2`)
- **list[int]**: Per-block components (e.g., `[2, 3, 2]` for 3 blocks)

#### `design`
Design matrix specifying block connections (J × J).
- Default: Full design (`1 - I`, all blocks connected)
- Set `design[i, j] = 0` to disconnect blocks i and j

```python
# Custom design: mRNA connects to Proteins and Metabolites,
# but Proteins and Metabolites don't connect directly
design = np.array([
    [0, 1, 1],  # mRNA
    [1, 0, 0],  # Proteins
    [1, 0, 0]   # Metabolites
])
```

#### `keepA` (Sparsity - Exact Variable Selection)
Keep exactly N variables per block per component.
```python
# Dict format (recommended): {block_name: [comp1, comp2, ...]}
keepA = {
    'mRNA': [30, 30],       # 30 genes on comp 1, 30 on comp 2
    'Proteins': [15, 15],   # 15 proteins per component
    'Metabolites': [10, 10] # 10 metabolites per component
}

# List format (internal): keepA[comp][block]
keepA = [[30, 15, 10], [30, 15, 10]]  # 2 components × 3 blocks
```

#### `penalty` (Sparsity - L1 Regularization)
Continuous L1 penalty per block (alternative to `keepA`).
```python
# Dict format
penalty = {'mRNA': 0.5, 'Proteins': 0.3, 'Metabolites': 0.2}

# List format (per block)
penalty = [0.5, 0.3, 0.2]
```
**Note**: Cannot use both `keepA` and `penalty` simultaneously.

#### `tau` (Regularization Path Selection)
Controls which algorithmic path to use:
- **`None` (default)**: **MINT path** - true batch correction via per-study aggregation
- **`"optimal"`**: **RGCCA path** - auto-estimate ridge penalty per block
- **`list[float]`**: Fixed tau per block (length J)
- **`list[list[float]]`**: Tau per component per block (length N × J)

```python
# MINT path (recommended for multi-cohort studies)
model = MINT(tau=None, ...)

# RGCCA path with auto tau estimation
model = MINT(tau="optimal", ...)

# RGCCA path with fixed tau
model = MINT(tau=[0.1, 0.2, 0.15], ...)

# RGCCA path with per-component tau
model = MINT(tau=[[0.1, 0.2, 0.15], [0.1, 0.2, 0.15]], ...)
```

#### `scheme` (Optimization Criterion)
- **`"horst"`** (default): Maximize sum of covariances
- **`"factorial"`**: Maximize sum of squared covariances
- **`"centroid"`**: Maximize sum of absolute covariances

#### `mode` (Deflation Strategy)
- **`"canonical"`** (default): Each block deflated on its own variate
- **`"classic"`**: Y block not deflated; X blocks deflated on average of other variates
- **`"invariant"`**: No deflation (components orthogonal by construction)
- **`"regression"`**: X blocks deflated on average of residuals from other blocks

---

## Methods

### `fit(X_dict, Y=None, study=None)`
Fit the MINT model to multi-omics data.

**Parameters:**
- `X_dict`: `dict[str, array-like]` - Data blocks `{block_name: (n_samples, n_features)}`
- `Y`: `array-like`, optional - Outcome variable for supervised analysis
- `study`: `array-like`, optional - Study/batch labels per sample (length n_samples)

**Returns:** `self`

**Example:**
```python
model.fit(X_dict=omics_data, Y=labels, study=batch_labels)
```

---

### `transform(X_dict)`
Project new data onto fitted MINT components.

**Parameters:**
- `X_dict`: `dict[str, array-like]` - New data with same blocks/features

**Returns:** `dict[str, np.ndarray]` - Variates per block `{block: (n_new, n_comp)}`

**Example:**
```python
# Project test set
test_variates = model.transform(X_test_dict)

# Access per block
mRNA_scores = test_variates['mRNA']  # shape: (n_test, n_comp)
```

---

### `fit_transform(X_dict, Y=None, study=None)`
Convenience method: fit and transform in one call.

**Example:**
```python
variates = model.fit_transform(X_dict, Y=labels, study=batches)
```

---

### `get_biomarkers(block, comp=1)`
Extract selected biomarkers for a specific block and component.

**Parameters:**
- `block`: `str` - Block name
- `comp`: `int` - Component index (1-based)

**Returns:** `pd.DataFrame` - Index=variable names, column=`value` (loading weights), sorted by absolute magnitude.

**Example:**
```python
# Get top biomarkers for mRNA on component 1
biomarkers = model.get_biomarkers('mRNA', comp=1)
print(biomarkers.head(20))

# Output:
#              value
# Variable          
# Gene_142    0.342
# Gene_007   -0.287
# Gene_256    0.245
# ...
```

---

### `get_expression_means(active_features, groups)`
Compute per-group expression means for visualization (used by circos plot).

**Parameters:**
- `active_features`: `dict` - Output from internal processing with 'X' arrays
- `groups`: `dict` - Group name → color mapping

**Returns:** `dict` - `{block: {group: mean_vector}}`

**Example:**
```python
# Prepare active features (typically done internally by visualizations)
active_features = {}
for block in model.block_names_:
    raw_loadings = model.loadings_[block][:, 0:2]
    active_idx = np.where(np.any(raw_loadings != 0, axis=1))[0]
    active_names = [model.names_[block][i] for i in active_idx]
    X_active = model.X_dict_[block][active_names].values
    active_features[block] = {'names': active_names, 'X': X_active}

# Group colors
groups = {0: '#3498DB', 1: '#E74C3C', 2: '#2ECC71', 3: '#9B59B6'}

means = model.get_expression_means(active_features, groups)
```

---

### `from_mudata(cls, mdata, target_col=None, study_col="study", **kwargs)`
Class method: Create MINT from a MuData object.

**Parameters:**
- `mdata`: `mudata.MuData` - Multi-modal data container
- `target_col`: `str`, optional - Column in `mdata.obs` for Y (supervised)
- `study_col`: `str`, default `"study"` - Column in `mdata.obs` for study labels
- `**kwargs`: Additional arguments passed to MINT constructor

**Returns:** `MINT` - Unfitted instance with `X_dict_`, `Y_`, `study_` populated

**Example:**
```python
import mudata as md

# Load or create MuData
mdata = md.read_h5mu("multiomics.h5mu")

# Create model from MuData
model = MINT.from_mudata(
    mdata=mdata,
    target_col="Subtype",      # Y variable
    study_col="Batch",         # Batch/study variable
    n_components=2,
    keepA={'mRNA': [20, 20], 'Proteins': [15, 15]}
)

# Fit (data already extracted from MuData)
model.fit()
```

---

## Visualization Integration

MINT is fully compatible with pythomics visualization functions:

```python
from pythomics.visualizations import (
    plot_circos, plot_cim, plot_network, 
    plot_biomarkers, plot_pca
)

# Color schemes
block_colors = {
    'mRNA': '#27AE60',
    'Proteins': '#8E44AD',
    'Metabolites': '#F39C12'
}
group_colors = {0: '#3498DB', 1: '#E74C3C', 2: '#2ECC71', 3: '#9B59B6'}

# 1. Circos Plot (multi-omics correlations)
fig, ax = plot_circos(
    model=model,
    cutoff=0.7,
    block_colors=block_colors,
    group_colors=group_colors,
    title="MINT Circos Plot"
)
fig.savefig("mint_circos.png", dpi=300, bbox_inches="tight")

# 2. Clustered Image Map (CIM) / Heatmap
grid = plot_cim(
    model=model,
    components=[1],
    block_colors=block_colors,
    group_colors=group_colors,
    show_sample_names=False,
    title="MINT CIM Heatmap"
)
grid.savefig("mint_cim.png", dpi=300, bbox_inches="tight")

# 3. Relevance Network
fig, ax = plot_network(
    model=model,
    cutoff=0.7,
    block_colors=block_colors,
    group_colors=group_colors,
    title="MINT Relevance Network"
)
fig.savefig("mint_network.png", dpi=300, bbox_inches="tight")

# 4. Biomarker Loadings
fig, axes = plot_biomarkers(
    model=model,
    block_colors=block_colors,
    top_n=20,
    title="MINT Selected Biomarkers"
)
fig.savefig("mint_biomarkers.png", dpi=300, bbox_inches="tight")

# 5. PCA Scatter (on MINT variates)
fig, ax = plot_pca(
    model=model,  # Uses model.variates_
    metadata=pd.DataFrame({'Group': Y}, index=X_dict[list(X_dict.keys())[0]].index),
    hue='Group',
    palette=group_colors,
    title="MINT Variates PCA"
)
fig.savefig("mint_pca.png", dpi=300, bbox_inches="tight")
```

---

## Biomarker Annotation & AI Reports

Integrate with `BioAnnotator` and `BioLLM` for functional interpretation:

```python
from pythomics.ai import BioAnnotator, BioLLM
from pythomics.reports import BioReportPDF
import os

# Collect biomarkers per block
biomarkers_dfs = {}
for block in model.block_names_:
    biomarkers_dfs[block] = model.get_biomarkers(block, comp=1)

# Annotate with biological databases
annotator = BioAnnotator()

# Generate LLM analysis (requires OPENROUTER_API_KEY in .env)
api_key = os.getenv("OPENROUTER_API_KEY")
llm = BioLLM(api_key=api_key, provider="openrouter", 
             model="meta-llama/llama-3-8b-instruct:free")

# Generate PDF report
report = BioReportPDF(title="MINT Multi-omics Integration Report")
report.generate_biomarkers_report(
    output_path="results/mint_report.pdf",
    biomarkers_dfs=biomarkers_dfs,
    biomarkers_plot_path=[
        "mint_biomarkers.png",
        "mint_cim.png", 
        "mint_circos.png",
        "mint_network.png"
    ],
    annotator_instance=annotator,
    llm_instance=llm
)
```

---

## Complete Pipeline Example

```python
"""
Complete MINT multi-omics pipeline with batch correction.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from pythomics.models import MINT
from pythomics.visualizations import (
    plot_circos, plot_cim, plot_network, 
    plot_biomarkers, plot_pca
)
from pythomics.ai import BioAnnotator, BioLLM
from pythomics.reports import BioReportPDF

# ============================================================
# 1. LOAD DATA (replace with your data loading logic)
# ============================================================
# Expected: X_dict = {block_name: DataFrame(samples x features)}
# study = array of batch/cohort labels per sample
# Y = array of outcome labels (optional, for supervised)

# Example synthetic data
np.random.seed(42)
n_samples = 120
X_dict = {
    'mRNA': pd.DataFrame(
        np.random.randn(n_samples, 1000),
        columns=[f'Gene_{i}' for i in range(1000)],
        index=[f'Sample_{i}' for i in range(n_samples)]
    ),
    'Proteins': pd.DataFrame(
        np.random.randn(n_samples, 300),
        columns=[f'Prot_{i}' for i in range(300)],
        index=[f'Sample_{i}' for i in range(n_samples)]
    ),
    'Methy': pd.DataFrame(
        np.random.randn(n_samples, 500),
        columns=[f'CpG_{i}' for i in range(500)],
        index=[f'Sample_{i}' for i in range(n_samples)]
    )
}

# Two cohorts (batch effect)
study = np.array([0]*60 + [1]*60)

# Clinical outcome (4 subtypes)
Y = np.array([0]*30 + [1]*30 + [2]*30 + [3]*30)

# ============================================================
# 2. FIT MINT MODEL
# ============================================================
print("Fitting MINT model...")

model = MINT(
    n_components=2,
    keepA={
        'mRNA': [30, 30],
        'Proteins': [20, 20],
        'Methy': [25, 25]
    },
    tau=None,           # MINT path for batch correction
    scheme='horst',
    mode='canonical',
    scale=True,
    max_iter=200,
    tol=1e-6,
    init='svd.single'
)

model.fit(X_dict=X_dict, Y=Y, study=study)

print(f"Fitted: {model.block_names_}")
print(f"Components per block: {model.n_components_}")

# ============================================================
# 3. EXTRACT BIOMARKERS
# ============================================================
os.makedirs("results/mint", exist_ok=True)

biomarkers_dfs = {}
for block in model.block_names_:
    bio = model.get_biomarkers(block, comp=1)
    biomarkers_dfs[block] = bio
    bio.to_csv(f"results/mint/biomarkers_{block}.csv")
    print(f"{block}: {len(bio)} biomarkers selected")

# ============================================================
# 4. VISUALIZATIONS
# ============================================================
block_colors = {'mRNA': '#27AE60', 'Proteins': '#8E44AD', 'Methy': '#F39C12'}
group_colors = {0: '#3498DB', 1: '#E74C3C', 2: '#2ECC71', 3: '#9B59B6'}

# Circos
fig, ax = plot_circos(model, cutoff=0.7, block_colors=block_colors, 
                      group_colors=group_colors, title="MINT Circos")
fig.savefig("results/mint/circos.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# CIM
grid = plot_cim(model, components=[1], block_colors=block_colors,
                group_colors=group_colors, show_sample_names=False,
                title="MINT CIM")
grid.savefig("results/mint/cim.png", dpi=300, bbox_inches="tight")

# Network
fig, ax = plot_network(model, cutoff=0.7, block_colors=block_colors,
                       group_colors=group_colors, title="MINT Network")
fig.savefig("results/mint/network.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Biomarkers
fig, axes = plot_biomarkers(model, block_colors=block_colors,
                            top_n=25, title="MINT Biomarkers")
fig.savefig("results/mint/biomarkers.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 5. ANNOTATION & REPORT (optional)
# ============================================================
try:
    annotator = BioAnnotator()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        llm = BioLLM(api_key=api_key, provider="openrouter",
                     model="meta-llama/llama-3-8b-instruct:free")
        
        report = BioReportPDF(title="MINT Multi-omics Integration Report")
        report.generate_biomarkers_report(
            output_path="results/mint/report.pdf",
            biomarkers_dfs=biomarkers_dfs,
            biomarkers_plot_path=[
                "results/mint/biomarkers.png",
                "results/mint/cim.png",
                "results/mint/circos.png",
                "results/mint/network.png"
            ],
            annotator_instance=annotator,
            llm_instance=llm
        )
        print("Report generated: results/mint/report.pdf")
    else:
        print("Skipping LLM report (OPENROUTER_API_KEY not set)")
except Exception as e:
    print(f"Report generation skipped: {e}")

print("Pipeline completed!")
```

---

## MINT vs DIABLO: When to Use Which?

| Scenario | Recommended Method |
|----------|-------------------|
| Single cohort, no batch effects | **DIABLO** (`DIABLOWrapper`) |
| Multiple cohorts/studies with batch effects | **MINT** (`MINT`, `tau=None`) |
| Single cohort but want ridge regularization | **MINT** (`MINT`, `tau="optimal"`) |
| Classification with known subtypes | Both (supervised mode) |
| Unsupervised integration/clustering | Both (no Y provided) |

### Key Differences

```python
# DIABLO (single study)
from pythomics.models import DIABLOWrapper
diablo = DIABLOWrapper.from_mudata(mdata, target_col="Subtype")
diablo.fit()

# MINT (multi-study with batch correction)
from pythomics.models import MINT
mint = MINT.from_mudata(mdata, target_col="Subtype", study_col="Batch")
mint.fit()  # Automatically uses study labels for batch correction
```

---

## Algorithm Details

### MINT Path (`tau=None`)

The MINT algorithm computes loadings by aggregating per-study contributions:

```
For each block q:
    loading_q = Σ_study (X_study_q.T @ Z_study)
    
Where Z_study is the design-weighted combination of variates within that study.
```

This prevents batch effects from driving the integrative components, as each study contributes independently to the loading direction.

### RGCCA Path (`tau="optimal"` or fixed)

Uses ridge-regularized covariance matrices:

```
M_q = pinv(tau_q * I + (1 - tau_q) * Cov(X_q))
loading_q = M_q @ (X_q.T @ Z_q)
```

- `tau = 0`: No regularization (standard CCA)
- `tau = 1`: Maximum regularization (identity covariance)
- `"optimal"`: Estimated via Strimmer's analytical formula

### Sparsity Mechanism

Both paths support sparsity via the `sparsity()` function:

1. **`keepA` (exact cardinality)**: Binary search for threshold yielding exactly k non-zeros
2. **`penalty` (L1)**: Soft-thresholding with fixed penalty

```python
# In sparse_mint_block_iteration (MINT path):
loadings = sparsity(loadings, keep_a=keepA[q], penalty=None)
loadings = l2_norm(loadings)  # Renormalize to unit L2 norm

# In sparse_rgcca_iteration (RGCCA path):
temp_norm = norm2(loadings)
loadings = sparsity(loadings, keep_a=keepA[q], penalty=None)
loadings = (loadings / norm2(loadings)) * temp_norm  # Preserve pre-sparsity norm
```

---

## Troubleshooting

### Convergence Warnings

```
UserWarning: sparse_mint_block_iteration: no convergio en 100 iteraciones.
```

**Solutions:**
- Increase `max_iter` (try 200-500)
- Increase `tol` (try 1e-4)
- Check for high collinearity in data
- Reduce number of components

### Memory Issues with Large Data

For very large feature sets (>10k features per block):
- Pre-filter features by variance
- Use `tau="optimal"` (RGCCA path uses dual formulation when n < p)
- Consider `keepA` to limit active variables

### Study Label Issues

```python
# Ensure study labels are consistent
print(f"Unique studies: {np.unique(study)}")  # Should be [0, 1, 2, ...]
print(f"Samples per study: {np.bincount(study)}")  # Should be balanced

# If study_col not found in MuData:
model = MINT.from_mudata(mdata, study_col="Batch")  # Try "Batch", "Cohort", "Study"
```

---

## API Reference Summary

```python
# Construction
model = MINT(**params)
model = MINT.from_mudata(mdata, target_col, study_col, **params)

# Fitting
model.fit(X_dict, Y, study)
model.fit(X_dict)  # If initialized from MuData

# Transformation
variates = model.transform(X_dict)
variates = model.fit_transform(X_dict, Y, study)

# Biomarkers
df = model.get_biomarkers(block_name, comp=1)

# Visualization compatibility
model.loadings_      # {block: (p, ncomp)}
model.variates_      # {block: (n, ncomp)}
model.names_         # {block: [feature_names]}
model.X_dict_        # {block: DataFrame}
model.Y_             # array or None
model.block_names_   # [block_names]
model.study_         # study labels
model.n_components_  # [ncomp_per_block]

# Internal result (advanced)
model._mint_result   # MINTResult dataclass with all internals
```

---

## References

1. **Rohart et al. (2017)** - "MINT: A multivariate integrative method to identify a reproducible molecular signature across independent experiments and platforms." *BMC Bioinformatics*.

2. **mixOmics R package** - `mint.block()`, `mint.spls()`, `mint.splsda()` implementations.

3. **Tenenhaus et al. (2014)** - "Variable selection for generalized canonical correlation analysis." *Biostatistics*.

---

## License

Part of the `pythomics` package. See main repository for license details.
