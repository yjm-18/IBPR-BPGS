# Software Environments by Component

The components were run in separate environments. The TL/boundary environment must
not be presented as the environment for SAT or the scikit-learn A/B models.
Versions below are the recorded reproduction environments.

| Component | Python | PyTorch | CUDA (PyTorch build) | cuDNN | NumPy | SciPy | scikit-learn | laspy |
|---|---|---|---|---|---|---|---|---|
| TreeLearn baseline and auxiliary boundary predictor | 3.12.9 | 2.5.1 | 11.8 | 9.1.0 | 2.2.4 | 1.15.2 | 1.6.1 | 2.6.1 |
| SegmentAnyTree baseline | 3.8.20 | 1.9.0 | 11.1 | 8.0.5 | 1.19.5 | 1.7.3 | 1.1.2 | 2.0.3 |
| IBPR and BPGS fitting, inference and evaluation | 3.11.7 | Not required by the core package | Not required | Not required | 1.26.4 | 1.13.1 | 1.5.1 | 2.5.4 for data preparation |

The A/B host environment also contained PyTorch 2.0.0+cu117 and cuDNN 8.7.0, but
these are not dependencies of the CPU scikit-learn correction package. Its additional
runtime packages are Numba 0.60.0 and threadpoolctl 3.5.0, as pinned in the root
requirements. Do not use the SAT environment to deserialize the A/B checkpoints.

The auxiliary environment uses spconv-cu118 2.3.8, cumm-cu118 0.7.11 and PyYAML 6.0.2,
plus the original SoftGroup voxelization extension. Install it separately from the
root A/B requirements. The root `environment.yml` defines the A/B environment only.

Machine-readable version records are in
[environment_audit.json](protocols/environment_audit.json). These are environment
observations, not a claim that every installed package is used by every component.
