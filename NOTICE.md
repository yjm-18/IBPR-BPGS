# Rights and Third-Party Components

IBPR + BPGS is maintained at https://github.com/yjm-18/IBPR-BPGS.
No software license is currently included for the project code. Public availability
of the repository does not itself grant a software license.

Third-party dependencies include NumPy, SciPy, scikit-learn, Numba, threadpoolctl,
PyTorch, spconv, laspy, PyYAML and SoftGroup voxelization operators. They retain their
own licenses. Applicable upstream notices for the auxiliary sparse boundary code
and inherited components must be preserved.

TreeLearn, SegmentAnyTree, their checkpoints and raw datasets are not redistributed
in the source package. Official baseline checkpoint links are documented in
`docs/BASELINES.md`. Hashes for the separate IBPR/BPGS and auxiliary boundary
checkpoints are recorded in `configs/checkpoints.json`.
