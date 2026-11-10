# Public Report Configurations

Each JSON file under `configs/` selects a dataset/model pair from the aggregate
CSV and specifies the displayed arms, columns and decimal precision. Paths are
resolved relative to the configuration file.

These files were newly constructed for this public preview. They are not the
original experiment configurations with renamed keys or obscured numeric values.
They contain no model hyperparameters, feature definitions, training seeds,
split manifests, checkpoint paths, acceptance thresholds or selection records.

The core inference and training configurations remain private. Passing these
report configurations to the unreleased method is not supported.
