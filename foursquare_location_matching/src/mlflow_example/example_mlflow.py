from mlflow import log_metric, log_param, log_artifact, set_experiment
from mlflow.tracking import MlflowClient

set_experiment("test_mlflow")

# Log a parameter (key-value pair)
log_param("param1", 43.123)

# Log a metric; metrics can be updated throughout the run
log_metric("foo", 1)
log_metric("foo", 2)

# logging existing files as artifacts (e.g. the own source code lol)
log_artifact("example_mlflow.py")

print("all done mlflow-wise")
