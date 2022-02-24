from mlflow import log_metric, log_param, log_artifacts, log_artifact, set_tracking_uri, set_experiment
set_tracking_uri("http://gpulab.ms.mff.cuni.cz:7022")
set_experiment("moo-as-voting-fast")


if __name__ == "__main__":
    print("Hello world")
