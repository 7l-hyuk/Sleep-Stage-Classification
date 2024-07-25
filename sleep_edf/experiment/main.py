import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
from mlflow_setting.mlflow_utils import get_mlflow_experiment

import setting
from src.model.model import Model


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment = get_mlflow_experiment(experiment_name=cfg.experiment.name)
    run_name = ""
    print(f'Experiment Name: {experiment.name}')

    for config_name in cfg.experiment.settings:
        with mlflow.start_run(
            run_name=f'{run_name}{config_name}',
            experiment_id=experiment.experiment_id
        ) as run:
            mlflow.log_artifact(
                local_path=f"../conf/{config_name}",
                artifact_path="config"
                )

            model = Model()
            mlflow.xgboost.autolog(log_input_examples=True)

            config_path = hydra.utils.to_absolute_path(
                f"../conf/{config_name}"
                )
            config = OmegaConf.load(config_path)
            model.train_test(config)

            accuracy = {
                'Test Accuracy': model.test_accuracy,
                'Test f1_weight': model.test_f1_weight,
                'Test f1_macro': model.test_f1_macro,
                'Unknown Accuracy': model.unknown_f1_weight,
                'Unknown f1_weight': model.unknown_f1_weight,
                'Unknown f1_macro': model.unknown_f1_macro,
            }
            mlflow.log_metrics(accuracy)

            print(
                f'''
run_id: {run.info.run_id}
experiment_id: {run.info.experiment_id}
status: {run.info.status}
start_time: {run.info.start_time}
end_time: {run.info.end_time}
lifecycle_stage: {run.info.lifecycle_stage}
''')


if __name__ == "__main__":
    main()
