from mlflow_setting.mlflow_utils import create_mlflow_experiment

if __name__ == '__main__':
    experiment_id = create_mlflow_experiment(
        experiment_name='experiment01',
        artifact_location='artifact01',
        tags={'env': 'dev', 'version': '1.0.0'},
    )

    print(f'Experiment id: {experiment_id}')
