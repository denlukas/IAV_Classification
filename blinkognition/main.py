import mlflow
import typer

from blinkognition.data import app as data_app
from blinkognition.ml import app as ml_app
from blinkognition.cv import app as cv_app

app = typer.Typer()


app.add_typer(data_app)
app.add_typer(ml_app)
app.add_typer(cv_app)

mlflow_servers = dict(
    local='http://127.0.0.1:8080',
    kit='https://mlflow.scc.kit.edu',
)


# This enables switching the MLflow server and/or experiment from the CLI.
# run like: `uv run app --experiment-name "experiment_42" train ...`
# or change the default from `mlflow_servers['local']` to 'kit'.
@app.callback()
def main(mlflow_server: str = mlflow_servers['local'],
         experiment_name: str = 'blinkognition'):
    mlflow.set_tracking_uri(uri=mlflow_server)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        print('Create new experiment: ', experiment_name)
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)


if __name__ == '__main__':
    app()
