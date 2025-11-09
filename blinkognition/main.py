import mlflow
import typer
from typing import Optional

from blinkognition.data import app as data_app
from blinkognition.ml import app as ml_app
from blinkognition.cv import app as cv_app

app = typer.Typer()

# register subcommand groups with names
app.add_typer(data_app, name="data", help="Data prep & utilities")
app.add_typer(ml_app,   name="ml",   help="Model training & evaluation")
app.add_typer(cv_app,   name="cv",   help="Computer vision tools")

mlflow_servers = dict(
    local='http://127.0.0.1:8080',
    kit='https://mlflow.scc.kit.edu',
)

@app.callback()
def main(
    mlflow_server: str = mlflow_servers['local'],
    experiment_name: str = 'blinkognition',
    experiment_id: str | None = None,   # default None!
):
    mlflow.set_tracking_uri(uri=mlflow_server)

    if experiment_id:
        exp = mlflow.get_experiment(experiment_id)
        if exp is None:
            raise ValueError(f"No MLflow experiment with id {experiment_id!r} at {mlflow_server}")
        mlflow.set_experiment(experiment_id=experiment_id)   # pass only one arg
        return

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        print('Create new experiment:', experiment_name)
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name=experiment_name)
