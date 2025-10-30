# IAV Classification

Description (ca. 4 Zeilen)

## Installation
Description
Please install uv if you want to use this code (Link zu uv angeben)
```bash
git clone https://github.com/denlukas/IAV_Classification.git
cd IAV_Classification
uv sync
```
activate environment

## Getting started
How to do the preprocessing steps

### How to make predictions
MLflow starten:

```bash
mlflow server --host 127.0.0.1 --port 8080 --serve-artifacts
uv run app evaluate
```
### How to evaluate the models
### How to train new models

References
evtl. Link zu dem Paper, falls es eins geben sollte