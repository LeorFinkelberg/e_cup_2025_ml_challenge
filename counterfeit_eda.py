import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Исследовательский анализ данных по треку "Контроль качества: автоматическое выявление поддельных товаров"
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import mlflow

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score, matthews_corrcoef

    from flaml.automl.automl import AutoML
    from flaml import tune
    return AutoML, mlflow, mo, np, pd, train_test_split, tune


@app.cell
def _(mlflow):
    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_tracking_uri("http://0.0.0.0:8081")
    mlflow.set_experiment("E-CUP Exp")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Загрузка данных""")
    return


@app.cell
def _(pd):
    train = pd.read_csv("/home/mary/Yandex.Disk/ml_ozon_сounterfeit_data/ml_ozon_сounterfeit_train.csv", index_col=0)
    test = pd.read_csv("/home/mary/Yandex.Disk/ml_ozon_сounterfeit_data/ml_ozon_сounterfeit_test.csv", index_col=0)
    return test, train


@app.cell
def _(mo):
    mo.md(
        r"""
    wget \ 
      --retry-connrefused \ 
      --waitretry=1 \ 
      --read-timeout=40 \ 
      --timeout=15 \ 
      -t 200 \
      https://storage.codenrock.com/public/companies/codenrock-13/ml_ozon_%D1%81ounterfeit_train_images.zip\?response-content-disposition\=attachment%3B+filename%3D%22ml_ozon_%D1%81ounterfeit_train_images.zip%22
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Константы""")
    return


@app.cell
def _():
    TASK = "classification"
    return (TASK,)


@app.cell
def _(train):
    train
    return


@app.cell
def _(train):
    train.shape
    return


@app.cell
def _(test):
    test.shape
    return


@app.cell
def _(train):
    train.describe()
    return


@app.cell
def _(train):
    train["resolution"].value_counts()
    return


@app.cell
def _(train):
    train.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Простой FLAML-конвейер на числовых фичах с подбором гиперпараметров (бюджет по времени: 30 мин)""")
    return


@app.cell
def _(np, train):
    train.select_dtypes(include=[np.number]).describe()
    return


@app.cell
def _(np, train):
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove("resolution")
    return (numeric_cols,)


@app.cell
def _(numeric_cols, test, train):
    X_train = train[numeric_cols].fillna(0)
    y_train = train["resolution"]

    X_test = test[numeric_cols].fillna(0)
    return X_test, X_train, y_train


@app.cell
def _(X_train, train_test_split, y_train):
    (
        X_sub_train,
        X_val,
        y_subtrain,
        y_val
    ) = train_test_split(X_train, y_train, test_size=0.2, random_state=88, stratify=y_train)
    return


@app.cell
def _(AutoML):
    automl = AutoML()
    return (automl,)


@app.cell
def _(tune):
    custom_hp = {
        "xgboost": {
            "n_estimators": {
                "domain": tune.lograndint(lower=50, upper=350),
                "low_cost_init_value": 50,
            },
            "max_depth": {
                "domain": tune.randint(lower=1, upper=12),
                "low_cost_init_value": 3 
            },
            "learning_rate": {
                "domain": tune.loguniform(lower=0.001, upper=1.0),
                "low_cost_init_value": 0.1 
            },
        },
        "lgbm": {
            "subsample": {
                "domain": tune.uniform(lower=0.1, upper=0.9),
                "low_cost_init_value": 0.1,
            },
            "subsample_freq": {
                "domain": 1,
            },
        }, 
        "catboost": {
            ...
        }
    }
    return


@app.cell
def _(TASK, X_test, X_train, automl, mlflow, pd, y_train):
    time_budget = 650;
    seed = 42;

    automl.fit(
        X_train, y_train,
        task=TASK, 
        estimator_list=[
            "xgboost",
            "lgbm",
            # "catboost"
        ],
    metric="f1",
    eval_method="holdout",
    split_type="stratified",
    time_budget=time_budget,
    seed=seed,
    )

    best_result = automl.best_result
    best_estimator = automl.best_estimator


    with mlflow.start_run(run_name="counterfeit_run"):
        mlflow.log_param("time _budget",time_budget)
        mlflow.log_param("seed",seed)
        mlflow.log_param("best_estimator",automl.best_estimator)
        mlflow.log_param("best_loss",automl.best_loss)
        mlflow.log_metric("f1_score",(1-automl.best_loss))
        mlflow.log_params(best_result["config"])
        if best_estimator=="xgboost":
            mlflow.sklearn.log_model(automl.model, "xgboost_model")
        elif best_estimator=="lgbm":
            mlflow.lightgbm.log_model(automl.model,"lightgbm_model")

        pd.Series(automl.predict(X_test)).value_counts()
        submission = pd.DataFrame({
        "id": X_test.index, 
        "prediction": automl.model.predict(X_test),
        })

        submission.to_csv("./submission.csv", index=False)
        mlflow.log_artifact("submission.csv")

    mlflow.end_run()
    return (submission,)


@app.cell
def _(automl):
    automl.best_result
    return


@app.cell
def _(X_test, automl, pd):
    pd.Series(automl.predict(X_test)).value_counts()
    return


@app.cell
def _(automl):
    automl.best_estimator
    return


@app.cell
def _():
    #submission = pd.DataFrame({
    #    "id": X_test.index, 
    #    "prediction": automl.model.predict(X_test),
    #})

    #submission.to_csv("./submission.csv", index=False)
    return


@app.cell
def _(submission):
    submission["prediction"].value_counts()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
