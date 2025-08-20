import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""## f1=0.6535""")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mlflow
    import category_encoders as ce

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score, matthews_corrcoef

    from flaml.automl.automl import AutoML
    from flaml import tune
    return AutoML, ce, mlflow, mo, pd, train_test_split


@app.cell
def _(mlflow):
    mlflow.autolog()
    return


@app.cell
def _(pd):
    train_with_text_embds = pd.read_csv("./ml_ozon_сounterfeit_data/train_with_text_embeddings_all-MiniLM-L6-v2.csv", index_col=0)
    test_with_text_embds = pd.read_csv("./ml_ozon_сounterfeit_data/test_with_text_embeddings_all-MiniLM-L6-v2.csv", index_col=0)
    return test_with_text_embds, train_with_text_embds


@app.cell
def _():
    TASK = "classification"
    SEED = 34534588
    return SEED, TASK


@app.cell
def _(train_with_text_embds):
    train_with_text_embds
    return


@app.cell
def _(train_with_text_embds):
    train_with_text_embds.columns
    return


@app.cell
def _(test_with_text_embds):
    test_with_text_embds
    return


@app.cell
def _(test_with_text_embds):
    test_with_text_embds.columns
    return


@app.cell
def _(test_with_text_embds, train_with_text_embds):
    cols_without_target = train_with_text_embds.columns.tolist()
    cols_without_target.remove("resolution")

    X_train, y_train = train_with_text_embds[cols_without_target].fillna(0), train_with_text_embds["resolution"].fillna(0)
    X_test = test_with_text_embds[cols_without_target].fillna(0)
    return X_test, X_train, y_train


@app.cell
def _(ce):
    encoder = ce.CountEncoder(
        cols=["CommercialTypeName4", "brand_name"],
        normalize=True,
        handle_missing=-1,
        handle_unknown=-1
    )
    return (encoder,)


@app.cell
def _(X_test, X_train, encoder):
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    return X_test_encoded, X_train_encoded


@app.cell
def _(X_train_encoded):
    numeric_cols = X_train_encoded.select_dtypes(include=["number"]).columns.tolist()
    return (numeric_cols,)


@app.cell
def _(X_test_encoded, X_train_encoded, numeric_cols):
    X_train_with_numeric_cols = X_train_encoded[numeric_cols]
    X_test_with_numeric_cols = X_test_encoded[numeric_cols]
    return X_test_with_numeric_cols, X_train_with_numeric_cols


@app.cell
def _(SEED, X_train_with_numeric_cols, train_test_split, y_train):
    (
        X_sub_train,
        X_val,
        y_subtrain,
        y_val
    ) = train_test_split(X_train_with_numeric_cols, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
    return (X_sub_train,)


@app.cell
def _(X_sub_train):
    X_sub_train.shape
    return


@app.cell
def _(X_train_with_numeric_cols):
    X_train_with_numeric_cols
    return


@app.cell
def _(AutoML):
    automl = AutoML()
    return (automl,)


@app.cell
def _(SEED, TASK, X_train_with_numeric_cols, automl, mlflow, y_train):
    mlflow.set_experiment("flaml-xgb-lgbm-text-embds-count-enc-budget=600-holdout")
    with mlflow.start_run():
        automl.fit(
            X_train_with_numeric_cols, y_train,
            task=TASK, 
            estimator_list=[
                "xgboost",
                "lgbm",
                # "catboost"
            ],
            metric="f1",
            eval_method="holdout",
            split_type="stratified",
            time_budget=3600,
            seed=SEED,
        )
    return


@app.cell
def _(automl):
    automl.best_estimator
    return


@app.cell
def _(automl):
    automl.best_config
    return


@app.cell
def _(X_test_with_numeric_cols, automl, pd):
    submission = pd.DataFrame({
        "id": X_test_with_numeric_cols.index, 
        "prediction": automl.model.predict(X_test_with_numeric_cols),
    })

    submission.to_csv("./submission.csv", index=False)
    return


@app.cell
def _(pd):
    pd.read_csv("./submission.csv")["prediction"].value_counts()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
