import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## f1=0.7001 https://codenrock.com/contests/e-cup-2025/tasks/2497/7055

    - Кодирование по нормализованной частоте атрибутов 'brand_name', 'CommercialTypeName4'
    - Обогащение набора текстовыми эмбедами по модели all-MiniLM-L6-v2
    - lightgbm(is_unbalance=True) на сбалансированном наборе
    """
    )
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
    return AutoML, ce, mlflow, mo, np, pd, train_test_split


@app.cell
def _():
    # mlflow.autolog()
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
def _(X_test):
    X_test.select_dtypes(exclude=["number"])
    return


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
def _(X_test_with_numeric_cols, X_train_with_numeric_cols):
    desc_embed_cols = [col_name for col_name in X_train_with_numeric_cols if col_name.startswith("desc_embed")]
    name_rus_embed_cols = [col_name for col_name in X_train_with_numeric_cols if col_name.startswith("name_rus_embed")]

    X_train_with_numeric_cols["desc_embed_name_rus_embed_dot_product"] = (
        X_train_with_numeric_cols[desc_embed_cols].values * X_train_with_numeric_cols[name_rus_embed_cols].values
    ).sum(axis=1)

    X_test_with_numeric_cols["desc_embed_name_rus_embed_dot_product"] = (
        X_test_with_numeric_cols[desc_embed_cols].values * X_test_with_numeric_cols[name_rus_embed_cols].values
    ).sum(axis=1)
    return desc_embed_cols, name_rus_embed_cols


@app.cell
def _(X_train_with_numeric_cols, desc_embed_cols, name_rus_embed_cols):
    numeric_cols_without_text_embds = [
        col_name for col_name in X_train_with_numeric_cols.columns 
        if (col_name not in desc_embed_cols) and (col_name not in name_rus_embed_cols)
    ]
    return (numeric_cols_without_text_embds,)


@app.cell
def _(numeric_cols_without_text_embds):
    numeric_cols_without_text_embds
    return


@app.cell
def _(X_test_with_numeric_cols, X_train_with_numeric_cols):
    del X_train_with_numeric_cols["desc_embed_name_rus_embed_dot_product"]
    del X_test_with_numeric_cols["desc_embed_name_rus_embed_dot_product"]
    return


@app.cell
def _(X_train_with_numeric_cols):
    X_train_with_numeric_cols.shape
    return


@app.cell
def _(AutoML):
    automl = AutoML()
    return (automl,)


@app.cell
def _(np, y_train):
    custom_hp = {
        "lgbm": {
            "is_unbalance": {
                "domain": True,
            },
        },
        "catboost": {
            "auto_class_weights": {
                "domain": "Balanced",
            },
        },
        "xgboost": {
            "scale_pos_weight": {
                "domain": np.sum(y_train == 0) / np.sum(y_train == 1)  
            },
        }
    }
    return (custom_hp,)


@app.cell
def _(
    SEED,
    TASK,
    X_train_with_numeric_cols,
    automl,
    custom_hp,
    mlflow,
    y_train,
):
    mlflow.set_experiment("flaml-xgb-lgbm-text-embds-count-enc-budget=600-holdout")
    with mlflow.start_run():
        automl.fit(
            X_train_with_numeric_cols,
            y_train,
            task=TASK,
            time_budget=3600,
            estimator_list=(
                "lgbm",
                # "xgboost",
                # "catboost"
            ),
            eval_method="cv",
            n_splits=3,
            metric="f1",
            split_type="stratified",
            custom_hp=custom_hp,
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
