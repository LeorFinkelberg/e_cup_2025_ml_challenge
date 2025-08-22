import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Кодирование по нормализованной частоте атрибутов 'brand_name', 'CommercialTypeName4'
    - Обогащение набора текстовыми эмбедами по модели all-MiniLM-L6-v2
    - lightgbm(is_unbalance=True) на сбалансированном наборе
    - 3-блочная перекрестная проверка; metric=f1
    - Второй запуск с теплым стартом дал f1=0.7191, но третий запуск -- f1=0.7086 
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
def _(mo):
    mo.md(
        r"""
    [flaml.automl.logger: 08-22 14:27:29] {1752} INFO - task = classification
    [flaml.automl.logger: 08-22 14:27:29] {1763} INFO - Evaluation method: cv
    [flaml.automl.logger: 08-22 14:28:59] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 08-22 14:28:59] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 08-22 14:28:59] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 08-22 14:29:08] {2417} INFO - Estimated sufficient time budget=88276s. Estimated necessary time budget=88s.
    [flaml.automl.logger: 08-22 14:29:08] {2466} INFO -  at 116.7s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 08-22 14:29:08] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 08-22 14:29:17] {2466} INFO -  at 125.9s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 08-22 14:29:17] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 08-22 14:29:27] {2466} INFO -  at 135.4s,	estimator lgbm's best error=0.4033,	best estimator lgbm's best error=0.4033
    [flaml.automl.logger: 08-22 14:29:27] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 08-22 14:29:36] {2466} INFO -  at 144.6s,	estimator lgbm's best error=0.4033,	best estimator lgbm's best error=0.4033
    [flaml.automl.logger: 08-22 14:29:36] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 08-22 14:29:46] {2466} INFO -  at 154.2s,	estimator lgbm's best error=0.4033,	best estimator lgbm's best error=0.4033
    [flaml.automl.logger: 08-22 14:29:46] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 08-22 14:29:56] {2466} INFO -  at 164.7s,	estimator lgbm's best error=0.4033,	best estimator lgbm's best error=0.4033
    [flaml.automl.logger: 08-22 14:29:56] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 08-22 14:30:06] {2466} INFO -  at 174.3s,	estimator lgbm's best error=0.4033,	best estimator lgbm's best error=0.4033
    [flaml.automl.logger: 08-22 14:30:06] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 08-22 14:30:14] {2466} INFO -  at 183.0s,	estimator lgbm's best error=0.4033,	best estimator lgbm's best error=0.4033
    [flaml.automl.logger: 08-22 14:30:14] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 08-22 14:30:29] {2466} INFO -  at 197.6s,	estimator lgbm's best error=0.3165,	best estimator lgbm's best error=0.3165
    [flaml.automl.logger: 08-22 14:30:29] {2282} INFO - iteration 9, current learner lgbm
    [flaml.automl.logger: 08-22 14:30:44] {2466} INFO -  at 212.5s,	estimator lgbm's best error=0.3165,	best estimator lgbm's best error=0.3165
    [flaml.automl.logger: 08-22 14:30:44] {2282} INFO - iteration 10, current learner lgbm
    [flaml.automl.logger: 08-22 14:31:00] {2466} INFO -  at 228.4s,	estimator lgbm's best error=0.3165,	best estimator lgbm's best error=0.3165
    [flaml.automl.logger: 08-22 14:31:00] {2282} INFO - iteration 11, current learner lgbm
    [flaml.automl.logger: 08-22 14:31:26] {2466} INFO -  at 255.0s,	estimator lgbm's best error=0.2577,	best estimator lgbm's best error=0.2577
    [flaml.automl.logger: 08-22 14:31:26] {2282} INFO - iteration 12, current learner lgbm
    [flaml.automl.logger: 08-22 14:31:41] {2466} INFO -  at 269.7s,	estimator lgbm's best error=0.2577,	best estimator lgbm's best error=0.2577
    [flaml.automl.logger: 08-22 14:31:41] {2282} INFO - iteration 13, current learner lgbm
    [flaml.automl.logger: 08-22 14:32:03] {2466} INFO -  at 291.5s,	estimator lgbm's best error=0.2577,	best estimator lgbm's best error=0.2577
    [flaml.automl.logger: 08-22 14:32:03] {2282} INFO - iteration 14, current learner lgbm
    [flaml.automl.logger: 08-22 14:32:33] {2466} INFO -  at 321.8s,	estimator lgbm's best error=0.2227,	best estimator lgbm's best error=0.2227
    ...
    [flaml.automl.logger: 08-22 15:17:01] {2282} INFO - iteration 37, current learner lgbm
    [flaml.automl.logger: 08-22 15:18:08] {2466} INFO -  at 3056.1s,	estimator lgbm's best error=0.1938,	best estimator lgbm's best error=0.1938
    [flaml.automl.logger: 08-22 15:18:08] {2282} INFO - iteration 38, current learner lgbm
    [flaml.automl.logger: 08-22 15:21:30] {2466} INFO -  at 3258.3s,	estimator lgbm's best error=0.1938,	best estimator lgbm's best error=0.1938
    [flaml.automl.logger: 08-22 15:21:30] {2282} INFO - iteration 39, current learner lgbm
    [flaml.automl.logger: 08-22 15:22:48] {2466} INFO -  at 3336.9s,	estimator lgbm's best error=0.1938,	best estimator lgbm's best error=0.1938
    [flaml.automl.logger: 08-22 15:22:48] {2282} INFO - iteration 40, current learner lgbm
    [flaml.automl.logger: 08-22 15:25:26] {2466} INFO -  at 3494.9s,	estimator lgbm's best error=0.1938,	best estimator lgbm's best error=0.1938
    [flaml.automl.logger: 08-22 15:25:26] {803} INFO - logging best model lgbm
    [flaml.automl.logger: 08-22 15:26:18] {2724} INFO - retrain lgbm for 51.9s
    [flaml.automl.logger: 08-22 15:26:18] {2727} INFO - retrained model: LGBMClassifier(is_unbalance=True, learning_rate=np.float64(0.10154574806498816),
                   max_bin=511, min_child_samples=80, n_estimators=225, n_jobs=-1,
                   num_leaves=146, reg_alpha=0.0009765625,
                   reg_lambda=np.float64(5.0063952712925746), verbose=-1)
    [flaml.automl.logger: 08-22 15:26:18] {2729} INFO - Best MLflow run name: delicate-kite-168_child_31
    [flaml.automl.logger: 08-22 15:26:18] {2730} INFO - Best MLflow run id: af1bbcc31a44497d861e5df16c0c1f00
    [flaml.automl.logger: 08-22 15:26:23] {2771} WARNING - Exception for record_state task run_29bb1a3517b74b17b88ae7f956fb2a52_requirements_updated: No such file or directory: '/Users/leorfinkelberg/Documents/Competitions/e_cup_2025_ml_challenge/mlruns/819154740489404635/29bb1a3517b74b17b88ae7f956fb2a52/artifacts/model/requirements.txt'
    [flaml.automl.logger: 08-22 15:26:23] {2009} INFO - fit succeeded
    [flaml.automl.logger: 08-22 15:26:23] {2010} INFO - Time taken to find the best model: 2257.70072388649
    """
    )
    return


@app.cell
def _(automl):
    automl.best_estimator
    return


@app.cell
def _(
    AutoML,
    SEED,
    TASK,
    X_train_with_numeric_cols,
    automl,
    custom_hp,
    y_train,
):
    automl_second_part = AutoML()

    automl_second_part.fit(
        X_train_with_numeric_cols, 
        y_train,
        task=TASK,
        time_budget=3600,
        eval_method="cv",
        n_splits=3,
        metric="f1",
        split_type="stratified",
        estimator_list=["lgbm"],
        custom_hp=custom_hp,
        seed=SEED,
        starting_points=automl.best_config_per_estimator,
    )
    return (automl_second_part,)


@app.cell
def _(mo):
    mo.md(
        r"""
    [flaml.automl.logger: 08-22 15:28:09] {1752} INFO - task = classification
    [flaml.automl.logger: 08-22 15:28:09] {1763} INFO - Evaluation method: cv
    [flaml.automl.logger: 08-22 15:28:10] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 08-22 15:28:10] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 08-22 15:28:10] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 08-22 15:30:17] {2417} INFO - Estimated sufficient time budget=1275041s. Estimated necessary time budget=1275s.
    [flaml.automl.logger: 08-22 15:30:17] {2466} INFO -  at 146.1s,	estimator lgbm's best error=0.1938,	best estimator lgbm's best error=0.1938
    [flaml.automl.logger: 08-22 15:30:17] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 08-22 15:30:47] {2466} INFO -  at 176.2s,	estimator lgbm's best error=0.1938,	best estimator lgbm's best error=0.1938
    [flaml.automl.logger: 08-22 15:30:47] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 08-22 15:34:40] {2466} INFO -  at 408.5s,	estimator lgbm's best error=0.1938,	best estimator lgbm's best error=0.1938
    [flaml.automl.logger: 08-22 15:34:40] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 08-22 15:35:59] {2466} INFO -  at 488.3s,	estimator lgbm's best error=0.1938,	best estimator lgbm's best error=0.1938
    [flaml.automl.logger: 08-22 15:35:59] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 08-22 15:39:02] {2466} INFO -  at 670.5s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 15:39:02] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 08-22 15:42:40] {2466} INFO -  at 888.6s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 15:42:40] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 08-22 15:43:40] {2466} INFO -  at 948.5s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 15:43:40] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 08-22 15:44:05] {2466} INFO -  at 973.8s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    ...
    [flaml.automl.logger: 08-22 15:43:40] {2466} INFO -  at 948.5s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 15:43:40] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 08-22 15:44:05] {2466} INFO -  at 973.8s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 15:44:05] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 08-22 15:55:10] {2466} INFO -  at 1638.9s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 15:55:10] {2282} INFO - iteration 9, current learner lgbm
    [flaml.automl.logger: 08-22 15:58:15] {2466} INFO -  at 1824.1s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 15:58:15] {2282} INFO - iteration 10, current learner lgbm
    [flaml.automl.logger: 08-22 16:00:45] {2466} INFO -  at 1973.8s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 16:00:45] {2282} INFO - iteration 11, current learner lgbm
    [flaml.automl.logger: 08-22 16:06:15] {2466} INFO -  at 2303.6s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 16:06:15] {2282} INFO - iteration 12, current learner lgbm
    [flaml.automl.logger: 08-22 16:07:15] {2466} INFO -  at 2363.8s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 16:07:15] {2282} INFO - iteration 13, current learner lgbm
    [flaml.automl.logger: 08-22 16:09:48] {2466} INFO -  at 2517.2s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 16:09:48] {2282} INFO - iteration 14, current learner lgbm
    [flaml.automl.logger: 08-22 16:13:17] {2466} INFO -  at 2726.1s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 16:13:17] {2282} INFO - iteration 15, current learner lgbm
    [flaml.automl.logger: 08-22 16:14:38] {2466} INFO -  at 2807.2s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 16:14:38] {2282} INFO - iteration 16, current learner lgbm
    [flaml.automl.logger: 08-22 16:19:58] {2466} INFO -  at 3126.7s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 16:19:58] {2282} INFO - iteration 17, current learner lgbm
    [flaml.automl.logger: 08-22 16:20:46] {2466} INFO -  at 3174.5s,	estimator lgbm's best error=0.1918,	best estimator lgbm's best error=0.1918
    [flaml.automl.logger: 08-22 16:20:46] {2282} INFO - iteration 18, current learner lgbm
    [flaml.automl.logger: 08-22 16:27:46] {2466} INFO -  at 3594.9s,	estimator lgbm's best error=0.1896,	best estimator lgbm's best error=0.1896
    [flaml.automl.logger: 08-22 16:32:30] {2724} INFO - retrain lgbm for 283.9s
    [flaml.automl.logger: 08-22 16:32:30] {2727} INFO - retrained model: LGBMClassifier(colsample_bytree=np.float64(0.9486664361547485),
                   is_unbalance=True, learning_rate=np.float64(0.03776068131457385),
                   max_bin=1023, min_child_samples=42, n_estimators=1170, n_jobs=-1,
                   num_leaves=136, reg_alpha=0.0009765625,
                   reg_lambda=np.float64(17.849687549035533), verbose=-1)
    [flaml.automl.logger: 08-22 16:32:30] {2009} INFO - fit succeeded
    [flaml.automl.logger: 08-22 16:32:30] {2010} INFO - Time taken to find the best model: 3594.9193618297577
    """
    )
    return


@app.cell
def _(
    AutoML,
    SEED,
    TASK,
    X_train_with_numeric_cols,
    automl_second_part,
    custom_hp,
    y_train,
):
    automl_third_part = AutoML()

    automl_third_part.fit(
        X_train_with_numeric_cols, 
        y_train,
        task=TASK,
        time_budget=7200,
        eval_method="cv",
        n_splits=3,
        metric="f1",
        split_type="stratified",
        estimator_list=["lgbm"],
        custom_hp=custom_hp,
        seed=SEED,
        starting_points=automl_second_part.best_config_per_estimator,
    )
    return (automl_third_part,)


@app.cell
def _(mo):
    mo.md(
        r"""
    [flaml.automl.logger: 08-22 16:44:16] {1752} INFO - task = classification
    [flaml.automl.logger: 08-22 16:44:16] {1763} INFO - Evaluation method: cv
    [flaml.automl.logger: 08-22 16:44:16] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 08-22 16:44:16] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 08-22 16:44:16] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 08-22 16:55:50] {2417} INFO - Estimated sufficient time budget=6943508s. Estimated necessary time budget=6944s.
    [flaml.automl.logger: 08-22 16:55:50] {2466} INFO -  at 712.9s,	estimator lgbm's best error=0.1895,	best estimator lgbm's best error=0.1895
    [flaml.automl.logger: 08-22 16:55:50] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 08-22 16:57:39] {2466} INFO -  at 821.6s,	estimator lgbm's best error=0.1895,	best estimator lgbm's best error=0.1895
    [flaml.automl.logger: 08-22 16:57:39] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 08-22 17:26:39] {2466} INFO -  at 2561.0s,	estimator lgbm's best error=0.1895,	best estimator lgbm's best error=0.1895
    [flaml.automl.logger: 08-22 17:26:39] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 08-22 17:34:00] {2466} INFO -  at 3002.1s,	estimator lgbm's best error=0.1895,	best estimator lgbm's best error=0.1895
    [flaml.automl.logger: 08-22 17:34:00] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 08-22 17:47:27] {2466} INFO -  at 3808.9s,	estimator lgbm's best error=0.1885,	best estimator lgbm's best error=0.1885
    [flaml.automl.logger: 08-22 17:47:27] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 08-22 18:04:42] {2466} INFO -  at 4844.7s,	estimator lgbm's best error=0.1885,	best estimator lgbm's best error=0.1885
    [flaml.automl.logger: 08-22 18:04:42] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 08-22 18:08:49] {2466} INFO -  at 5091.3s,	estimator lgbm's best error=0.1885,	best estimator lgbm's best error=0.1885
    [flaml.automl.logger: 08-22 18:08:49] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 08-22 18:10:05] {2466} INFO -  at 5167.3s,	estimator lgbm's best error=0.1885,	best estimator lgbm's best error=0.1885
    [flaml.automl.logger: 08-22 18:10:05] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 08-22 18:44:18] {2466} INFO -  at 7220.2s,	estimator lgbm's best error=0.1885,	best estimator lgbm's best error=0.1885
    [flaml.automl.logger: 08-22 18:50:02] {2724} INFO - retrain lgbm for 344.3s
    [flaml.automl.logger: 08-22 18:50:02] {2727} INFO - retrained model: LGBMClassifier(colsample_bytree=np.float64(0.9781580567219995),
                   is_unbalance=True, learning_rate=np.float64(0.05304161693123269),
                   max_bin=1023, min_child_samples=40, n_estimators=2442, n_jobs=-1,
                   num_leaves=61, reg_alpha=0.0009765625,
                   reg_lambda=np.float64(108.61245004543387), verbose=-1)
    [flaml.automl.logger: 08-22 18:50:02] {2009} INFO - fit succeeded
    [flaml.automl.logger: 08-22 18:50:02] {2010} INFO - Time taken to find the best model: 3808.947667121887
    """
    )
    return


@app.cell
def _(X_test_with_numeric_cols, automl_third_part, pd):
    submission = pd.DataFrame({
        "id": X_test_with_numeric_cols.index, 
        "prediction": automl_third_part.model.predict(X_test_with_numeric_cols),
    })

    submission.to_csv("./submission.csv", index=False)
    return


@app.cell
def _(pd):
    pd.read_csv("./submission_f1=0.7086.csv")["prediction"].value_counts()
    return


@app.cell
def _(pd):
    pd.read_csv("./submission_f1=0.7001.csv")["prediction"].value_counts()
    return


@app.cell
def _(pd):
    pd.read_csv("./submission_f1=0.7191.csv")["prediction"].value_counts()
    return


@app.cell
def _(pd):
    submission_f1_07086 = pd.read_csv("./submission_f1=0.7086.csv")
    submission_f1_07191 = pd.read_csv("./submission_f1=0.7191.csv")
    return submission_f1_07086, submission_f1_07191


@app.cell
def _(submission_f1_07086, submission_f1_07191):
    submision_joined = submission_f1_07191.join(submission_f1_07086, rsuffix="_right")
    return (submision_joined,)


@app.cell
def _(submision_joined):
    submision_joined
    return


@app.cell
def _(np, submision_joined):
    np.where(submision_joined["prediction"] == submision_joined["prediction_right"], submision_joined["prediction"], submision_joined["prediction_right"]).sum()
    return


@app.cell
def _(submision_joined):
    submision_joined[~(submision_joined["prediction"] == submision_joined["prediction_right"])].loc[:, ["id", "prediction", "prediction_right"]]
    return


@app.cell
def _(submision_joined):
    submision_joined[(submision_joined["prediction"] == 1) & (submision_joined["prediction_right"] == 0)]["prediction"].sum()
    return


@app.cell
def _(submision_joined):
    mask = (submision_joined["prediction"] == 1) & (submision_joined["prediction_right"] == 0)
    submision_joined.loc[mask, "prediction"] = 0
    return


@app.cell
def _(submision_joined):
    submision_joined["prediction"].value_counts()
    return


@app.cell
def _(submision_joined):
    submision_joined[["id", "prediction"]].to_csv("./submission.csv", index=False)
    return


@app.cell
def _(pd):
    pd.read_csv("./submission_f1=0.7086.csv")["prediction"].value_counts()
    return


@app.cell
def _(pd):
    pd.read_csv("./submission_f1=0.7190.csv")["prediction"].value_counts()
    return


@app.cell
def _(pd):
    pd.read_csv("./submission_f1=0.7191.csv")["prediction"].value_counts()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
