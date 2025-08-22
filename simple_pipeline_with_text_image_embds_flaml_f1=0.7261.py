import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


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
def _(np, pd):
    def load_embeddings(path_to_embeddings: str) -> pd.DataFrame:
        data = np.load(path_to_embeddings)
        embds = data["embeddings"]
        ids = data["ids"]
        embds_col_prefix: str = data["hugging_face_model_path"].item().split("/")[-1]

        return pd.DataFrame(
            embds,
            index=ids,
            columns=[
                f"{embds_col_prefix}_{idx}" for idx in range(embds.shape[1])
            ]
        )
    return (load_embeddings,)


@app.cell
def _():
    TASK = "classification"
    SEED = 34534588
    return SEED, TASK


@app.cell
def _(mlflow):
    mlflow.autolog()
    return


@app.cell
def _(pd):
    train_with_text_embds = pd.read_csv("./ml_ozon_сounterfeit_data/train_with_text_embeddings_all-MiniLM-L6-v2.csv", index_col=0)
    test_with_text_embds = pd.read_csv("./ml_ozon_сounterfeit_data//test_with_text_embeddings_all-MiniLM-L6-v2.csv", index_col=0)
    return test_with_text_embds, train_with_text_embds


@app.cell
def _(load_embeddings):
    train_with_image_embds = load_embeddings("./ml_ozon_сounterfeit_data//train_image_embeddings_vit-base-patch16-224-in21k.npz")
    test_with_image_embds = load_embeddings("./ml_ozon_сounterfeit_data//test_image_embeddings_vit-base-patch16-224-in21k.npz")

    train_with_image_embds.index.name = "ItemID"
    test_with_image_embds.index.name = "ItemID"
    return test_with_image_embds, train_with_image_embds


@app.cell
def _(train_with_text_embds):
    train_with_text_embds
    return


@app.cell
def _(test_with_text_embds):
    test_with_text_embds
    return


@app.cell
def _(train_with_text_embds):
    train_with_text_embds.shape
    return


@app.cell
def _(train_with_image_embds):
    train_with_image_embds.shape
    return


@app.cell
def _(
    test_with_image_embds,
    test_with_text_embds,
    train_with_image_embds,
    train_with_text_embds,
):
    train_with_text_image_embds = train_with_text_embds.join(train_with_image_embds, on="ItemID", how="left").fillna(0)
    test_with_text_image_embds = test_with_text_embds.join(test_with_image_embds, on="ItemID", how="left").fillna(0)
    return test_with_text_image_embds, train_with_text_image_embds


@app.cell
def _(train_with_text_image_embds):
    train_with_text_image_embds.shape
    return


@app.cell
def _(test_with_text_image_embds):
    test_with_text_image_embds.shape
    return


@app.cell
def _(test_with_text_image_embds, train_with_text_image_embds):
    cols_without_target = train_with_text_image_embds.columns.tolist()
    cols_without_target.remove("resolution")

    X_train, y_train = train_with_text_image_embds[cols_without_target], train_with_text_image_embds["resolution"]

    X_test = test_with_text_image_embds[cols_without_target]
    return X_test, X_train, y_train


@app.cell
def _(X_train):
    X_train.shape
    return


@app.cell
def _(X_test):
    X_test.shape
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
    X_train_encoded["brand_name"]
    return


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
def _(X_train_with_numeric_cols):
    X_train_with_numeric_cols.shape
    return


@app.cell
def _(X_test_with_numeric_cols):
    X_test_with_numeric_cols.shape
    return


@app.cell
def _(SEED, X_train_with_numeric_cols, train_test_split, y_train):
    (
        X_sub_train,
        X_val,
        y_sub_train,
        y_val
    ) = train_test_split(X_train_with_numeric_cols, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
    return X_sub_train, X_val


@app.cell
def _(X_sub_train):
    X_sub_train.shape
    return


@app.cell
def _(X_val):
    X_val.shape
    return


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
def _(AutoML):
    automl = AutoML()
    return (automl,)


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
    mlflow.set_experiment("flaml-lgbm-text-image-embds-count-enc-budget=3600-balanced")
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
    [flaml.automl.logger: 08-22 20:54:08] {1752} INFO - task = classification
    [flaml.automl.logger: 08-22 20:54:08] {1763} INFO - Evaluation method: cv
    [flaml.automl.logger: 08-22 20:57:04] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 08-22 20:57:04] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 08-22 20:57:04] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 08-22 20:57:22] {2417} INFO - Estimated sufficient time budget=177244s. Estimated necessary time budget=177s.
    [flaml.automl.logger: 08-22 20:57:22] {2466} INFO -  at 231.1s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 08-22 20:57:22] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 08-22 20:57:40] {2466} INFO -  at 249.4s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 08-22 20:57:40] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 08-22 20:57:59] {2466} INFO -  at 268.0s,	estimator lgbm's best error=0.4058,	best estimator lgbm's best error=0.4058
    [flaml.automl.logger: 08-22 20:57:59] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 08-22 20:58:17] {2466} INFO -  at 286.3s,	estimator lgbm's best error=0.4058,	best estimator lgbm's best error=0.4058
    [flaml.automl.logger: 08-22 20:58:17] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 08-22 20:58:37] {2466} INFO -  at 305.9s,	estimator lgbm's best error=0.4058,	best estimator lgbm's best error=0.4058
    [flaml.automl.logger: 08-22 20:58:37] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 08-22 20:58:57] {2466} INFO -  at 326.4s,	estimator lgbm's best error=0.4058,	best estimator lgbm's best error=0.4058
    [flaml.automl.logger: 08-22 20:58:57] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 08-22 20:59:16] {2466} INFO -  at 345.5s,	estimator lgbm's best error=0.4058,	best estimator lgbm's best error=0.4058
    [flaml.automl.logger: 08-22 20:59:16] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 08-22 20:59:34] {2466} INFO -  at 363.1s,	estimator lgbm's best error=0.4058,	best estimator lgbm's best error=0.4058
    [flaml.automl.logger: 08-22 20:59:34] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 08-22 21:00:01] {2466} INFO -  at 390.0s,	estimator lgbm's best error=0.3182,	best estimator lgbm's best error=0.3182
    [flaml.automl.logger: 08-22 21:00:01] {2282} INFO - iteration 9, current learner lgbm
    [flaml.automl.logger: 08-22 21:00:27] {2466} INFO -  at 416.6s,	estimator lgbm's best error=0.3182,	best estimator lgbm's best error=0.3182
    [flaml.automl.logger: 08-22 21:00:27] {2282} INFO - iteration 10, current learner lgbm
    [flaml.automl.logger: 08-22 21:00:57] {2466} INFO -  at 446.1s,	estimator lgbm's best error=0.3182,	best estimator lgbm's best error=0.3182
    [flaml.automl.logger: 08-22 21:00:57] {2282} INFO - iteration 11, current learner lgbm
    [flaml.automl.logger: 08-22 21:01:45] {2466} INFO -  at 494.8s,	estimator lgbm's best error=0.2559,	best estimator lgbm's best error=0.2559
    [flaml.automl.logger: 08-22 21:01:45] {2282} INFO - iteration 12, current learner lgbm
    [flaml.automl.logger: 08-22 21:02:12] {2466} INFO -  at 521.5s,	estimator lgbm's best error=0.2559,	best estimator lgbm's best error=0.2559
    [flaml.automl.logger: 08-22 21:02:12] {2282} INFO - iteration 13, current learner lgbm
    [flaml.automl.logger: 08-22 21:02:53] {2466} INFO -  at 562.0s,	estimator lgbm's best error=0.2559,	best estimator lgbm's best error=0.2559
    [flaml.automl.logger: 08-22 21:02:53] {2282} INFO - iteration 14, current learner lgbm
    [flaml.automl.logger: 08-22 21:03:48] {2466} INFO -  at 617.6s,	estimator lgbm's best error=0.2297,	best estimator lgbm's best error=0.2297
    [flaml.automl.logger: 08-22 21:03:48] {2282} INFO - iteration 15, current learner lgbm
    [flaml.automl.logger: 08-22 21:04:22] {2466} INFO -  at 651.3s,	estimator lgbm's best error=0.2297,	best estimator lgbm's best error=0.2297
    [flaml.automl.logger: 08-22 21:04:22] {2282} INFO - iteration 16, current learner lgbm
    [flaml.automl.logger: 08-22 21:05:52] {2466} INFO -  at 741.0s,	estimator lgbm's best error=0.2297,	best estimator lgbm's best error=0.2297
    ...
    [flaml.automl.logger: 08-22 21:25:29] {2282} INFO - iteration 25, current learner lgbm
    [flaml.automl.logger: 08-22 21:29:30] {2466} INFO -  at 2159.5s,	estimator lgbm's best error=0.2006,	best estimator lgbm's best error=0.2006
    [flaml.automl.logger: 08-22 21:29:30] {2282} INFO - iteration 26, current learner lgbm
    [flaml.automl.logger: 08-22 21:31:45] {2466} INFO -  at 2293.9s,	estimator lgbm's best error=0.2006,	best estimator lgbm's best error=0.2006
    [flaml.automl.logger: 08-22 21:31:45] {2282} INFO - iteration 27, current learner lgbm
    [flaml.automl.logger: 08-22 21:32:28] {2466} INFO -  at 2337.0s,	estimator lgbm's best error=0.2006,	best estimator lgbm's best error=0.2006
    [flaml.automl.logger: 08-22 21:32:28] {2282} INFO - iteration 28, current learner lgbm
    [flaml.automl.logger: 08-22 21:45:32] {2466} INFO -  at 3121.3s,	estimator lgbm's best error=0.2006,	best estimator lgbm's best error=0.2006
    [flaml.automl.logger: 08-22 21:45:32] {2282} INFO - iteration 29, current learner lgbm
    [flaml.automl.logger: 08-22 21:47:19] {2466} INFO -  at 3227.9s,	estimator lgbm's best error=0.2006,	best estimator lgbm's best error=0.2006
    [flaml.automl.logger: 08-22 21:47:19] {2282} INFO - iteration 30, current learner lgbm
    [flaml.automl.logger: 08-22 21:51:12] {2466} INFO -  at 3461.3s,	estimator lgbm's best error=0.2006,	best estimator lgbm's best error=0.2006
    [flaml.automl.logger: 08-22 21:51:12] {803} INFO - logging best model lgbm
    [flaml.automl.logger: 08-22 21:52:22] {2724} INFO - retrain lgbm for 70.0s
    [flaml.automl.logger: 08-22 21:52:22] {2727} INFO - retrained model: LGBMClassifier(colsample_bytree=np.float64(0.8788648628159693),
                   is_unbalance=True, learning_rate=np.float64(0.07642648485911652),
                   max_bin=255, min_child_samples=71, n_estimators=202, n_jobs=-1,
                   num_leaves=194, reg_alpha=0.0009765625,
                   reg_lambda=np.float64(0.9305515095247977), verbose=-1)
    [flaml.automl.logger: 08-22 21:52:22] {2729} INFO - Best MLflow run name: painted-perch-942_child_18
    [flaml.automl.logger: 08-22 21:52:22] {2730} INFO - Best MLflow run id: 86bc70bc44874de793354fe3cfa46f6e
    [flaml.automl.logger: 08-22 21:52:25] {2771} WARNING - Exception for record_state task run_9c68811a7ee64192affcb4b13204209d_requirements_updated: No such file or directory: '/Users/leorfinkelberg/Documents/Competitions/e_cup_2025_ml_challenge/mlruns/978153445019697853/9c68811a7ee64192affcb4b13204209d/artifacts/model/requirements.txt'
    [flaml.automl.logger: 08-22 21:52:25] {2009} INFO - fit succeeded
    [flaml.automl.logger: 08-22 21:52:25] {2010} INFO - Time taken to find the best model: 938.0540709495544
    """
    )
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
        time_budget=7200,
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
    [flaml.automl.logger: 08-22 22:03:04] {1752} INFO - task = classification
    [flaml.automl.logger: 08-22 22:03:04] {1763} INFO - Evaluation method: cv
    [flaml.automl.logger: 08-22 22:03:04] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 08-22 22:03:04] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 08-22 22:03:04] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 08-22 22:05:52] {2417} INFO - Estimated sufficient time budget=1675793s. Estimated necessary time budget=1676s.
    [flaml.automl.logger: 08-22 22:05:52] {2466} INFO -  at 204.2s,	estimator lgbm's best error=0.2006,	best estimator lgbm's best error=0.2006
    [flaml.automl.logger: 08-22 22:05:52] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 08-22 22:06:42] {2466} INFO -  at 254.0s,	estimator lgbm's best error=0.2006,	best estimator lgbm's best error=0.2006
    [flaml.automl.logger: 08-22 22:06:42] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 08-22 22:12:12] {2466} INFO -  at 584.3s,	estimator lgbm's best error=0.2006,	best estimator lgbm's best error=0.2006
    [flaml.automl.logger: 08-22 22:12:12] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 08-22 22:14:05] {2466} INFO -  at 697.2s,	estimator lgbm's best error=0.2006,	best estimator lgbm's best error=0.2006
    [flaml.automl.logger: 08-22 22:14:05] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 08-22 22:18:38] {2466} INFO -  at 970.0s,	estimator lgbm's best error=0.1959,	best estimator lgbm's best error=0.1959
    ...
    [flaml.automl.logger: 08-22 23:03:19] {2282} INFO - iteration 14, current learner lgbm
    [flaml.automl.logger: 08-22 23:08:36] {2466} INFO -  at 3968.4s,	estimator lgbm's best error=0.1959,	best estimator lgbm's best error=0.1959
    [flaml.automl.logger: 08-22 23:08:36] {2282} INFO - iteration 15, current learner lgbm
    [flaml.automl.logger: 08-22 23:10:47] {2466} INFO -  at 4099.6s,	estimator lgbm's best error=0.1959,	best estimator lgbm's best error=0.1959
    [flaml.automl.logger: 08-22 23:10:47] {2282} INFO - iteration 16, current learner lgbm
    [flaml.automl.logger: 08-22 23:18:51] {2466} INFO -  at 4583.2s,	estimator lgbm's best error=0.1959,	best estimator lgbm's best error=0.1959
    [flaml.automl.logger: 08-22 23:18:51] {2282} INFO - iteration 17, current learner lgbm
    [flaml.automl.logger: 08-22 23:20:19] {2466} INFO -  at 4670.9s,	estimator lgbm's best error=0.1959,	best estimator lgbm's best error=0.1959
    [flaml.automl.logger: 08-22 23:20:19] {2282} INFO - iteration 18, current learner lgbm
    [flaml.automl.logger: 08-22 23:36:48] {2466} INFO -  at 5660.1s,	estimator lgbm's best error=0.1953,	best estimator lgbm's best error=0.1953
    [flaml.automl.logger: 08-22 23:36:48] {2282} INFO - iteration 19, current learner lgbm
    [flaml.automl.logger: 08-22 23:42:42] {2466} INFO -  at 6013.8s,	estimator lgbm's best error=0.1953,	best estimator lgbm's best error=0.1953
    [flaml.automl.logger: 08-22 23:42:42] {2282} INFO - iteration 20, current learner lgbm
    [flaml.automl.logger: 08-23 00:02:19] {2466} INFO -  at 7191.7s,	estimator lgbm's best error=0.1953,	best estimator lgbm's best error=0.1953
    [flaml.automl.logger: 08-23 00:09:51] {2724} INFO - retrain lgbm for 451.4s
    [flaml.automl.logger: 08-23 00:09:51] {2727} INFO - retrained model: LGBMClassifier(colsample_bytree=np.float64(0.8570229195379688),
                   is_unbalance=True,
                   learning_rate=np.float64(0.028419861921852693), max_bin=511,
                   min_child_samples=38, n_estimators=1051, n_jobs=-1,
                   num_leaves=181, reg_alpha=0.0009765625,
                   reg_lambda=np.float64(3.317767134478085), verbose=-1)
    [flaml.automl.logger: 08-23 00:09:51] {2009} INFO - fit succeeded
    [flaml.automl.logger: 08-23 00:09:51] {2010} INFO - Time taken to find the best model: 5660.060078859329
    """
    )
    return


@app.cell
def _(X_test_with_numeric_cols, automl_second_part, pd):
    submission = pd.DataFrame({
        "id": X_test_with_numeric_cols.index, 
        "prediction": automl_second_part.predict(X_test_with_numeric_cols),
    })

    submission.to_csv("./submission.csv", index=False)
    return


@app.cell
def _(pd):
    pd.read_csv("./submission.csv")["prediction"].value_counts()
    return


if __name__ == "__main__":
    app.run()
