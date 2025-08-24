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
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.decomposition import PCA, KernelPCA

    from flaml.automl.automl import AutoML
    from flaml import tune
    return AutoML, PCA, QuantileTransformer, ce, mlflow, mo, np, pd


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
    IMAGE_N_COMPONENTS = 350
    TEXT_N_COMPONENTS = 150 
    return IMAGE_N_COMPONENTS, SEED, TASK, TEXT_N_COMPONENTS


@app.cell
def _():
    # IMAGE_N_COMPONENTS = 100, TEXT_N_COMPONENTS = 50 -> 0.1915
    # IMAGE_N_COMPONENTS = 200, TEXT_N_COMPONENTS = 100 -> 0.1985
    # IMAGE_N_COMPONENTS = 50, TEXT_N_COMPONENTS = 25 -> 0.1899
    return


@app.cell
def _():
    # mlflow.autolog()
    return


@app.cell
def _(pd):
    train = pd.read_csv("./ml_ozon_сounterfeit_data/ml_ozon_сounterfeit_train.csv", index_col=0)
    return (train,)


@app.cell
def _(train):
    original_cols = train.columns.tolist()
    original_cols.remove("resolution")
    original_cols.remove("ItemID")
    original_cols.remove("description")
    original_cols.remove("name_rus")
    return (original_cols,)


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
    train_with_text_image_embds = train_with_text_embds.join(train_with_image_embds, on="ItemID", how="left")
    test_with_text_image_embds = test_with_text_embds.join(test_with_image_embds, on="ItemID", how="left")
    return test_with_text_image_embds, train_with_text_image_embds


@app.cell
def _(train_with_text_image_embds):
    image_embds_col_names = [col_name for col_name in train_with_text_image_embds.columns if col_name.startswith("vit-base-patch16-224-in21k")]
    return (image_embds_col_names,)


@app.cell
def _():
    """
    IMAGE_WINDOW = 5

    train_with_text_image_embds[image_embds_col_names] = train_with_text_image_embds.apply(lambda row: row[image_embds_col_names].rolling(window=IMAGE_WINDOW).mean().fillna(0), axis=1)

    test_with_text_image_embds[image_embds_col_names] = test_with_text_image_embds.apply(lambda row: row[image_embds_col_names].rolling(window=IMAGE_WINDOW).mean().fillna(0), axis=1)
    """
    return


@app.cell
def _(train_with_text_image_embds):
    desc_embds_col_names = [col_name for col_name in train_with_text_image_embds.columns if col_name.startswith("desc_embed")]
    name_rus_embds_col_names = [col_name for col_name in train_with_text_image_embds.columns if col_name.startswith("name_rus_embed")]
    return desc_embds_col_names, name_rus_embds_col_names


@app.cell
def _():
    """
    TEXT_WINDOW = 5

    train_with_text_image_embds[desc_embds_col_names] = train_with_text_image_embds.apply(lambda row: row[desc_embds_col_names].rolling(window=TEXT_WINDOW).mean().fillna(0), axis=1)

    test_with_text_image_embds[desc_embds_col_names] = test_with_text_image_embds.apply(lambda row: row[desc_embds_col_names].rolling(window=TEXT_WINDOW).mean().fillna(0), axis=1)

    train_with_text_image_embds[name_rus_embds_col_names] = train_with_text_image_embds.apply(lambda row: row[name_rus_embds_col_names].rolling(window=TEXT_WINDOW).mean().fillna(0), axis=1)

    test_with_text_image_embds[name_rus_embds_col_names] = test_with_text_image_embds.apply(lambda row: row[name_rus_embds_col_names].rolling(window=TEXT_WINDOW).mean().fillna(0), axis=1)
    """
    return


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
    cols_without_target.remove("ItemID")

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
def _(X_train):
    X_train.isna().sum()
    return


@app.cell
def _(X_test):
    X_test.isna().sum()
    return


@app.cell
def _(ce):
    encoder = ce.CountEncoder(
        cols=[
            "CommercialTypeName4",
            "brand_name",
        ],
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
def _(X_train_with_numeric_cols):
    X_train_with_numeric_cols.shape
    return


@app.cell
def _(X_test_with_numeric_cols):
    X_test_with_numeric_cols.shape
    return


@app.cell
def _(X_test_with_numeric_cols, X_train_with_numeric_cols):
    X_train_with_numeric_cols_wo_nan = X_train_with_numeric_cols.fillna(X_train_with_numeric_cols.mean())
    X_test_with_numeric_cols_wo_nan = X_test_with_numeric_cols.fillna(X_test_with_numeric_cols.mean())
    return X_test_with_numeric_cols_wo_nan, X_train_with_numeric_cols_wo_nan


@app.cell
def _(X_train_with_numeric_cols_wo_nan, image_embds_col_names):
    X_train_with_numeric_cols_wo_nan[image_embds_col_names].iloc[0].plot()
    return


@app.cell
def _(X_train_with_numeric_cols_wo_nan, desc_embds_col_names):
    X_train_with_numeric_cols_wo_nan[desc_embds_col_names].iloc[0].plot()
    return


@app.cell
def _(X_train_with_numeric_cols_wo_nan, name_rus_embds_col_names):
    X_train_with_numeric_cols_wo_nan[name_rus_embds_col_names].iloc[0].plot()
    return


@app.cell
def _(IMAGE_N_COMPONENTS, PCA, SEED):
    image_pca = PCA(n_components=IMAGE_N_COMPONENTS, random_state=SEED)
    return (image_pca,)


@app.cell
def _(IMAGE_N_COMPONENTS):
    image_embds_pca_col_names = [f"image_embds_pca{idx}" for idx in range(IMAGE_N_COMPONENTS)]
    return (image_embds_pca_col_names,)


@app.cell
def _(
    X_test_with_numeric_cols_wo_nan,
    X_train_with_numeric_cols_wo_nan,
    image_embds_col_names,
    image_embds_pca_col_names,
    image_pca,
    pd,
):
    X_train_image_embds_pca = pd.DataFrame(
        image_pca.fit_transform(X_train_with_numeric_cols_wo_nan[image_embds_col_names]),
        columns=image_embds_pca_col_names,
        index=X_train_with_numeric_cols_wo_nan.index
    )

    X_test_image_embds_pca = pd.DataFrame(
        image_pca.transform(X_test_with_numeric_cols_wo_nan[image_embds_col_names]),
        columns=image_embds_pca_col_names,
        index=X_test_with_numeric_cols_wo_nan.index
    )
    return X_test_image_embds_pca, X_train_image_embds_pca


@app.cell
def _(X_train_image_embds_pca):
    X_train_image_embds_pca.shape
    return


@app.cell
def _(X_test_image_embds_pca):
    X_test_image_embds_pca.shape
    return


@app.cell
def _(PCA, SEED, TEXT_N_COMPONENTS):
    desc_pca = PCA(n_components=TEXT_N_COMPONENTS, random_state=SEED)
    name_rus_pca = PCA(n_components=TEXT_N_COMPONENTS, random_state=SEED)
    return desc_pca, name_rus_pca


@app.cell
def _(TEXT_N_COMPONENTS):
    desc_embds_pca_col_names = [f"desc_embds_pca{idx}" for idx in range(TEXT_N_COMPONENTS)]
    name_rus_embds_pca_col_names = [f"name_rus_embds_pca{idx}" for idx in range(TEXT_N_COMPONENTS)]
    return desc_embds_pca_col_names, name_rus_embds_pca_col_names


@app.cell
def _(
    X_test_with_numeric_cols_wo_nan,
    X_train_with_numeric_cols_wo_nan,
    desc_embds_col_names,
    desc_embds_pca_col_names,
    desc_pca,
    pd,
):
    X_train_desc_embds_pca = pd.DataFrame(
        desc_pca.fit_transform(X_train_with_numeric_cols_wo_nan[desc_embds_col_names]),
        columns=desc_embds_pca_col_names,
        index=X_train_with_numeric_cols_wo_nan.index
    )

    X_test_desc_embds_pca = pd.DataFrame(
        desc_pca.transform(X_test_with_numeric_cols_wo_nan[desc_embds_col_names]),
        columns=desc_embds_pca_col_names,
        index=X_test_with_numeric_cols_wo_nan.index
    )
    return X_test_desc_embds_pca, X_train_desc_embds_pca


@app.cell
def _(
    X_test_with_numeric_cols_wo_nan,
    X_train_with_numeric_cols_wo_nan,
    name_rus_embds_col_names,
    name_rus_embds_pca_col_names,
    name_rus_pca,
    pd,
):
    X_train_name_rus_embds_pca = pd.DataFrame(
        name_rus_pca.fit_transform(X_train_with_numeric_cols_wo_nan[name_rus_embds_col_names]),
        columns=name_rus_embds_pca_col_names,
        index=X_train_with_numeric_cols_wo_nan.index
    )

    X_test_name_rus_embds_pca = pd.DataFrame(
        name_rus_pca.transform(X_test_with_numeric_cols_wo_nan[name_rus_embds_col_names]),
        columns=name_rus_embds_pca_col_names,
        index=X_test_with_numeric_cols_wo_nan.index
    )
    return X_test_name_rus_embds_pca, X_train_name_rus_embds_pca


@app.cell
def _(
    X_test_desc_embds_pca,
    X_test_image_embds_pca,
    X_test_name_rus_embds_pca,
    X_test_with_numeric_cols_wo_nan,
    X_train_desc_embds_pca,
    X_train_image_embds_pca,
    X_train_name_rus_embds_pca,
    X_train_with_numeric_cols_wo_nan,
    original_cols,
    pd,
):
    X_train_final = pd.concat([
        X_train_with_numeric_cols_wo_nan[original_cols],
        X_train_desc_embds_pca,
        X_train_name_rus_embds_pca,
        X_train_image_embds_pca,
    ], axis=1)

    X_test_final = pd.concat([
        X_test_with_numeric_cols_wo_nan[original_cols],
        X_test_desc_embds_pca,
        X_test_name_rus_embds_pca,
        X_test_image_embds_pca,
    ], axis=1)
    return X_test_final, X_train_final


@app.cell
def _(X_train_final):
    X_train_final.shape
    return


@app.cell
def _(X_test_final):
    X_test_final.shape
    return


@app.cell
def _(QuantileTransformer, SEED):
    quantile_transformer = QuantileTransformer(output_distribution="normal", n_quantiles=350, random_state=SEED)
    return (quantile_transformer,)


@app.cell
def _(X_train_final, original_cols, quantile_transformer):
    X_train_final[original_cols] = quantile_transformer.fit_transform(X_train_final[original_cols])
    return


@app.cell
def _(X_test_final, original_cols, quantile_transformer):
    X_test_final[original_cols] = quantile_transformer.transform(X_test_final[original_cols])
    return


@app.cell
def _(X_train_final):
    X_train_final.shape
    return


@app.cell
def _(X_test_final):
    X_test_final.shape
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
def _(SEED, TASK, X_train_final, automl, custom_hp, mlflow, y_train):
    # mlflow.set_experiment("flaml-lgbm-text-image-embds-count-enc-budget=3600-balanced")
    with mlflow.start_run():
        automl.fit(
            X_train_final,
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
    [flaml.automl.logger: 08-24 03:53:53] {1752} INFO - task = classification
    [flaml.automl.logger: 08-24 03:53:53] {1763} INFO - Evaluation method: cv
    [flaml.automl.logger: 08-24 03:54:21] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 08-24 03:54:21] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 08-24 03:54:21] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 08-24 03:54:24] {2417} INFO - Estimated sufficient time budget=26088s. Estimated necessary time budget=26s.
    [flaml.automl.logger: 08-24 03:54:24] {2466} INFO -  at 35.4s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 08-24 03:54:24] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 08-24 03:54:26] {2466} INFO -  at 38.1s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 08-24 03:54:26] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 08-24 03:54:29] {2466} INFO -  at 40.9s,	estimator lgbm's best error=0.3918,	best estimator lgbm's best error=0.3918
    [flaml.automl.logger: 08-24 03:54:29] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 08-24 03:54:32] {2466} INFO -  at 43.7s,	estimator lgbm's best error=0.3918,	best estimator lgbm's best error=0.3918
    [flaml.automl.logger: 08-24 03:54:32] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 08-24 03:54:35] {2466} INFO -  at 46.7s,	estimator lgbm's best error=0.3918,	best estimator lgbm's best error=0.3918
    [flaml.automl.logger: 08-24 03:54:35] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 08-24 03:54:39] {2466} INFO -  at 50.2s,	estimator lgbm's best error=0.3918,	best estimator lgbm's best error=0.3918
    [flaml.automl.logger: 08-24 03:54:39] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 08-24 03:54:41] {2466} INFO -  at 52.9s,	estimator lgbm's best error=0.3918,	best estimator lgbm's best error=0.3918
    [flaml.automl.logger: 08-24 03:54:41] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 08-24 03:54:44] {2466} INFO -  at 55.5s,	estimator lgbm's best error=0.3918,	best estimator lgbm's best error=0.3918
    [flaml.automl.logger: 08-24 03:54:44] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 08-24 03:54:50] {2466} INFO -  at 61.5s,	estimator lgbm's best error=0.3269,	best estimator lgbm's best error=0.3269
    [flaml.automl.logger: 08-24 03:54:50] {2282} INFO - iteration 9, current learner lgbm
    [flaml.automl.logger: 08-24 03:54:56] {2466} INFO -  at 67.7s,	estimator lgbm's best error=0.3269,	best estimator lgbm's best error=0.3269
    [flaml.automl.logger: 08-24 03:54:56] {2282} INFO - iteration 10, current learner lgbm
    [flaml.automl.logger: 08-24 03:55:02] {2466} INFO -  at 73.3s,	estimator lgbm's best error=0.3269,	best estimator lgbm's best error=0.3269
    [flaml.automl.logger: 08-24 03:55:02] {2282} INFO - iteration 11, current learner lgbm
    [flaml.automl.logger: 08-24 03:55:12] {2466} INFO -  at 83.4s,	estimator lgbm's best error=0.2623,	best estimator lgbm's best error=0.2623
    [flaml.automl.logger: 08-24 03:55:12] {2282} INFO - iteration 12, current learner lgbm
    [flaml.automl.logger: 08-24 03:55:18] {2466} INFO -  at 89.4s,	estimator lgbm's best error=0.2623,	best estimator lgbm's best error=0.2623
    [flaml.automl.logger: 08-24 03:55:18] {2282} INFO - iteration 13, current learner lgbm
    [flaml.automl.logger: 08-24 03:55:26] {2466} INFO -  at 97.7s,	estimator lgbm's best error=0.2623,	best estimator lgbm's best error=0.2623
    [flaml.automl.logger: 08-24 03:55:26] {2282} INFO - iteration 14, current learner lgbm
    [flaml.automl.logger: 08-24 03:55:38] {2466} INFO -  at 109.9s,	estimator lgbm's best error=0.2308,	best estimator lgbm's best error=0.2308
    [flaml.automl.logger: 08-24 03:55:38] {2282} INFO - iteration 15, current learner lgbm
    [flaml.automl.logger: 08-24 03:55:45] {2466} INFO -  at 116.5s,	estimator lgbm's best error=0.2308,	best estimator lgbm's best error=0.2308
    [flaml.automl.logger: 08-24 03:55:45] {2282} INFO - iteration 16, current learner lgbm
    [flaml.automl.logger: 08-24 03:56:09] {2466} INFO -  at 140.2s,	estimator lgbm's best error=0.2308,	best estimator lgbm's best error=0.2308
    [flaml.automl.logger: 08-24 03:56:09] {2282} INFO - iteration 17, current learner lgbm
    [flaml.automl.logger: 08-24 03:56:13] {2466} INFO -  at 144.7s,	estimator lgbm's best error=0.2308,	best estimator lgbm's best error=0.2308
    [flaml.automl.logger: 08-24 03:56:13] {2282} INFO - iteration 18, current learner lgbm
    [flaml.automl.logger: 08-24 03:56:59] {2466} INFO -  at 190.4s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 03:56:59] {2282} INFO - iteration 19, current learner lgbm
    [flaml.automl.logger: 08-24 03:57:14] {2466} INFO -  at 205.3s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 03:57:14] {2282} INFO - iteration 20, current learner lgbm
    [flaml.automl.logger: 08-24 03:59:27] {2466} INFO -  at 338.4s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 03:59:27] {2282} INFO - iteration 21, current learner lgbm
    [flaml.automl.logger: 08-24 04:00:13] {2466} INFO -  at 384.6s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 04:00:13] {2282} INFO - iteration 22, current learner lgbm
    [flaml.automl.logger: 08-24 04:00:45] {2466} INFO -  at 417.2s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 04:00:45] {2282} INFO - iteration 23, current learner lgbm
    [flaml.automl.logger: 08-24 04:01:39] {2466} INFO -  at 470.2s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 04:01:39] {2282} INFO - iteration 24, current learner lgbm
    [flaml.automl.logger: 08-24 04:02:14] {2466} INFO -  at 505.5s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 04:02:14] {2282} INFO - iteration 25, current learner lgbm
    [flaml.automl.logger: 08-24 04:03:48] {2466} INFO -  at 599.2s,	estimator lgbm's best error=0.1960,	best estimator lgbm's best error=0.1960
    [flaml.automl.logger: 08-24 04:03:48] {2282} INFO - iteration 26, current learner lgbm
    [flaml.automl.logger: 08-24 04:04:34] {2466} INFO -  at 645.7s,	estimator lgbm's best error=0.1960,	best estimator lgbm's best error=0.1960
    [flaml.automl.logger: 08-24 04:04:34] {2282} INFO - iteration 27, current learner lgbm
    [flaml.automl.logger: 08-24 04:04:48] {2466} INFO -  at 660.1s,	estimator lgbm's best error=0.1960,	best estimator lgbm's best error=0.1960
    [flaml.automl.logger: 08-24 04:04:48] {2282} INFO - iteration 28, current learner lgbm
    [flaml.automl.logger: 08-24 04:26:38] {2466} INFO -  at 1969.3s,	estimator lgbm's best error=0.1960,	best estimator lgbm's best error=0.1960
    [flaml.automl.logger: 08-24 04:26:38] {2282} INFO - iteration 29, current learner lgbm
    [flaml.automl.logger: 08-24 04:30:33] {2466} INFO -  at 2204.5s,	estimator lgbm's best error=0.1960,	best estimator lgbm's best error=0.1960
    [flaml.automl.logger: 08-24 04:30:33] {2282} INFO - iteration 30, current learner lgbm
    [flaml.automl.logger: 08-24 04:35:35] {2466} INFO -  at 2506.4s,	estimator lgbm's best error=0.1960,	best estimator lgbm's best error=0.1960
    [flaml.automl.logger: 08-24 04:35:35] {2282} INFO - iteration 31, current learner lgbm
    [flaml.automl.logger: 08-24 04:41:12] {2466} INFO -  at 2843.9s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 04:41:12] {2282} INFO - iteration 32, current learner lgbm
    [flaml.automl.logger: 08-24 04:42:45] {2466} INFO -  at 2936.5s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 04:42:45] {2282} INFO - iteration 33, current learner lgbm
    [flaml.automl.logger: 08-24 04:45:35] {2466} INFO -  at 3106.5s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 04:45:35] {2282} INFO - iteration 34, current learner lgbm
    [flaml.automl.logger: 08-24 04:46:00] {2466} INFO -  at 3131.3s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 04:46:00] {2282} INFO - iteration 35, current learner lgbm
    [flaml.automl.logger: 08-24 04:50:33] {2466} INFO -  at 3404.7s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 04:50:33] {803} INFO - logging best model lgbm
    [flaml.automl.logger: 08-24 04:51:30] {2724} INFO - retrain lgbm for 56.6s
    [flaml.automl.logger: 08-24 04:51:30] {2727} INFO - retrained model: LGBMClassifier(is_unbalance=True, learning_rate=np.float64(0.0952587288316166),
                   max_bin=127, min_child_samples=122, n_estimators=342, n_jobs=-1,
                   num_leaves=266, reg_alpha=0.0009765625,
                   reg_lambda=np.float64(9.461164679607368), verbose=-1)
    [flaml.automl.logger: 08-24 04:51:30] {2729} INFO - Best MLflow run name: serious-cod-383_child_31
    [flaml.automl.logger: 08-24 04:51:30] {2730} INFO - Best MLflow run id: a2cc9657bd7d430b83851f062028cc64
    [flaml.automl.logger: 08-24 04:51:32] {2771} WARNING - Exception for record_state task run_18f64fcf34e147e19ca6caf85c4b8213_requirements_updated: No such file or directory: '/Users/leorfinkelberg/Documents/Competitions/e_cup_2025_ml_challenge/mlruns/0/18f64fcf34e147e19ca6caf85c4b8213/artifacts/model/requirements.txt'
    [flaml.automl.logger: 08-24 04:51:32] {2009} INFO - fit succeeded
    [flaml.automl.logger: 08-24 04:51:32] {2010} INFO - Time taken to find the best model: 2843.910061120987
    """
    )
    return


@app.cell
def _(X_test_final, automl, pd):
    submission = pd.DataFrame({
        "id": X_test_final.index, 
        "prediction": automl.predict(X_test_final),
    })

    submission.to_csv("./submission.csv", index=False)
    return


@app.cell
def _(pd):
    pd.read_csv("./submission.csv")["prediction"].value_counts()
    return


@app.cell
def _(AutoML, SEED, TASK, X_train_final, automl, custom_hp, y_train):
    automl_second_part = AutoML()

    automl_second_part.fit(
        X_train_final, 
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
        time_budget=10_800,
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
    [flaml.automl.logger: 08-23 01:19:26] {1752} INFO - task = classification
    [flaml.automl.logger: 08-23 01:19:26] {1763} INFO - Evaluation method: cv
    [flaml.automl.logger: 08-23 01:19:26] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 08-23 01:19:26] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 08-23 01:19:26] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 08-23 01:36:19] {2417} INFO - Estimated sufficient time budget=10126321s. Estimated necessary time budget=10126s.
    [flaml.automl.logger: 08-23 01:36:19] {2466} INFO -  at 1048.3s,	estimator lgbm's best error=0.1953,	best estimator lgbm's best error=0.1953
    [flaml.automl.logger: 08-23 01:36:19] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 08-23 01:39:49] {2466} INFO -  at 1258.4s,	estimator lgbm's best error=0.1953,	best estimator lgbm's best error=0.1953
    [flaml.automl.logger: 08-23 01:39:49] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 08-23 02:13:08] {2466} INFO -  at 3256.9s,	estimator lgbm's best error=0.1953,	best estimator lgbm's best error=0.1953
    [flaml.automl.logger: 08-23 02:13:08] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 08-23 02:22:59] {2466} INFO -  at 3848.5s,	estimator lgbm's best error=0.1953,	best estimator lgbm's best error=0.1953
    [flaml.automl.logger: 08-23 02:22:59] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 08-23 02:47:36] {2466} INFO -  at 5325.7s,	estimator lgbm's best error=0.1928,	best estimator lgbm's best error=0.1928
    [flaml.automl.logger: 08-23 02:47:36] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 08-23 03:13:41] {2466} INFO -  at 6889.9s,	estimator lgbm's best error=0.1928,	best estimator lgbm's best error=0.1928
    [flaml.automl.logger: 08-23 03:13:41] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 08-23 03:22:02] {2466} INFO -  at 7390.9s,	estimator lgbm's best error=0.1928,	best estimator lgbm's best error=0.1928
    [flaml.automl.logger: 08-23 03:22:02] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 08-23 03:24:39] {2466} INFO -  at 7548.1s,	estimator lgbm's best error=0.1928,	best estimator lgbm's best error=0.1928
    [flaml.automl.logger: 08-23 03:24:39] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 08-23 04:19:09] {2466} INFO -  at 10818.2s,	estimator lgbm's best error=0.1928,	best estimator lgbm's best error=0.1928
    [flaml.automl.logger: 08-23 04:29:57] {2724} INFO - retrain lgbm for 647.9s
    [flaml.automl.logger: 08-23 04:29:57] {2727} INFO - retrained model: LGBMClassifier(colsample_bytree=np.float64(0.8865145401052198),
                   is_unbalance=True, learning_rate=np.float64(0.03992076882139407),
                   max_bin=1023, min_child_samples=36, n_estimators=2194, n_jobs=-1,
                   num_leaves=81, reg_alpha=0.0009765625,
                   reg_lambda=np.float64(20.188074226282694), verbose=-1)
    [flaml.automl.logger: 08-23 04:29:57] {2009} INFO - fit succeeded
    [flaml.automl.logger: 08-23 04:29:57] {2010} INFO - Time taken to find the best model: 5325.731877088547
    """
    )
    return


@app.cell
def _(automl):
    automl.save_best_config("./best_config_f1=0.7201.txt")
    return


@app.cell
def _(automl_second_part):
    automl_second_part.save_best_config("./best_config_f1=0.7261.txt")
    return


@app.cell
def _(automl_third_part):
    automl_third_part.save_best_config("./best_config_3part.txt")
    return


@app.cell
def _(X_test_with_numeric_cols, automl_third_part, pd):
    submission = pd.DataFrame({
        "id": X_test_with_numeric_cols.index, 
        "prediction": automl_third_part.predict(X_test_with_numeric_cols),
    })

    submission.to_csv("./submission.csv", index=False)
    return


@app.cell
def _(pd):
    pd.read_csv("./submission_f1=0.7086.csv")["prediction"].value_counts()
    return


@app.cell
def _(pd):
    pd.read_csv("./submission_f1=0.7191.csv")["prediction"].value_counts()
    return


@app.cell
def _(pd):
    pd.read_csv("./submission_f1=0.7261.csv")["prediction"].value_counts()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
