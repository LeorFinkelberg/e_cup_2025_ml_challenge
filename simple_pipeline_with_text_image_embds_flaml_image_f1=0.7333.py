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
    IMAGE_N_COMPONENTS = 100
    TEXT_N_COMPONENTS = 50 
    return IMAGE_N_COMPONENTS, SEED, TASK, TEXT_N_COMPONENTS


@app.cell
def _():
    # IMAGE_N_COMPONENTS = 100, TEXT_N_COMPONENTS = 50 -> 0.1879
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
            "SellerID",
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
def _(AutoML, SEED, TASK, X_train_final, custom_hp, y_train):
    ### ЗАПУСК С РАННЕЙ ОСТАНОВКОЙ

    automl = AutoML()
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
        eval_method="holdout",
        metric="f1",
        split_type="stratified",
        custom_hp=custom_hp,
        early_stop=True,
        seed=SEED,
    )
    return (automl,)


@app.cell
def _(AutoML, SEED, TASK, X_train_final, custom_hp, mlflow, y_train):
    ### ЗАПУСК БЕЗ РАННЕЙ ОСТАНОВКИ

    automl = AutoML()
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
    return (automl,)


@app.cell
def _(mo):
    mo.md(
        r"""
    [flaml.automl.logger: 08-24 14:27:34] {1752} INFO - task = classification
    [flaml.automl.logger: 08-24 14:27:34] {1763} INFO - Evaluation method: cv
    [flaml.automl.logger: 08-24 14:28:02] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 08-24 14:28:02] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 08-24 14:28:02] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 08-24 14:28:05] {2417} INFO - Estimated sufficient time budget=27836s. Estimated necessary time budget=28s.
    [flaml.automl.logger: 08-24 14:28:05] {2466} INFO -  at 36.5s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 08-24 14:28:05] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 08-24 14:28:08] {2466} INFO -  at 39.3s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 08-24 14:28:08] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 08-24 14:28:11] {2466} INFO -  at 42.3s,	estimator lgbm's best error=0.3918,	best estimator lgbm's best error=0.3918
    [flaml.automl.logger: 08-24 14:28:11] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 08-24 14:28:14] {2466} INFO -  at 45.3s,	estimator lgbm's best error=0.3918,	best estimator lgbm's best error=0.3918
    [flaml.automl.logger: 08-24 14:28:14] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 08-24 14:28:17] {2466} INFO -  at 48.5s,	estimator lgbm's best error=0.3918,	best estimator lgbm's best error=0.3918
    [flaml.automl.logger: 08-24 14:28:17] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 08-24 14:28:20] {2466} INFO -  at 52.2s,	estimator lgbm's best error=0.3918,	best estimator lgbm's best error=0.3918
    [flaml.automl.logger: 08-24 14:28:20] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 08-24 14:28:23] {2466} INFO -  at 55.1s,	estimator lgbm's best error=0.3918,	best estimator lgbm's best error=0.3918
    [flaml.automl.logger: 08-24 14:28:23] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 08-24 14:28:26] {2466} INFO -  at 57.8s,	estimator lgbm's best error=0.3918,	best estimator lgbm's best error=0.3918
    [flaml.automl.logger: 08-24 14:28:26] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 08-24 14:28:32] {2466} INFO -  at 64.0s,	estimator lgbm's best error=0.3269,	best estimator lgbm's best error=0.3269
    [flaml.automl.logger: 08-24 14:28:32] {2282} INFO - iteration 9, current learner lgbm
    [flaml.automl.logger: 08-24 14:28:39] {2466} INFO -  at 70.6s,	estimator lgbm's best error=0.3269,	best estimator lgbm's best error=0.3269
    [flaml.automl.logger: 08-24 14:28:39] {2282} INFO - iteration 10, current learner lgbm
    [flaml.automl.logger: 08-24 14:28:45] {2466} INFO -  at 76.5s,	estimator lgbm's best error=0.3269,	best estimator lgbm's best error=0.3269
    [flaml.automl.logger: 08-24 14:28:45] {2282} INFO - iteration 11, current learner lgbm
    [flaml.automl.logger: 08-24 14:28:55] {2466} INFO -  at 87.1s,	estimator lgbm's best error=0.2623,	best estimator lgbm's best error=0.2623
    [flaml.automl.logger: 08-24 14:28:55] {2282} INFO - iteration 12, current learner lgbm
    [flaml.automl.logger: 08-24 14:29:02] {2466} INFO -  at 93.3s,	estimator lgbm's best error=0.2623,	best estimator lgbm's best error=0.2623
    [flaml.automl.logger: 08-24 14:29:02] {2282} INFO - iteration 13, current learner lgbm
    [flaml.automl.logger: 08-24 14:29:10] {2466} INFO -  at 101.9s,	estimator lgbm's best error=0.2623,	best estimator lgbm's best error=0.2623
    [flaml.automl.logger: 08-24 14:29:10] {2282} INFO - iteration 14, current learner lgbm
    [flaml.automl.logger: 08-24 14:29:23] {2466} INFO -  at 114.5s,	estimator lgbm's best error=0.2308,	best estimator lgbm's best error=0.2308
    [flaml.automl.logger: 08-24 14:29:23] {2282} INFO - iteration 15, current learner lgbm
    [flaml.automl.logger: 08-24 14:29:30] {2466} INFO -  at 121.5s,	estimator lgbm's best error=0.2308,	best estimator lgbm's best error=0.2308
    [flaml.automl.logger: 08-24 14:29:30] {2282} INFO - iteration 16, current learner lgbm
    [flaml.automl.logger: 08-24 14:29:54] {2466} INFO -  at 146.0s,	estimator lgbm's best error=0.2308,	best estimator lgbm's best error=0.2308
    [flaml.automl.logger: 08-24 14:29:54] {2282} INFO - iteration 17, current learner lgbm
    [flaml.automl.logger: 08-24 14:29:59] {2466} INFO -  at 150.7s,	estimator lgbm's best error=0.2308,	best estimator lgbm's best error=0.2308
    [flaml.automl.logger: 08-24 14:29:59] {2282} INFO - iteration 18, current learner lgbm
    [flaml.automl.logger: 08-24 14:30:47] {2466} INFO -  at 198.3s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 14:30:47] {2282} INFO - iteration 19, current learner lgbm
    [flaml.automl.logger: 08-24 14:31:02] {2466} INFO -  at 213.3s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 14:31:02] {2282} INFO - iteration 20, current learner lgbm
    [flaml.automl.logger: 08-24 14:33:17] {2466} INFO -  at 348.3s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 14:33:17] {2282} INFO - iteration 21, current learner lgbm
    [flaml.automl.logger: 08-24 14:34:05] {2466} INFO -  at 396.8s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 14:34:05] {2282} INFO - iteration 22, current learner lgbm
    [flaml.automl.logger: 08-24 14:34:38] {2466} INFO -  at 430.2s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 14:34:38] {2282} INFO - iteration 23, current learner lgbm
    [flaml.automl.logger: 08-24 14:35:33] {2466} INFO -  at 484.3s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 14:35:33] {2282} INFO - iteration 24, current learner lgbm
    [flaml.automl.logger: 08-24 14:36:08] {2466} INFO -  at 520.1s,	estimator lgbm's best error=0.1964,	best estimator lgbm's best error=0.1964
    [flaml.automl.logger: 08-24 14:36:08] {2282} INFO - iteration 25, current learner lgbm
    [flaml.automl.logger: 08-24 14:37:43] {2466} INFO -  at 614.7s,	estimator lgbm's best error=0.1960,	best estimator lgbm's best error=0.1960
    [flaml.automl.logger: 08-24 14:37:43] {2282} INFO - iteration 26, current learner lgbm
    [flaml.automl.logger: 08-24 14:38:31] {2466} INFO -  at 662.9s,	estimator lgbm's best error=0.1960,	best estimator lgbm's best error=0.1960
    [flaml.automl.logger: 08-24 14:38:31] {2282} INFO - iteration 27, current learner lgbm
    [flaml.automl.logger: 08-24 14:38:46] {2466} INFO -  at 677.7s,	estimator lgbm's best error=0.1960,	best estimator lgbm's best error=0.1960
    [flaml.automl.logger: 08-24 14:38:46] {2282} INFO - iteration 28, current learner lgbm
    [flaml.automl.logger: 08-24 14:47:02] {2466} INFO -  at 1174.0s,	estimator lgbm's best error=0.1960,	best estimator lgbm's best error=0.1960
    [flaml.automl.logger: 08-24 14:47:02] {2282} INFO - iteration 29, current learner lgbm
    [flaml.automl.logger: 08-24 14:47:52] {2466} INFO -  at 1223.8s,	estimator lgbm's best error=0.1960,	best estimator lgbm's best error=0.1960
    [flaml.automl.logger: 08-24 14:47:52] {2282} INFO - iteration 30, current learner lgbm
    [flaml.automl.logger: 08-24 14:49:28] {2466} INFO -  at 1319.6s,	estimator lgbm's best error=0.1960,	best estimator lgbm's best error=0.1960
    [flaml.automl.logger: 08-24 14:49:28] {2282} INFO - iteration 31, current learner lgbm
    [flaml.automl.logger: 08-24 14:50:53] {2466} INFO -  at 1404.9s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 14:50:53] {2282} INFO - iteration 32, current learner lgbm
    [flaml.automl.logger: 08-24 14:52:26] {2466} INFO -  at 1498.1s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 14:52:26] {2282} INFO - iteration 33, current learner lgbm
    [flaml.automl.logger: 08-24 14:55:17] {2466} INFO -  at 1668.8s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 14:55:17] {2282} INFO - iteration 34, current learner lgbm
    [flaml.automl.logger: 08-24 14:55:42] {2466} INFO -  at 1693.4s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 14:55:42] {2282} INFO - iteration 35, current learner lgbm
    [flaml.automl.logger: 08-24 15:00:18] {2466} INFO -  at 1969.9s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 15:00:18] {2282} INFO - iteration 36, current learner lgbm
    [flaml.automl.logger: 08-24 15:00:45] {2466} INFO -  at 1996.5s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 15:00:45] {2282} INFO - iteration 37, current learner lgbm
    [flaml.automl.logger: 08-24 15:01:37] {2466} INFO -  at 2048.8s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 15:01:37] {2282} INFO - iteration 38, current learner lgbm
    [flaml.automl.logger: 08-24 15:03:35] {2466} INFO -  at 2166.3s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 15:03:35] {2282} INFO - iteration 39, current learner lgbm
    [flaml.automl.logger: 08-24 15:04:27] {2466} INFO -  at 2219.0s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 15:04:27] {2282} INFO - iteration 40, current learner lgbm
    [flaml.automl.logger: 08-24 15:06:47] {2466} INFO -  at 2358.2s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 15:06:47] {2282} INFO - iteration 41, current learner lgbm
    [flaml.automl.logger: 08-24 15:07:07] {2466} INFO -  at 2378.7s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 15:07:07] {2282} INFO - iteration 42, current learner lgbm
    [flaml.automl.logger: 08-24 15:13:04] {2466} INFO -  at 2735.5s,	estimator lgbm's best error=0.1915,	best estimator lgbm's best error=0.1915
    [flaml.automl.logger: 08-24 15:13:04] {2282} INFO - iteration 43, current learner lgbm
    [flaml.automl.logger: 08-24 15:15:46] {2466} INFO -  at 2898.2s,	estimator lgbm's best error=0.1891,	best estimator lgbm's best error=0.1891
    [flaml.automl.logger: 08-24 15:15:46] {2282} INFO - iteration 44, current learner lgbm
    [flaml.automl.logger: 08-24 15:17:15] {2466} INFO -  at 2986.4s,	estimator lgbm's best error=0.1891,	best estimator lgbm's best error=0.1891
    [flaml.automl.logger: 08-24 15:17:15] {2282} INFO - iteration 45, current learner lgbm
    [flaml.automl.logger: 08-24 15:18:03] {2466} INFO -  at 3035.1s,	estimator lgbm's best error=0.1891,	best estimator lgbm's best error=0.1891
    [flaml.automl.logger: 08-24 15:18:03] {2282} INFO - iteration 46, current learner lgbm
    [flaml.automl.logger: 08-24 15:27:21] {2466} INFO -  at 3592.2s,	estimator lgbm's best error=0.1879,	best estimator lgbm's best error=0.1879
    [flaml.automl.logger: 08-24 15:27:21] {803} INFO - logging best model lgbm
    [flaml.automl.logger: 08-24 15:30:57] {2724} INFO - retrain lgbm for 216.4s
    [flaml.automl.logger: 08-24 15:30:57] {2727} INFO - retrained model: LGBMClassifier(colsample_bytree=np.float64(0.8011498227432036),
                   is_unbalance=True, learning_rate=np.float64(0.0688619952134549),
                   max_bin=255, min_child_samples=128, n_estimators=4837, n_jobs=-1,
                   num_leaves=111, reg_alpha=np.float64(0.016900277209403555),
                   reg_lambda=np.float64(1.5919392250234898), verbose=-1)
    [flaml.automl.logger: 08-24 15:30:57] {2729} INFO - Best MLflow run name: incongruous-fawn-779_child_46
    [flaml.automl.logger: 08-24 15:30:57] {2730} INFO - Best MLflow run id: a737471c05c44263b08a2cd130cd8de8
    [flaml.automl.logger: 08-24 15:31:00] {2771} WARNING - Exception for record_state task run_5f79b4af98cf49a79f8bdb49ca2a31f6_requirements_updated: No such file or directory: '/Users/leorfinkelberg/Documents/Competitions/e_cup_2025_ml_challenge/mlruns/0/5f79b4af98cf49a79f8bdb49ca2a31f6/artifacts/model/requirements.txt'
    [flaml.automl.logger: 08-24 15:31:00] {2009} INFO - fit succeeded
    [flaml.automl.logger: 08-24 15:31:00] {2010} INFO - Time taken to find the best model: 3592.2205741405487
    """
    )
    return


@app.cell
def _(automl):
    automl.save_best_config("./best_config_f1=0.7333.txt")
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
    [flaml.automl.logger: 08-24 15:39:18] {1752} INFO - task = classification
    [flaml.automl.logger: 08-24 15:39:18] {1763} INFO - Evaluation method: cv
    [flaml.automl.logger: 08-24 15:39:18] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 08-24 15:39:18] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 08-24 15:39:18] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 08-24 15:48:34] {2417} INFO - Estimated sufficient time budget=5558709s. Estimated necessary time budget=5559s.
    [flaml.automl.logger: 08-24 15:48:34] {2466} INFO -  at 561.4s,	estimator lgbm's best error=0.1879,	best estimator lgbm's best error=0.1879
    [flaml.automl.logger: 08-24 15:48:34] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 08-24 15:49:52] {2466} INFO -  at 639.1s,	estimator lgbm's best error=0.1879,	best estimator lgbm's best error=0.1879
    [flaml.automl.logger: 08-24 15:49:52] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 08-24 16:13:31] {2466} INFO -  at 2058.1s,	estimator lgbm's best error=0.1879,	best estimator lgbm's best error=0.1879
    [flaml.automl.logger: 08-24 16:13:31] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 08-24 16:17:23] {2466} INFO -  at 2289.8s,	estimator lgbm's best error=0.1879,	best estimator lgbm's best error=0.1879
    [flaml.automl.logger: 08-24 16:17:23] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 08-24 16:30:22] {2466} INFO -  at 3069.3s,	estimator lgbm's best error=0.1877,	best estimator lgbm's best error=0.1877
    [flaml.automl.logger: 08-24 16:30:22] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 08-24 17:03:01] {2466} INFO -  at 5028.0s,	estimator lgbm's best error=0.1877,	best estimator lgbm's best error=0.1877
    [flaml.automl.logger: 08-24 17:03:01] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 08-24 17:07:05] {2466} INFO -  at 5272.4s,	estimator lgbm's best error=0.1877,	best estimator lgbm's best error=0.1877
    [flaml.automl.logger: 08-24 17:07:05] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 08-24 17:08:17] {2466} INFO -  at 5344.0s,	estimator lgbm's best error=0.1877,	best estimator lgbm's best error=0.1877
    [flaml.automl.logger: 08-24 17:08:17] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 08-24 17:40:11] {2466} INFO -  at 7258.0s,	estimator lgbm's best error=0.1877,	best estimator lgbm's best error=0.1877
    [flaml.automl.logger: 08-24 17:45:17] {2724} INFO - retrain lgbm for 306.4s
    [flaml.automl.logger: 08-24 17:45:17] {2727} INFO - retrained model: LGBMClassifier(colsample_bytree=np.float64(0.8306414433104545),
                   is_unbalance=True, learning_rate=np.float64(0.0967289636753122),
                   max_bin=511, min_child_samples=122, n_estimators=10097,
                   n_jobs=-1, num_leaves=50,
                   reg_alpha=np.float64(0.0010265023922303373),
                   reg_lambda=np.float64(9.6866916621503), verbose=-1)
    [flaml.automl.logger: 08-24 17:45:17] {2009} INFO - fit succeeded
    [flaml.automl.logger: 08-24 17:45:17] {2010} INFO - Time taken to find the best model: 3069.3475227355957
    """
    )
    return


@app.cell
def _(X_test_final, automl_second_part, pd):
    submission_second = pd.DataFrame({
        "id": X_test_final.index, 
        "prediction": automl_second_part.predict(X_test_final),
    })

    submission_second.to_csv("./submission.csv", index=False)
    return


@app.cell
def _(pd):
    pd.read_csv("./submission.csv")["prediction"].value_counts()
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
