import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mlflow
    import category_encoders as ce

    from pyod.models.copod import COPOD

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score, matthews_corrcoef
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.cluster import KMeans

    import re

    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    from pymorphy3 import MorphAnalyzer  # Для русского языка

    from flaml.automl.automl import AutoML
    from flaml import tune
    return AutoML, COPOD, KMeans, PCA, QuantileTransformer, ce, mo, np, pd


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
    train_with_text_embds = pd.read_csv("./ml_ozon_сounterfeit_data/train_with_text_embeddings_after_clean_all-MiniLM-L6-v2.csv", index_col=0)
    test_with_text_embds = pd.read_csv("./ml_ozon_сounterfeit_data//test_with_text_embeddings_after_clean_all-MiniLM-L6-v2.csv", index_col=0)
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
def _(test_with_text_image_embds, train_with_text_image_embds):
    cols_without_target = train_with_text_image_embds.columns.tolist()
    cols_without_target.remove("resolution")
    cols_without_target.remove("ItemID")

    X_train, y_train = train_with_text_image_embds[cols_without_target], train_with_text_image_embds["resolution"]

    X_test = test_with_text_image_embds[cols_without_target]
    return X_test, X_train, y_train


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
    X_train_final_ = pd.concat([
        X_train_with_numeric_cols_wo_nan[original_cols],
        X_train_desc_embds_pca,
        X_train_name_rus_embds_pca,
        X_train_image_embds_pca,
    ], axis=1)

    X_test_final_ = pd.concat([
        X_test_with_numeric_cols_wo_nan[original_cols],
        X_test_desc_embds_pca,
        X_test_name_rus_embds_pca,
        X_test_image_embds_pca,
    ], axis=1)
    return X_test_final_, X_train_final_


@app.cell
def _(QuantileTransformer, SEED):
    quantile_transformer = QuantileTransformer(output_distribution="normal", n_quantiles=350, random_state=SEED)
    return (quantile_transformer,)


@app.cell
def _(X_train_final_, original_cols, quantile_transformer):
    X_train_final_[original_cols] = quantile_transformer.fit_transform(X_train_final_[original_cols])
    return


@app.cell
def _(X_test_final_, original_cols, quantile_transformer):
    X_test_final_[original_cols] = quantile_transformer.transform(X_test_final_[original_cols])
    return


@app.cell
def _(X_train_final_):
    X_train_final_.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Скор аномальности как признак_""")
    return


@app.cell
def _(COPOD):
    detector = COPOD(contamination=0.07, n_jobs=-1)
    return (detector,)


@app.cell
def _(X_train_final_, detector):
    detector.fit(X_train_final_)
    return


@app.cell
def _(X_train_final):
    X_train_final.shape
    return


@app.cell
def _(X_test_final_, X_train_final_, detector):
    X_train_final_.loc[:, "copod_score"] = detector.predict_proba(X_train_final_)[:, 1]
    X_test_final_.loc[:, "copod_score"] = detector.predict_proba(X_test_final_)[:, 1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Индекс кластера как признак_""")
    return


@app.cell
def _(KMeans):
    kmeans3 = KMeans(n_clusters=3)
    kmeans9 = KMeans(n_clusters=9)
    kmeans18 = KMeans(n_clusters=18)
    return kmeans18, kmeans3, kmeans9


@app.cell
def _(X_test_final_, X_train_final_, kmeans18, kmeans3, kmeans9):
    X_train_final_["kmeans3"] = kmeans3.fit_predict(X_train_final_)
    X_test_final_["kmeans3"] = kmeans3.predict(X_test_final_)

    X_train_final_["kmeans9"] = kmeans9.fit_predict(X_train_final_)
    X_test_final_["kmeans9"] = kmeans9.predict(X_test_final_)

    X_train_final_["kmeans18"] = kmeans18.fit_predict(X_train_final_)
    X_test_final_["kmeans18"] = kmeans18.predict(X_test_final_)
    return


@app.cell
def _(ce):
    kmeans_encoder = ce.CatBoostEncoder(cols=["kmeans3", "kmeans9", "kmeans18"])
    return (kmeans_encoder,)


@app.cell
def _(X_train_final_, kmeans_encoder, y_train):
    X_train_final = kmeans_encoder.fit_transform(X_train_final_, y_train)
    return (X_train_final,)


@app.cell
def _(X_test_final_, kmeans_encoder):
    X_test_final = kmeans_encoder.transform(X_test_final_)
    return (X_test_final,)


@app.cell
def _(np, y_train):
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

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

    automl_with_early_stopping = AutoML()
    automl_with_early_stopping.fit(
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
    return


@app.cell
def _(AutoML, SEED, TASK, X_train_final, custom_hp, y_train):
    ### ЗАПУСК БЕЗ РАННЕЙ ОСТАНОВКИ

    automl_wo_early_stopping = AutoML()
    automl_wo_early_stopping.fit(
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
        split_ratio=0.2,
        # n_splits=3,
        metric="f1",
        split_type="stratified",
        custom_hp=custom_hp,
        seed=SEED,
    )
    return (automl_wo_early_stopping,)


@app.cell
def _(mo):
    mo.md(
        r"""
    [flaml.automl.logger: 08-29 05:35:41] {1752} INFO - task = classification
    [flaml.automl.logger: 08-29 05:35:41] {1763} INFO - Evaluation method: cv
    [flaml.automl.logger: 08-29 05:35:41] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 08-29 05:35:41] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 08-29 05:35:41] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 08-29 05:35:44] {2417} INFO - Estimated sufficient time budget=26418s. Estimated necessary time budget=26s.
    [flaml.automl.logger: 08-29 05:35:44] {2466} INFO -  at 7.7s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 08-29 05:35:44] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 08-29 05:35:47] {2466} INFO -  at 10.4s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 08-29 05:35:47] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 08-29 05:35:50] {2466} INFO -  at 13.2s,	estimator lgbm's best error=0.3936,	best estimator lgbm's best error=0.3936
    [flaml.automl.logger: 08-29 05:35:50] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 08-29 05:35:52] {2466} INFO -  at 16.0s,	estimator lgbm's best error=0.3936,	best estimator lgbm's best error=0.3936
    [flaml.automl.logger: 08-29 05:35:52] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 08-29 05:35:55] {2466} INFO -  at 19.1s,	estimator lgbm's best error=0.3936,	best estimator lgbm's best error=0.3936
    [flaml.automl.logger: 08-29 05:35:55] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 08-29 05:35:59] {2466} INFO -  at 22.5s,	estimator lgbm's best error=0.3936,	best estimator lgbm's best error=0.3936
    [flaml.automl.logger: 08-29 05:35:59] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 08-29 05:36:02] {2466} INFO -  at 25.3s,	estimator lgbm's best error=0.3936,	best estimator lgbm's best error=0.3936
    [flaml.automl.logger: 08-29 05:36:02] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 08-29 05:36:04] {2466} INFO -  at 27.9s,	estimator lgbm's best error=0.3936,	best estimator lgbm's best error=0.3936
    [flaml.automl.logger: 08-29 05:36:04] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 08-29 05:36:10] {2466} INFO -  at 33.8s,	estimator lgbm's best error=0.3222,	best estimator lgbm's best error=0.3222
    [flaml.automl.logger: 08-29 05:36:10] {2282} INFO - iteration 9, current learner lgbm
    [flaml.automl.logger: 08-29 05:36:17] {2466} INFO -  at 40.2s,	estimator lgbm's best error=0.3222,	best estimator lgbm's best error=0.3222
    [flaml.automl.logger: 08-29 05:36:17] {2282} INFO - iteration 10, current learner lgbm
    [flaml.automl.logger: 08-29 05:36:23] {2466} INFO -  at 46.3s,	estimator lgbm's best error=0.3222,	best estimator lgbm's best error=0.3222
    [flaml.automl.logger: 08-29 05:36:23] {2282} INFO - iteration 11, current learner lgbm
    [flaml.automl.logger: 08-29 05:36:33] {2466} INFO -  at 56.6s,	estimator lgbm's best error=0.2650,	best estimator lgbm's best error=0.2650
    [flaml.automl.logger: 08-29 05:36:33] {2282} INFO - iteration 12, current learner lgbm
    [flaml.automl.logger: 08-29 05:36:39] {2466} INFO -  at 62.7s,	estimator lgbm's best error=0.2650,	best estimator lgbm's best error=0.2650
    [flaml.automl.logger: 08-29 05:36:39] {2282} INFO - iteration 13, current learner lgbm
    [flaml.automl.logger: 08-29 05:36:47] {2466} INFO -  at 71.0s,	estimator lgbm's best error=0.2650,	best estimator lgbm's best error=0.2650
    [flaml.automl.logger: 08-29 05:36:47] {2282} INFO - iteration 14, current learner lgbm
    [flaml.automl.logger: 08-29 05:37:00] {2466} INFO -  at 83.4s,	estimator lgbm's best error=0.2254,	best estimator lgbm's best error=0.2254
    [flaml.automl.logger: 08-29 05:37:00] {2282} INFO - iteration 15, current learner lgbm
    [flaml.automl.logger: 08-29 05:37:07] {2466} INFO -  at 90.1s,	estimator lgbm's best error=0.2254,	best estimator lgbm's best error=0.2254
    [flaml.automl.logger: 08-29 05:37:07] {2282} INFO - iteration 16, current learner lgbm
    [flaml.automl.logger: 08-29 05:37:31] {2466} INFO -  at 114.2s,	estimator lgbm's best error=0.2254,	best estimator lgbm's best error=0.2254
    [flaml.automl.logger: 08-29 05:37:31] {2282} INFO - iteration 17, current learner lgbm
    [flaml.automl.logger: 08-29 05:37:35] {2466} INFO -  at 118.7s,	estimator lgbm's best error=0.2254,	best estimator lgbm's best error=0.2254
    [flaml.automl.logger: 08-29 05:37:35] {2282} INFO - iteration 18, current learner lgbm
    [flaml.automl.logger: 08-29 05:38:21] {2466} INFO -  at 165.0s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:38:21] {2282} INFO - iteration 19, current learner lgbm
    [flaml.automl.logger: 08-29 05:38:36] {2466} INFO -  at 179.5s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:38:36] {2282} INFO - iteration 20, current learner lgbm
    [flaml.automl.logger: 08-29 05:40:50] {2466} INFO -  at 313.8s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:40:50] {2282} INFO - iteration 21, current learner lgbm
    [flaml.automl.logger: 08-29 05:41:37] {2466} INFO -  at 360.5s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:41:37] {2282} INFO - iteration 22, current learner lgbm
    [flaml.automl.logger: 08-29 05:42:10] {2466} INFO -  at 393.4s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:42:10] {2282} INFO - iteration 23, current learner lgbm
    [flaml.automl.logger: 08-29 05:43:03] {2466} INFO -  at 446.1s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:43:03] {2282} INFO - iteration 24, current learner lgbm
    [flaml.automl.logger: 08-29 05:43:38] {2466} INFO -  at 481.3s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:43:38] {2282} INFO - iteration 25, current learner lgbm
    [flaml.automl.logger: 08-29 05:45:11] {2466} INFO -  at 574.4s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:45:11] {2282} INFO - iteration 26, current learner lgbm
    [flaml.automl.logger: 08-29 05:45:39] {2466} INFO -  at 603.0s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:45:39] {2282} INFO - iteration 27, current learner lgbm
    [flaml.automl.logger: 08-29 05:45:48] {2466} INFO -  at 612.1s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:45:48] {2282} INFO - iteration 28, current learner lgbm
    [flaml.automl.logger: 08-29 05:50:55] {2466} INFO -  at 919.1s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:50:55] {2282} INFO - iteration 29, current learner lgbm
    [flaml.automl.logger: 08-29 05:51:19] {2466} INFO -  at 942.5s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:51:19] {2282} INFO - iteration 30, current learner lgbm
    [flaml.automl.logger: 08-29 05:52:50] {2466} INFO -  at 1033.3s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:52:50] {2282} INFO - iteration 31, current learner lgbm
    [flaml.automl.logger: 08-29 05:53:43] {2466} INFO -  at 1086.3s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:53:43] {2282} INFO - iteration 32, current learner lgbm
    [flaml.automl.logger: 08-29 05:54:31] {2466} INFO -  at 1134.4s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:54:31] {2282} INFO - iteration 33, current learner lgbm
    [flaml.automl.logger: 08-29 05:57:06] {2466} INFO -  at 1289.7s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:57:06] {2282} INFO - iteration 34, current learner lgbm
    [flaml.automl.logger: 08-29 05:57:20] {2466} INFO -  at 1303.6s,	estimator lgbm's best error=0.1900,	best estimator lgbm's best error=0.1900
    [flaml.automl.logger: 08-29 05:57:20] {2282} INFO - iteration 35, current learner lgbm
    [flaml.automl.logger: 08-29 06:00:24] {2466} INFO -  at 1487.4s,	estimator lgbm's best error=0.1861,	best estimator lgbm's best error=0.1861
    [flaml.automl.logger: 08-29 06:00:24] {2282} INFO - iteration 36, current learner lgbm
    [flaml.automl.logger: 08-29 06:01:10] {2466} INFO -  at 1533.7s,	estimator lgbm's best error=0.1861,	best estimator lgbm's best error=0.1861
    [flaml.automl.logger: 08-29 06:01:10] {2282} INFO - iteration 37, current learner lgbm
    [flaml.automl.logger: 08-29 06:03:00] {2466} INFO -  at 1643.6s,	estimator lgbm's best error=0.1861,	best estimator lgbm's best error=0.1861
    [flaml.automl.logger: 08-29 06:03:00] {2282} INFO - iteration 38, current learner lgbm
    [flaml.automl.logger: 08-29 06:22:47] {2466} INFO -  at 2830.3s,	estimator lgbm's best error=0.1827,	best estimator lgbm's best error=0.1827
    [flaml.automl.logger: 08-29 06:24:19] {2724} INFO - retrain lgbm for 92.1s
    [flaml.automl.logger: 08-29 06:24:19] {2727} INFO - retrained model: LGBMClassifier(colsample_bytree=np.float64(0.6953215089893051),
                   is_unbalance=True, learning_rate=np.float64(0.10644748589515138),
                   max_bin=1023, min_child_samples=54, n_estimators=1252, n_jobs=-1,
                   num_leaves=129, reg_alpha=0.0009765625,
                   reg_lambda=np.float64(1.2245998946760492), verbose=-1)
    [flaml.automl.logger: 08-29 06:24:19] {2009} INFO - fit succeeded
    [flaml.automl.logger: 08-29 06:24:19] {2010} INFO - Time taken to find the best model: 2830.2626099586487
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    retrained model: LGBMClassifier(colsample_bytree=np.float64(0.8771225442915792),
                   is_unbalance=True, learning_rate=np.float64(0.15498210767922252),
                   max_bin=1023, min_child_samples=5, n_estimators=3894, n_jobs=-1,
                   num_leaves=76, reg_alpha=0.0009765625,
                   reg_lambda=np.float64(0.011614476432374917), verbose=-1)
    """
    )
    return


@app.cell
def _(X_test_final, automl_wo_early_stopping, pd):
    submission = pd.DataFrame({
        "id": X_test_final.index, 
        "prediction": automl_wo_early_stopping.predict(X_test_final),
    })

    submission.to_csv("./submission.csv", index=False)
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
    X_train_final,
    automl_wo_early_stopping,
    custom_hp,
    y_train,
):
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
        starting_points=automl_wo_early_stopping.best_config_per_estimator,
    )
    return (automl_second_part,)


@app.cell
def _(mo):
    mo.md(
        r"""
    [flaml.automl.logger: 08-29 06:33:12] {1752} INFO - task = classification
    [flaml.automl.logger: 08-29 06:33:12] {1763} INFO - Evaluation method: cv
    [flaml.automl.logger: 08-29 06:33:12] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 08-29 06:33:12] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 08-29 06:33:12] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 08-29 06:37:07] {2417} INFO - Estimated sufficient time budget=2349853s. Estimated necessary time budget=2350s.
    [flaml.automl.logger: 08-29 06:37:07] {2466} INFO -  at 240.1s,	estimator lgbm's best error=0.1809,	best estimator lgbm's best error=0.1809
    [flaml.automl.logger: 08-29 06:37:07] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 08-29 06:37:35] {2466} INFO -  at 268.4s,	estimator lgbm's best error=0.1809,	best estimator lgbm's best error=0.1809
    [flaml.automl.logger: 08-29 06:37:35] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 08-29 06:52:08] {2466} INFO -  at 1141.4s,	estimator lgbm's best error=0.1809,	best estimator lgbm's best error=0.1809
    [flaml.automl.logger: 08-29 06:52:08] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 08-29 06:55:14] {2466} INFO -  at 1327.5s,	estimator lgbm's best error=0.1809,	best estimator lgbm's best error=0.1809
    [flaml.automl.logger: 08-29 06:55:14] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 08-29 06:59:36] {2466} INFO -  at 1589.7s,	estimator lgbm's best error=0.1804,	best estimator lgbm's best error=0.1804
    [flaml.automl.logger: 08-29 06:59:36] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 08-29 07:42:54] {2466} INFO -  at 4187.2s,	estimator lgbm's best error=0.1804,	best estimator lgbm's best error=0.1804
    [flaml.automl.logger: 08-29 07:42:54] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 08-29 07:44:05] {2466} INFO -  at 4258.7s,	estimator lgbm's best error=0.1804,	best estimator lgbm's best error=0.1804
    [flaml.automl.logger: 08-29 07:44:05] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 08-29 07:44:28] {2466} INFO -  at 4281.5s,	estimator lgbm's best error=0.1804,	best estimator lgbm's best error=0.1804
    [flaml.automl.logger: 08-29 07:44:28] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 08-29 08:20:41] {2466} INFO -  at 6454.2s,	estimator lgbm's best error=0.1804,	best estimator lgbm's best error=0.1804
    [flaml.automl.logger: 08-29 08:20:41] {2282} INFO - iteration 9, current learner lgbm
    [flaml.automl.logger: 08-29 08:25:46] {2466} INFO -  at 6759.5s,	estimator lgbm's best error=0.1804,	best estimator lgbm's best error=0.1804
    [flaml.automl.logger: 08-29 08:25:46] {2282} INFO - iteration 10, current learner lgbm
    [flaml.automl.logger: 08-29 08:29:21] {2466} INFO -  at 6974.4s,	estimator lgbm's best error=0.1804,	best estimator lgbm's best error=0.1804
    [flaml.automl.logger: 08-29 08:31:04] {2724} INFO - retrain lgbm for 102.9s
    [flaml.automl.logger: 08-29 08:31:04] {2727} INFO - retrained model: LGBMClassifier(colsample_bytree=np.float64(0.7248131295565561),
                   is_unbalance=True, learning_rate=np.float64(0.1495244940923316),
                   max_bin=1023, min_child_samples=51, n_estimators=2614, n_jobs=-1,
                   num_leaves=58, reg_alpha=0.0009765625,
                   reg_lambda=np.float64(7.4514914908598895), verbose=-1)
    [flaml.automl.logger: 08-29 08:31:04] {2009} INFO - fit succeeded
    [flaml.automl.logger: 08-29 08:31:04] {2010} INFO - Time taken to find the best model: 1589.7304277420044
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
