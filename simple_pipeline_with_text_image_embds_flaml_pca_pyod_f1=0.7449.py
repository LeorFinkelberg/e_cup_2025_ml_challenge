import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Добавлены индексы кластера как фичи
    - Добавлены скоры детектора аномалий как фичи
    - Предобработаны текстовые фичи (токеницазия, лемматизаци etc.)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Импорты_""")
    return


@app.cell
def _():
    import marimo as mo
    import typing as t
    import pathlib
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mlflow
    import category_encoders as ce
    import lightgbm as lgb

    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.suod import SUOD
    from pyod.models.hbos import HBOS
    from pyod.models.iforest import IForest

    from datetime import datetime

    from tqdm import tqdm

    from joblib import parallel_backend

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, silhouette_score
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.cluster import KMeans, BisectingKMeans
    from sklearn.ensemble import VotingClassifier

    import plotly.express as px

    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    from pymorphy3 import MorphAnalyzer  # Для русского языка

    from flaml.automl.automl import AutoML
    from flaml import tune
    return (
        AutoML,
        COPOD,
        ColumnTransformer,
        KMeans,
        PCA,
        Pipeline,
        QuantileTransformer,
        VotingClassifier,
        ce,
        datetime,
        lgb,
        mo,
        np,
        pathlib,
        pd,
        plt,
        px,
        t,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Вспомогательные функции_""")
    return


@app.cell
def _(np, pathlib, pd):
    def save_submission(y_pred: np.typing.ArrayLike, X_test: pd.DataFrame) -> None:
        PATH_TO_SUBMISSION_FILE = "./submission.csv"

        pd.DataFrame({
            "id": X_test.index, 
            "prediction": y_pred,
        }).to_csv(PATH_TO_SUBMISSION_FILE, index=False)

        if not pathlib.Path(PATH_TO_SUBMISSION_FILE).exists():
            raise FileNotFoundError(f"Error! File {PATH_TO_SUBMISSION_FILE} not found ...")

        print(f"File {PATH_TO_SUBMISSION_FILE} was recorded successfully!")
    return (save_submission,)


@app.cell
def _(pathlib, pd, plt, t):
    def get_ratio_to_f1(
        test_negs: t.Optional[float] = None,
        figsize: tuple[int, int] = (10, 5),
    ):
        fig, ax = plt.subplots(figsize=figsize)

        skip_files = 0
        ratio_to_f1 = []
        for submission in pathlib.Path("./submissions").glob("*.csv"):
            try:
                f1 = float(pathlib.Path(submission).with_suffix("").name.split("=")[-1])
                _ratio = pd.read_csv(submission, index_col=False)["prediction"].value_counts()
                # ratio_to_f1.append((f1, _ratio.iloc[1] / _ratio.iloc[0]))
                ratio_to_f1.append((f1, _ratio.iloc[1]))
            except ValueError:
                skip_files += 1
                continue

        print(f"Find {len(ratio_to_f1)} files")

        pd.DataFrame.from_records(
            ratio_to_f1,
            columns=["f1", "ratio"]
        ) \
        .sort_values("ratio") \
        .plot.scatter(ax=ax, x="ratio", y="f1", marker="o")

        best_point = max(ratio_to_f1)
        ax.scatter(best_point[1], best_point[0], s=150, color="red")
        ax.text(best_point[1] + 0.001, best_point[0], best_point[0], size=12, color="red")

        if test_negs:
            ax.axvline(test_negs, ls="--", color="grey")

        ax.set_xlabel("negs")
        ax.set_ylabel("f1")

        fig.tight_layout()

        return fig
    return (get_ratio_to_f1,)


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
def _(np, pd, px):
    def plot_heatmap(
        X_train: pd.DataFrame,
        title: str,
        color_continuous_scale: str = "Viridis",
        threshold: float = 0.5,
    ) -> None:
        fig = px.imshow(
            X_train.corr()[np.abs(X_train.corr()) > threshold],
            color_continuous_scale=color_continuous_scale,
            title=title,
            aspect="auto",
        )

        return fig
    return (plot_heatmap,)


@app.cell
def _(SEED, VotingClassifier, lgb, np, pd, t):
    def make_predict_with_top_k_models(
        X_train: pd.DataFrame,
        y_train: np.typing.ArrayLike,
        X_test: pd.DataFrame,
        path_to_log_file: str, 
        top_k: int = 5,
        voting: str = "soft",
        predict_proba: bool = False,
        threshold: float = 0.5,
        weights: t.Optional[np.typing.ArrayLike] = None,
        seed: int = SEED,
    ) -> np.typing.ArrayLike:
        from pprint import pprint 

        NEGS_POSS_MSG = "[{model_name}] 0/1: {negs_poss[1]} / {negs_poss[0]}"

        if top_k < 1:
            raise ValueError(f"Error! top_k must be grater then 0")

        trials_log = pd.read_json(path_to_log_file, lines=True) \
            .sort_values("validation_loss") \
            .iloc[:top_k]

        models = []
        print("Start ...")
        for record in trials_log.itertuples():
            pprint(f"MODEL CONFIG: {record.config}")
            model = lgb.LGBMClassifier(**record.config, force_col_wise=True, random_state=seed)
            models.append((f"clf-{int(record.record_id)}", model))

        print(f"Get {len(models)} models ...")

        if len(models) > 1:
            print("Start voter fitting ...")

            model = VotingClassifier(
                estimators=models,
                voting="soft",
                weights=weights,
            ).fit(X_train, y_train)

            negs_poss = pd.Series(model.predict(X_test)).value_counts().values
            print(NEGS_POSS_MSG.format(model_name=model.__class__.__name__, negs_poss=negs_poss))

            for model_name, model_ in models: 
                negs_poss = pd.Series(model.named_estimators_[model_name].predict(X_test)).value_counts().values
                print(NEGS_POSS_MSG.format(model_name=model_name, negs_poss=negs_poss))
        else:
            print("Start model fitting with val-score-best-config ...")
            model = models[-1].fit(X_train, y_train)

            negs_poss = pd.Series(model.predict(X_test)).value_counts().values
            print(NEGS_POSS_MSG.format(model_name=model.__class__.__name__, negs_poss=negs_poss))

        print(f"PREDICT PROBA MODE: {predict_proba}")
        y_test_pred = (
            model.predict_proba(X_test)[:, 1]
            if predict_proba else
            model.predict(X_test)
        )

        return (
            (y_test_pred > threshold).astype(int)
            if predict_proba else
            y_test_pred
        )
    return (make_predict_with_top_k_models,)


@app.function
def add_cols_to_original_and_custom_col_names(
    original_and_custom_col_names: list[str],
    new_col_names: list[str]
) -> None:
    new_cols_count = 0
    for new_col in set(new_col_names):
        if new_col not in original_and_custom_col_names:
            original_and_custom_col_names.append(new_col)
            new_cols_count += 1

    print(f"Was added {new_cols_count} new cols. Total cols: {len(original_and_custom_col_names)}")


@app.function
def remove_cols_from_original_and_custom_col_names(
    original_and_custom_col_names: list[str],
    col_names_for_removed: list[str],
) -> None:
    removed_col_names = 0
    for col_name in set(col_names_for_removed):
        if col_name in original_and_custom_col_names:
            original_and_custom_col_names.remove(col_name)
            removed_col_names += 1
        else:
            print(f"Warning! Column {col_name!r} not found")

    print(f"Was removed {removed_col_names} cols. Total cols: {len(original_and_custom_col_names)}")


@app.cell
def _():
    TASK = "classification"
    # SEED = 34534588
    SEED = 42
    IMAGE_N_COMPONENTS = 100
    IMAGE_WINDOW = 10
    TEXT_WINDOW = 5
    TEXT_N_COMPONENTS = 50 
    return IMAGE_N_COMPONENTS, SEED, TASK, TEXT_N_COMPONENTS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Зависимость метрики F1 от отношения числа позитивов к числу негативов_""")
    return


@app.cell
def _(get_ratio_to_f1):
    get_ratio_to_f1(test_negs=1213)
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
    return


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
def _():
    remove_col_names = [
        "description",
        "name_rus",
        "rating_1_count",
        "rating_2_count",
        "rating_3_count",
        "rating_4_count",
        "rating_5_count",
        "item_count_returns7", 
        "item_count_returns30", 
        "item_count_returns90", 
        "item_count_fake_returns7",
        "item_count_fake_returns30",
        "item_count_fake_returns90",
        "GmvTotal7",
        "GmvTotal30",
        "GmvTotal90",
        "item_count_sales7",
        "item_count_sales30",
        "item_count_sales90",
    ]
    return (remove_col_names,)


@app.cell
def _(X_train, remove_col_names):
    original_and_custom_col_names = list(set(X_train.columns).difference(set(remove_col_names)))
    len(original_and_custom_col_names)
    return (original_and_custom_col_names,)


@app.cell
def _(X_test, X_train, original_and_custom_col_names):
    rating_5_count_ratio_col_name = "rating_5_count_ratio"

    X_train.loc[:, rating_5_count_ratio_col_name] = X_train["rating_5_count"] / (
        X_train["rating_1_count"]
        + X_train["rating_2_count"]
        + X_train["rating_3_count"]
        + X_train["rating_4_count"]
        + X_train["rating_5_count"]
    )

    X_test.loc[:, rating_5_count_ratio_col_name] = X_test["rating_5_count"] / (
        X_test["rating_1_count"]
        + X_test["rating_2_count"]
        + X_test["rating_3_count"]
        + X_test["rating_4_count"]
        + X_test["rating_5_count"]
    )
    add_cols_to_original_and_custom_col_names(original_and_custom_col_names, new_col_names=[rating_5_count_ratio_col_name])
    return


@app.cell
def _(X_test, X_train, original_and_custom_col_names):
    item_count_fake_returns_spike7_col_name = "item_count_fake_returns_spike7"

    X_train.loc[:, item_count_fake_returns_spike7_col_name] = (
        X_train["item_count_fake_returns30"] > 3 * X_train["item_count_fake_returns7"]
    ).astype(int)

    X_test.loc[:, item_count_fake_returns_spike7_col_name] = (
        X_test["item_count_fake_returns30"] > 3 * X_test["item_count_fake_returns7"]
    ).astype(int)

    add_cols_to_original_and_custom_col_names(original_and_custom_col_names, new_col_names=[item_count_fake_returns_spike7_col_name])
    return


@app.cell
def _():
    """
    item_count_sales_decay7_col_name = "item_count_sales_decay7"

    X_train.loc[:, item_count_sales_decay7_col_name] = (
        X_train["item_count_sales30"] > 3 * X_train["item_count_sales7"]
    ).astype(int)

    X_test.loc[:, item_count_sales_decay7_col_name] = (
        X_test["item_count_sales30"] > 3 * X_test["item_count_sales7"]
    ).astype(int)

    add_cols_to_original_and_custom_col_names(original_and_custom_col_names, new_col_names=[item_count_sales_decay7_col_name])
    """
    return


@app.cell
def _():
    categorical_features = [
        "CommercialTypeName4",
        "brand_name",
        "SellerID",
    ]
    return (categorical_features,)


@app.cell
def _():
    binary_features = [
        "item_count_fake_returns_spike7",
        # "item_count_sales_decay7",
    ]
    return (binary_features,)


@app.cell
def _(ColumnTransformer, binary_features, categorical_features, ce):
    encoder = ColumnTransformer(
        transformers=[
            (
                "category_encoder", 
                ce.CountEncoder(
                    cols=categorical_features,
                    normalize=True,
                    handle_missing=-1,
                    handle_unknown=-1
                ),
                categorical_features,
            ),
            (
                "binary_encoder",
                ce.CatBoostEncoder(cols=binary_features),
                binary_features,
            )
        ]
    )
    return (encoder,)


@app.cell
def _(Pipeline, encoder):
    encoder_pipeline = Pipeline(steps=[("encoder", encoder)])
    return (encoder_pipeline,)


@app.cell
def _(
    X_test,
    X_train,
    binary_features,
    categorical_features,
    encoder_pipeline,
    y_train,
):
    X_train[categorical_features + binary_features] = encoder_pipeline.fit_transform(X_train, y_train)
    X_test[categorical_features + binary_features] = encoder_pipeline.transform(X_test)
    return


@app.cell
def _(X_train):
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    return (numeric_cols,)


@app.cell
def _(X_test, X_train, numeric_cols):
    X_train_with_numeric_cols = X_train[numeric_cols]
    X_test_with_numeric_cols = X_test[numeric_cols]
    return X_test_with_numeric_cols, X_train_with_numeric_cols


@app.cell
def _(X_test_with_numeric_cols, X_train_with_numeric_cols):
    X_train_with_numeric_cols_wo_nan = X_train_with_numeric_cols.fillna(X_train_with_numeric_cols.mean())
    X_test_with_numeric_cols_wo_nan = X_test_with_numeric_cols.fillna(X_test_with_numeric_cols.mean())
    return X_test_with_numeric_cols_wo_nan, X_train_with_numeric_cols_wo_nan


@app.cell
def _():
    # X_train_with_numeric_cols_wo_nan[image_embds_col_names].iloc[0].plot()
    return


@app.cell
def _():
    # X_train_with_numeric_cols_wo_nan[desc_embds_col_names].iloc[0].plot()
    return


@app.cell
def _():
    # X_train_with_numeric_cols_wo_nan[name_rus_embds_col_names].iloc[0].plot()
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
    desc_embds_col_names,
    image_embds_col_names,
    name_rus_embds_col_names,
    original_and_custom_col_names,
):
    remove_cols_from_original_and_custom_col_names(
        original_and_custom_col_names,
        col_names_for_removed=(
            image_embds_col_names
            + desc_embds_col_names
            + name_rus_embds_col_names
        )
    )
    return


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
    original_and_custom_col_names,
    pd,
):
    X_train_final_ = pd.concat([
        X_train_with_numeric_cols_wo_nan[original_and_custom_col_names],
        X_train_image_embds_pca,
        X_train_desc_embds_pca,
        X_train_name_rus_embds_pca,
    ], axis=1)

    X_test_final_ = pd.concat([
        X_test_with_numeric_cols_wo_nan[original_and_custom_col_names],
        X_test_image_embds_pca,
        X_test_desc_embds_pca,
        X_test_name_rus_embds_pca,
    ], axis=1)
    return X_test_final_, X_train_final_


@app.cell
def _(X_train_final_):
    X_train_final_
    return


@app.cell
def _(X_test_final_):
    X_test_final_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Скор аномальности как признак_""")
    return


@app.cell
def _(np, y_train):
    CONTAMINATION = np.sum(y_train == 1) / np.sum(y_train == 0)
    CONTAMINATION
    return (CONTAMINATION,)


@app.cell
def _(CONTAMINATION, COPOD):
    detector = COPOD(contamination=CONTAMINATION, n_jobs=-1)
    return (detector,)


@app.cell
def _(X_train_final_, detector):
    detector.fit(X_train_final_)
    return


@app.cell
def _(X_test_final_, X_train_final_, detector, original_and_custom_col_names):
    X_train_final_.loc[:, "copod_score"] = detector.predict_proba(X_train_final_)[:, 1]
    X_test_final_.loc[:, "copod_score"] = detector.predict_proba(X_test_final_)[:, 1]
    add_cols_to_original_and_custom_col_names(original_and_custom_col_names, new_col_names=["copod_score"])
    return


@app.cell
def _():
    # detector.explain_outlier(ind=10, feature_names=X_train_final_[list(set(final_col_names).difference(set(["copod_score"])))])
    return


@app.cell
def _():
    """
    suod_avg_col_names = list(set(final_col_names).difference(set(["copod_score"])))

    with parallel_backend("loky"):
        detector_ensemble = SUOD(
            base_estimators=[
                IForest(
                    n_estimators=100,
                    contamination=CONTAMINATION,
                    random_state=SEED,
                ),
                IForest(
                    n_estimators=350,
                    contamination=CONTAMINATION,
                    random_state=SEED,
                ),
                ECOD(contamination=CONTAMINATION),
            ],
            combination="average",
            n_jobs=1,
            verbose=True,
        ).fit(X_train_final_[suod_avg_col_names])
    """
    return


@app.cell
def _():
    """
    X_train_final_proba = detector_ensemble.predict_proba(X_train_final_[suod_avg_col_names])[:, 1]
    X_test_final_proba = detector_ensemble.predict_proba(X_test_final_[suod_avg_col_names])[:, 1]
    """
    return


@app.cell
def _():
    # X_train_final_.loc[:, "suod_avg_thshld_0.25"] = (X_train_final_proba > 0.25).astype(int)
    # X_test_final_.loc[:, "suod_avg_thshld_0.25"] = (X_test_final_proba > 0.25).astype(int)
    # add_cols_to_final_col_names(final_col_names, new_col_names=["suod_avg_thshld_0.25"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Индекс кластера как признак_""")
    return


@app.cell
def _():
    """
    X_train_clusters = X_train_final_[final_col_names]

    for n_clusters in range(50, 400, 50):
        print(f"{n_clusters=}")
        kmeans = BisectingKMeans(
            n_clusters=n_clusters,
            max_iter=100,
            verbose=0,
            random_state=SEED
        )
        print(silhouette_score(X_train_clusters, kmeans.fit_predict(X_train_clusters)))
    """
    return


@app.cell
def _():
    return


@app.cell
def _(QuantileTransformer, SEED):
    quantile_transformer = QuantileTransformer(output_distribution="uniform", n_quantiles=350, random_state=SEED)
    return (quantile_transformer,)


@app.cell
def _(
    X_test_final_,
    X_train_final_,
    original_and_custom_col_names,
    quantile_transformer,
):
    X_train_final_[original_and_custom_col_names] = quantile_transformer.fit_transform(X_train_final_[original_and_custom_col_names])
    X_test_final_[original_and_custom_col_names] = quantile_transformer.transform(X_test_final_[original_and_custom_col_names])
    return


@app.cell
def _(KMeans, SEED):
    kmeans50 = KMeans(n_clusters=50, random_state=SEED)
    return (kmeans50,)


@app.cell
def _(X_test_final_, X_train_final_, kmeans50, original_and_custom_col_names):
    X_train_final_["kmeans50"] = kmeans50.fit_predict(X_train_final_)
    X_test_final_["kmeans50"] = kmeans50.predict(X_test_final_)

    add_cols_to_original_and_custom_col_names(
        original_and_custom_col_names,
        new_col_names=[
            "kmeans50",
        ]
    )
    return


@app.cell
def _(ce):
    kmeans_encoder = ce.CatBoostEncoder(cols=["kmeans50"])
    return (kmeans_encoder,)


@app.cell
def _(X_test_final_, X_train_final_, kmeans_encoder, y_train):
    X_train_final = kmeans_encoder.fit_transform(X_train_final_, y_train)
    X_test_final = kmeans_encoder.transform(X_test_final_)
    return X_test_final, X_train_final


@app.cell
def _(X_train_final):
    X_train_final.shape
    return


@app.cell
def _(
    X_train_final,
    desc_embds_pca_col_names,
    image_embds_pca_col_names,
    name_rus_embds_pca_col_names,
    original_and_custom_col_names,
    plot_heatmap,
):
    plot_heatmap(
        X_train_final.loc[:, list(set(original_and_custom_col_names).difference(set(
            image_embds_pca_col_names
            + desc_embds_pca_col_names
            + name_rus_embds_pca_col_names
        )))],
        title="corr_matrix",
        threshold=0.55,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _FLAML_""")
    return


@app.cell
def _(np, y_train):
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

    custom_hp = {
        "lgbm": {
            # "is_unbalance": {
                # "domain": True,
            # },
            "scale_pos_weight": {
                "domain": scale_pos_weight, 
            },
        },
        "catboost": {
            "auto_class_weights": {
                "domain": "Balanced",
            },
        },
        "xgboost": {
            "scale_pos_weight": {
                "domain": scale_pos_weight, 
            },
        }
    }
    return (custom_hp,)


@app.cell
def _():
    """
    ### ЗАПУСК С РАННЕЙ ОСТАНОВКОЙ

    datetime_postfix = datetime.now().strftime("%d_%m_%y_T%H%M")

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
        log_file_name=f"./log_flaml_tuning_{datetime_postfix}.log",
        log_type="all",
        seed=SEED,
    )
    """
    return


@app.cell
def _(AutoML, SEED, TASK, X_train_final, custom_hp, datetime, y_train):
    ### ЗАПУСК БЕЗ РАННЕЙ ОСТАНОВКИ

    datetime_postfix = datetime.now().strftime("%d_%m_%yT%H%M")

    automl_wo_early_stopping = AutoML()
    automl_wo_early_stopping.fit(
        X_train_final,
        y_train,
        task=TASK,
        time_budget=3600,
        estimator_list=(
            "lgbm",
        ),
        eval_method="holdout",
        split_ratio=0.2,
        metric="f1",
        split_type="stratified",
        custom_hp=custom_hp,
        log_file_name=f"./log_flaml_tuning_{datetime_postfix}.log",
        log_type="all",
        seed=SEED,
    )
    return (automl_wo_early_stopping,)


@app.cell
def _(mo):
    mo.md(
        r"""
    [flaml.automl.logger: 08-31 22:00:04] {1752} INFO - task = classification
    [flaml.automl.logger: 08-31 22:00:04] {1763} INFO - Evaluation method: holdout
    [flaml.automl.logger: 08-31 22:00:05] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 08-31 22:00:05] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 08-31 22:00:05] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:05] {2417} INFO - Estimated sufficient time budget=11334s. Estimated necessary time budget=11s.
    [flaml.automl.logger: 08-31 22:00:05] {2466} INFO -  at 5.5s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 08-31 22:00:05] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:05] {2466} INFO -  at 5.5s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 08-31 22:00:05] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:05] {2466} INFO -  at 5.6s,	estimator lgbm's best error=0.4908,	best estimator lgbm's best error=0.4908
    [flaml.automl.logger: 08-31 22:00:05] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:05] {2466} INFO -  at 5.7s,	estimator lgbm's best error=0.4908,	best estimator lgbm's best error=0.4908
    [flaml.automl.logger: 08-31 22:00:05] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:05] {2466} INFO -  at 5.8s,	estimator lgbm's best error=0.4908,	best estimator lgbm's best error=0.4908
    [flaml.automl.logger: 08-31 22:00:05] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:05] {2466} INFO -  at 5.9s,	estimator lgbm's best error=0.4908,	best estimator lgbm's best error=0.4908
    [flaml.automl.logger: 08-31 22:00:05] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:05] {2466} INFO -  at 6.0s,	estimator lgbm's best error=0.4908,	best estimator lgbm's best error=0.4908
    [flaml.automl.logger: 08-31 22:00:05] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:06] {2466} INFO -  at 6.8s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 08-31 22:00:06] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:07] {2466} INFO -  at 7.6s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 08-31 22:00:07] {2282} INFO - iteration 9, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:08] {2466} INFO -  at 8.5s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 08-31 22:00:08] {2282} INFO - iteration 10, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:09] {2466} INFO -  at 9.3s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 08-31 22:00:09] {2282} INFO - iteration 11, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:09] {2466} INFO -  at 10.1s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 08-31 22:00:09] {2282} INFO - iteration 12, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:10] {2466} INFO -  at 11.0s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 08-31 22:00:10] {2282} INFO - iteration 13, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:11] {2466} INFO -  at 11.8s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 08-31 22:00:11] {2282} INFO - iteration 14, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:12] {2466} INFO -  at 12.7s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 08-31 22:00:12] {2282} INFO - iteration 15, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:13] {2466} INFO -  at 13.5s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 08-31 22:00:13] {2282} INFO - iteration 16, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:14] {2466} INFO -  at 14.7s,	estimator lgbm's best error=0.4390,	best estimator lgbm's best error=0.4390
    [flaml.automl.logger: 08-31 22:00:14] {2282} INFO - iteration 17, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:15] {2466} INFO -  at 15.5s,	estimator lgbm's best error=0.4390,	best estimator lgbm's best error=0.4390
    [flaml.automl.logger: 08-31 22:00:15] {2282} INFO - iteration 18, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:16] {2466} INFO -  at 16.3s,	estimator lgbm's best error=0.4390,	best estimator lgbm's best error=0.4390
    [flaml.automl.logger: 08-31 22:00:16] {2282} INFO - iteration 19, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:19] {2466} INFO -  at 19.7s,	estimator lgbm's best error=0.2429,	best estimator lgbm's best error=0.2429
    [flaml.automl.logger: 08-31 22:00:19] {2282} INFO - iteration 20, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:27] {2466} INFO -  at 27.9s,	estimator lgbm's best error=0.1999,	best estimator lgbm's best error=0.1999
    [flaml.automl.logger: 08-31 22:00:27] {2282} INFO - iteration 21, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:31] {2466} INFO -  at 31.2s,	estimator lgbm's best error=0.1999,	best estimator lgbm's best error=0.1999
    [flaml.automl.logger: 08-31 22:00:31] {2282} INFO - iteration 22, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:39] {2466} INFO -  at 39.7s,	estimator lgbm's best error=0.1969,	best estimator lgbm's best error=0.1969
    [flaml.automl.logger: 08-31 22:00:39] {2282} INFO - iteration 23, current learner lgbm
    [flaml.automl.logger: 08-31 22:00:42] {2466} INFO -  at 42.4s,	estimator lgbm's best error=0.1969,	best estimator lgbm's best error=0.1969
    [flaml.automl.logger: 08-31 22:00:42] {2282} INFO - iteration 24, current learner lgbm
    [flaml.automl.logger: 08-31 22:01:17] {2466} INFO -  at 77.2s,	estimator lgbm's best error=0.1758,	best estimator lgbm's best error=0.1758
    [flaml.automl.logger: 08-31 22:01:17] {2282} INFO - iteration 25, current learner lgbm
    [flaml.automl.logger: 08-31 22:01:45] {2466} INFO -  at 105.6s,	estimator lgbm's best error=0.1758,	best estimator lgbm's best error=0.1758
    [flaml.automl.logger: 08-31 22:01:45] {2282} INFO - iteration 26, current learner lgbm
    [flaml.automl.logger: 08-31 22:02:25] {2466} INFO -  at 145.3s,	estimator lgbm's best error=0.1758,	best estimator lgbm's best error=0.1758
    [flaml.automl.logger: 08-31 22:02:25] {2282} INFO - iteration 27, current learner lgbm
    [flaml.automl.logger: 08-31 22:02:45] {2466} INFO -  at 166.1s,	estimator lgbm's best error=0.1758,	best estimator lgbm's best error=0.1758
    [flaml.automl.logger: 08-31 22:02:45] {2282} INFO - iteration 28, current learner lgbm
    [flaml.automl.logger: 08-31 22:03:33] {2466} INFO -  at 213.8s,	estimator lgbm's best error=0.1758,	best estimator lgbm's best error=0.1758
    [flaml.automl.logger: 08-31 22:03:33] {2282} INFO - iteration 29, current learner lgbm
    [flaml.automl.logger: 08-31 22:04:48] {2466} INFO -  at 289.0s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 08-31 22:04:48] {2282} INFO - iteration 30, current learner lgbm
    [flaml.automl.logger: 08-31 22:05:28] {2466} INFO -  at 329.1s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 08-31 22:05:28] {2282} INFO - iteration 31, current learner lgbm
    [flaml.automl.logger: 08-31 22:06:21] {2466} INFO -  at 381.7s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 08-31 22:06:21] {2282} INFO - iteration 32, current learner lgbm
    [flaml.automl.logger: 08-31 22:07:21] {2466} INFO -  at 441.9s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 08-31 22:07:21] {2282} INFO - iteration 33, current learner lgbm
    [flaml.automl.logger: 08-31 22:08:39] {2466} INFO -  at 519.7s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 08-31 22:08:39] {2282} INFO - iteration 34, current learner lgbm
    [flaml.automl.logger: 08-31 22:08:46] {2466} INFO -  at 526.2s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 08-31 22:08:46] {2282} INFO - iteration 35, current learner lgbm
    [flaml.automl.logger: 08-31 22:09:01] {2466} INFO -  at 541.8s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 08-31 22:09:01] {2282} INFO - iteration 36, current learner lgbm
    [flaml.automl.logger: 08-31 22:10:43] {2466} INFO -  at 643.7s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 08-31 22:10:43] {2282} INFO - iteration 37, current learner lgbm
    [flaml.automl.logger: 08-31 22:11:33] {2466} INFO -  at 693.9s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 08-31 22:11:33] {2282} INFO - iteration 38, current learner lgbm
    [flaml.automl.logger: 08-31 22:12:00] {2466} INFO -  at 720.9s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 08-31 22:12:00] {2282} INFO - iteration 39, current learner lgbm
    [flaml.automl.logger: 08-31 22:13:35] {2466} INFO -  at 816.2s,	estimator lgbm's best error=0.1716,	best estimator lgbm's best error=0.1716
    [flaml.automl.logger: 08-31 22:13:35] {2282} INFO - iteration 40, current learner lgbm
    [flaml.automl.logger: 08-31 22:14:30] {2466} INFO -  at 870.4s,	estimator lgbm's best error=0.1705,	best estimator lgbm's best error=0.1705
    [flaml.automl.logger: 08-31 22:14:30] {2282} INFO - iteration 41, current learner lgbm
    [flaml.automl.logger: 08-31 22:15:06] {2466} INFO -  at 906.7s,	estimator lgbm's best error=0.1705,	best estimator lgbm's best error=0.1705
    [flaml.automl.logger: 08-31 22:15:06] {2282} INFO - iteration 42, current learner lgbm
    [flaml.automl.logger: 08-31 22:16:27] {2466} INFO -  at 988.1s,	estimator lgbm's best error=0.1705,	best estimator lgbm's best error=0.1705
    [flaml.automl.logger: 08-31 22:16:27] {2282} INFO - iteration 43, current learner lgbm
    [flaml.automl.logger: 08-31 22:17:48] {2466} INFO -  at 1068.8s,	estimator lgbm's best error=0.1705,	best estimator lgbm's best error=0.1705
    [flaml.automl.logger: 08-31 22:17:48] {2282} INFO - iteration 44, current learner lgbm
    [flaml.automl.logger: 08-31 22:17:53] {2466} INFO -  at 1073.8s,	estimator lgbm's best error=0.1705,	best estimator lgbm's best error=0.1705
    [flaml.automl.logger: 08-31 22:17:53] {2282} INFO - iteration 45, current learner lgbm
    [flaml.automl.logger: 08-31 22:18:36] {2466} INFO -  at 1116.6s,	estimator lgbm's best error=0.1705,	best estimator lgbm's best error=0.1705
    [flaml.automl.logger: 08-31 22:18:36] {2282} INFO - iteration 46, current learner lgbm
    [flaml.automl.logger: 08-31 22:19:28] {2466} INFO -  at 1169.2s,	estimator lgbm's best error=0.1705,	best estimator lgbm's best error=0.1705
    [flaml.automl.logger: 08-31 22:19:28] {2282} INFO - iteration 47, current learner lgbm
    [flaml.automl.logger: 08-31 22:22:02] {2466} INFO -  at 1323.1s,	estimator lgbm's best error=0.1700,	best estimator lgbm's best error=0.1700
    [flaml.automl.logger: 08-31 22:22:02] {2282} INFO - iteration 48, current learner lgbm
    [flaml.automl.logger: 08-31 22:25:07] {2466} INFO -  at 1507.7s,	estimator lgbm's best error=0.1691,	best estimator lgbm's best error=0.1691
    [flaml.automl.logger: 08-31 22:25:07] {2282} INFO - iteration 49, current learner lgbm
    [flaml.automl.logger: 08-31 22:32:57] {2466} INFO -  at 1977.3s,	estimator lgbm's best error=0.1691,	best estimator lgbm's best error=0.1691
    [flaml.automl.logger: 08-31 22:32:57] {2282} INFO - iteration 50, current learner lgbm
    [flaml.automl.logger: 08-31 22:33:48] {2466} INFO -  at 2028.6s,	estimator lgbm's best error=0.1691,	best estimator lgbm's best error=0.1691
    [flaml.automl.logger: 08-31 22:33:48] {2282} INFO - iteration 51, current learner lgbm
    [flaml.automl.logger: 08-31 22:37:25] {2466} INFO -  at 2245.9s,	estimator lgbm's best error=0.1691,	best estimator lgbm's best error=0.1691
    [flaml.automl.logger: 08-31 22:37:25] {2282} INFO - iteration 52, current learner lgbm
    """
    )
    return


@app.cell
def _(X_test_final, automl_wo_early_stopping, pd):
    pd.Series(automl_wo_early_stopping.predict(X_test_final)).value_counts()
    return


@app.cell
def _(X_test_final, automl_wo_early_stopping, save_submission):
    save_submission(automl_wo_early_stopping.predict(X_test_final), X_test_final)
    return


@app.cell
def _(pd):
    pd.read_csv("./submission.csv")["prediction"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Мягкое голосование top-k_""")
    return


@app.cell
def _(X_test_final, X_train_final, make_predict_with_top_k_models, y_train):
    y_pred_with_top_k_models = make_predict_with_top_k_models(
        X_train_final,
        y_train,
        X_test_final,
        path_to_log_file="./log_flaml_tuning_31_08_25T2159.log",
        top_k=5,
        predict_proba=False,
        threshold=0.5
    )
    return (y_pred_with_top_k_models,)


@app.cell
def _(pd, y_pred_with_top_k_models):
    poss_to_negs = pd.Series(y_pred_with_top_k_models).value_counts()
    poss_to_negs
    return


@app.cell
def _(pd):
    pd.read_csv("./submissions/submission_f1=0.7400.csv")["prediction"].value_counts()
    return


@app.cell
def _(pd):
    pd.read_csv("./submissions/submission_f1=0.7453.csv")["prediction"].value_counts()
    return


@app.cell
def _(X_test_final, save_submission, y_pred_with_top_k_models):
    save_submission(y_pred_with_top_k_models, X_test_final)
    return


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
