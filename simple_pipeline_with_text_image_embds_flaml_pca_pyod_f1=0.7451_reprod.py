import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Импорты_""")
    return


@app.cell
def _():
    import marimo as mo
    import os
    import typing as t
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mlflow
    import category_encoders as ce
    import lightgbm as lgb
    import random

    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.suod import SUOD
    from pyod.models.hbos import HBOS
    from pyod.models.iforest import IForest

    from datetime import datetime

    from tqdm import tqdm

    from pathlib import Path

    from joblib import parallel_backend

    from sklearn.utils.validation import check_random_state
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV, StratifiedKFold
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
        Path,
        Pipeline,
        QuantileTransformer,
        StratifiedKFold,
        TunedThresholdClassifierCV,
        VotingClassifier,
        ce,
        datetime,
        lgb,
        mo,
        np,
        os,
        pd,
        plt,
        px,
        random,
        t,
    )


@app.cell
def _():
    SEED = 42
    TASK = "classification"
    IMAGE_N_COMPONENTS = 100
    IMAGE_WINDOW = 10
    TEXT_WINDOW = 5
    TEXT_N_COMPONENTS = 50
    return IMAGE_N_COMPONENTS, SEED, TASK, TEXT_N_COMPONENTS


@app.cell
def _(SEED, np, os, random):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Вспомогательные функции_""")
    return


@app.cell
def _(Path, np, pd):
    def save_submission(y_pred: np.typing.ArrayLike, X_test: pd.DataFrame) -> None:
        PATH_TO_SUBMISSION_FILE = "./submission.csv"

        pd.DataFrame({
            "id": X_test.index, 
            "prediction": y_pred,
        }).to_csv(PATH_TO_SUBMISSION_FILE, index=False)

        if not Path(PATH_TO_SUBMISSION_FILE).exists():
            raise FileNotFoundError(f"Error! File {PATH_TO_SUBMISSION_FILE} not found ...")

        print(f"File {PATH_TO_SUBMISSION_FILE} was recorded successfully!")
    return (save_submission,)


@app.cell
def _(Path, pd, plt, t):
    def get_ratio_to_f1(
        test_negs: t.Optional[float] = None,
        figsize: tuple[int, int] = (10, 5),
    ):
        fig, ax = plt.subplots(figsize=figsize)

        skip_files = 0
        ratio_to_f1 = []
        for submission in Path("./submissions").glob("*.csv"):
            try:
                f1 = float(Path(submission).with_suffix("").name.split("=")[-1])
                _ratio = pd.read_csv(submission, index_col=False)["prediction"].value_counts()
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
def _(Path, VotingClassifier, lgb, np, path_to_log_file, pd, t):
    class Booster(t.NamedTuple):
        model_name: str
        model: lgb.LGBMClassifier

    class Result(t.NamedTuple):
        model: "Model"
        predict: np.typing.ArrayLike


    def make_predict_with_top_k_models(
        X_train: pd.DataFrame,
        y_train: np.typing.ArrayLike,
        X_test: pd.DataFrame,
        path_to_log_dir: str, 
        log_file: t.Optional[str] = None,
        get_last_log_file: bool = True,
        top_k: int = 5,
        voting: str = "soft",
        predict_proba: bool = False,
        threshold: float = 0.5,
        weights: t.Optional[np.typing.ArrayLike] = None,
        n_jobs: int = 1,
    ) -> Result: 
        from pprint import pprint 

        NEGS_POSS_MSG = "[{model_name}] 0/1: {negs_poss[1]} / {negs_poss[0]}"

        if top_k < 1:
            raise ValueError(f"Error! top_k must be grater then 0")

        if log_file is not None: 
            path_to_log_file = Path(path_to_log_dir) / Path(path_to_log_file)
            print(f"[NOTE] The file {str(path_to_log_file)!r} will be use ...")
            trials_log = pd.read_json(path_to_log_file, lines=True)
        else:
            log_files: list[Path] = []
            for log_file in Path(path_to_log_dir).glob("*.log"):
                log_files.append(log_file.absolute())

            path_to_log_file = sorted(log_files)[-1]
            print(f"[NOTE] The last file {str(path_to_log_file)!r} will be use ...")
            trials_log = pd.read_json(path_to_log_file, lines=True)

        trials_log = trials_log \
            .sort_values(["validation_loss", "record_id"]) \
            .iloc[:top_k]

        models = []
        print("Run ...")
        for record in trials_log.itertuples():
            pprint(f"MODEL CONFIG: {record.config}")
            model = lgb.LGBMClassifier(
                **record.config,
                n_jobs=n_jobs,
            )
            models.append(Booster(model_name=f"clf-{int(record.record_id)}", model=model))

        print(f"[NOTE] Get {len(models)} models ...")

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
            model = models[0].model.fit(X_train, y_train)

            negs_poss = pd.Series(model.predict(X_test)).value_counts().values
            print(NEGS_POSS_MSG.format(model_name=model.__class__.__name__, negs_poss=negs_poss))

        print(f"Predict proba mode: {predict_proba}")
        y_test_pred = (
            model.predict_proba(X_test)[:, 1]
            if predict_proba else
            model.predict(X_test)
        )

        return Result(
            model=model,
            predict=(
                (y_test_pred > threshold).astype(int)
                if predict_proba else
                y_test_pred
            )
        )
    return (make_predict_with_top_k_models,)


@app.function
def add_cols_to_original_and_custom_col_names(
    original_and_custom_col_names: list[str],
    new_col_names: list[str]
) -> None:
    new_cols_count = 0
    for new_col in new_col_names:  # Убрать set() - сохраняем порядок
        if new_col not in original_and_custom_col_names:
            original_and_custom_col_names.append(new_col)
            new_cols_count += 1

    print(f"Was added {new_cols_count} new cols. Total cols: {len(original_and_custom_col_names)}")


@app.function
def remove_cols_from_original_and_custom_col_names(
    original_and_custom_col_names: list[str],
    col_names_for_removed: list[str],
) -> None:
    # Сохраняем порядок остающихся колонок
    new_list = [col for col in original_and_custom_col_names 
               if col not in col_names_for_removed]

    original_and_custom_col_names.clear()
    original_and_custom_col_names.extend(new_list)

    removed_count = len(original_and_custom_col_names) - len(new_list)
    print(f"Was removed {removed_count} cols. Total cols: {len(original_and_custom_col_names)}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Зависимость метрики F1 от отношения числа позитивов к числу негативов_""")
    return


@app.cell
def _(get_ratio_to_f1):
    get_ratio_to_f1(test_negs=1213)
    return


@app.cell
def _(pd):
    train = pd.read_csv("./ml_ozon_сounterfeit_data/ml_ozon_сounterfeit_train.csv", index_col=0).sort_index()
    return (train,)


@app.cell
def _(pd, train):
    print(f"Train shape: {train.shape}, Train checksum: {pd.util.hash_pandas_object(train).sum()}")
    return


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
    train_with_text_embds = pd.read_csv("./ml_ozon_сounterfeit_data/train_with_text_embeddings_after_clean_all-MiniLM-L6-v2.csv", index_col=0).sort_index()
    test_with_text_embds = pd.read_csv("./ml_ozon_сounterfeit_data//test_with_text_embeddings_after_clean_all-MiniLM-L6-v2.csv", index_col=0).sort_index()
    return test_with_text_embds, train_with_text_embds


@app.cell
def _(load_embeddings):
    train_with_image_embds = load_embeddings("./ml_ozon_сounterfeit_data//train_image_embeddings_vit-base-patch16-224-in21k.npz").sort_index()
    test_with_image_embds = load_embeddings("./ml_ozon_сounterfeit_data//test_image_embeddings_vit-base-patch16-224-in21k.npz").sort_index()

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
    train_with_text_image_embds = train_with_text_embds.join(train_with_image_embds, on="ItemID", how="left").sort_index()
    test_with_text_image_embds = test_with_text_embds.join(test_with_image_embds, on="ItemID", how="left").sort_index()
    return test_with_text_image_embds, train_with_text_image_embds


@app.cell
def _(pd, train_with_text_image_embds):
    print(f"Train-with-text-image-embds shape: {train_with_text_image_embds.shape}, Train checksum: {pd.util.hash_pandas_object(train_with_text_image_embds).sum()}")
    return


@app.cell
def _(train_with_text_image_embds):
    image_embds_col_names = [col_name for col_name in train_with_text_image_embds.columns if col_name.startswith("vit-base-patch16-224-in21k")]
    return (image_embds_col_names,)


@app.cell
def _(train_with_text_image_embds):
    desc_embds_col_names = [col_name for col_name in train_with_text_image_embds.columns if col_name.startswith("desc_embed")]
    name_rus_embds_col_names = [col_name for col_name in train_with_text_image_embds.columns if col_name.startswith("name_rus_embed")]
    return desc_embds_col_names, name_rus_embds_col_names


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
    original_and_custom_col_names = [col_name for col_name in X_train.columns if col_name not in remove_col_names]

    len(original_and_custom_col_names)
    return (original_and_custom_col_names,)


@app.cell
def _(X_test, X_train, original_and_custom_col_names):
    rating_5_count_ratio_col_name = "rating_5_count_ratio"

    X_train[rating_5_count_ratio_col_name] = X_train["rating_5_count"] / (
        X_train["rating_1_count"]
        + X_train["rating_2_count"]
        + X_train["rating_3_count"]
        + X_train["rating_4_count"]
        + X_train["rating_5_count"]
    )

    X_test[rating_5_count_ratio_col_name] = X_test["rating_5_count"] / (
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

    X_train[item_count_fake_returns_spike7_col_name] = (
        X_train["item_count_fake_returns30"] > 3 * X_train["item_count_fake_returns7"]
    ).astype(int)

    X_test[item_count_fake_returns_spike7_col_name] = (
        X_test["item_count_fake_returns30"] > 3 * X_test["item_count_fake_returns7"]
    ).astype(int)

    add_cols_to_original_and_custom_col_names(original_and_custom_col_names, new_col_names=[item_count_fake_returns_spike7_col_name])
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
def _(ColumnTransformer, SEED, binary_features, categorical_features, ce):
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
                ce.CatBoostEncoder(cols=binary_features, random_state=SEED),
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
    detector = COPOD(contamination=CONTAMINATION, n_jobs=1)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Индекс кластера как признак_""")
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
def _(SEED, ce):
    kmeans_encoder = ce.CatBoostEncoder(cols=["kmeans50"], random_state=SEED)
    return (kmeans_encoder,)


@app.cell
def _(X_test_final_, X_train_final_, kmeans_encoder, y_train):
    sorted_col_names = sorted(X_train_final_.columns.tolist())
    X_train_final = kmeans_encoder.fit_transform(X_train_final_, y_train)[sorted_col_names]
    X_test_final = kmeans_encoder.transform(X_test_final_)[sorted_col_names]
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
def _(SEED, np, y_train):
    SCALE_POS_WEIGHT = np.sum(y_train == 0) / np.sum(y_train == 1)

    # Альтернативно вместо `scale_pos_weight` можно использовать `is_unbalance = True`
    custom_hp = {
        "lgbm": {
            "scale_pos_weight": {
                "domain": SCALE_POS_WEIGHT, 
            },
            "random_state": {
                "domain": SEED, 
            },
            "deterministic": {
                "domain": True,
            },
            "force_col_wise": {
                "domain": True,
            },
        },
    }
    return (custom_hp,)


@app.cell
def _(AutoML, SEED, TASK, X_train_final, custom_hp, datetime, y_train):
    ### ЗАПУСК БЕЗ РАННЕЙ ОСТАНОВКИ

    datetime_postfix = datetime.now().strftime("%d%m%y_T%H%M%S")

    automl_wo_early_stopping = AutoML()
    automl_wo_early_stopping.fit(
        X_train_final,
        y_train,
        task=TASK,
        time_budget=250,
        estimator_list=(
            "lgbm",
        ),
        eval_method="holdout",
        split_ratio=0.2,
        metric="f1",
        split_type="stratified",
        custom_hp=custom_hp,
        log_file_name=f"./logs/log_flaml_tuning_{datetime_postfix}.log",
        log_type="all",
        seed=10 * SEED,
        early_stop=True,
    )
    return (automl_wo_early_stopping,)


@app.cell
def _(mo):
    mo.md(
        r"""
    [flaml.automl.logger: 09-01 23:02:51] {1752} INFO - task = classification
    [flaml.automl.logger: 09-01 23:02:51] {1763} INFO - Evaluation method: holdout
    [flaml.automl.logger: 09-01 23:02:52] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 09-01 23:02:52] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 09-01 23:02:52] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:52] {2417} INFO - Estimated sufficient time budget=10391s. Estimated necessary time budget=10s.
    [flaml.automl.logger: 09-01 23:02:52] {2466} INFO -  at 5.4s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 09-01 23:02:52] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:52] {2466} INFO -  at 5.5s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 09-01 23:02:52] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:52] {2466} INFO -  at 5.6s,	estimator lgbm's best error=0.5112,	best estimator lgbm's best error=0.5112
    [flaml.automl.logger: 09-01 23:02:52] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:52] {2466} INFO -  at 5.6s,	estimator lgbm's best error=0.5112,	best estimator lgbm's best error=0.5112
    [flaml.automl.logger: 09-01 23:02:52] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:52] {2466} INFO -  at 5.7s,	estimator lgbm's best error=0.5112,	best estimator lgbm's best error=0.5112
    [flaml.automl.logger: 09-01 23:02:52] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:52] {2466} INFO -  at 5.8s,	estimator lgbm's best error=0.5112,	best estimator lgbm's best error=0.5112
    [flaml.automl.logger: 09-01 23:02:52] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:52] {2466} INFO -  at 5.9s,	estimator lgbm's best error=0.5112,	best estimator lgbm's best error=0.5112
    [flaml.automl.logger: 09-01 23:02:52] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:53] {2466} INFO -  at 6.7s,	estimator lgbm's best error=0.4598,	best estimator lgbm's best error=0.4598
    [flaml.automl.logger: 09-01 23:02:53] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:54] {2466} INFO -  at 7.5s,	estimator lgbm's best error=0.4598,	best estimator lgbm's best error=0.4598
    [flaml.automl.logger: 09-01 23:02:54] {2282} INFO - iteration 9, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:54] {2466} INFO -  at 8.3s,	estimator lgbm's best error=0.4598,	best estimator lgbm's best error=0.4598
    [flaml.automl.logger: 09-01 23:02:54] {2282} INFO - iteration 10, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:55] {2466} INFO -  at 9.0s,	estimator lgbm's best error=0.4598,	best estimator lgbm's best error=0.4598
    [flaml.automl.logger: 09-01 23:02:55] {2282} INFO - iteration 11, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:56] {2466} INFO -  at 9.8s,	estimator lgbm's best error=0.4598,	best estimator lgbm's best error=0.4598
    [flaml.automl.logger: 09-01 23:02:56] {2282} INFO - iteration 12, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:57] {2466} INFO -  at 10.6s,	estimator lgbm's best error=0.4598,	best estimator lgbm's best error=0.4598
    [flaml.automl.logger: 09-01 23:02:57] {2282} INFO - iteration 13, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:58] {2466} INFO -  at 11.4s,	estimator lgbm's best error=0.4598,	best estimator lgbm's best error=0.4598
    [flaml.automl.logger: 09-01 23:02:58] {2282} INFO - iteration 14, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:58] {2466} INFO -  at 12.1s,	estimator lgbm's best error=0.4598,	best estimator lgbm's best error=0.4598
    [flaml.automl.logger: 09-01 23:02:58] {2282} INFO - iteration 15, current learner lgbm
    [flaml.automl.logger: 09-01 23:02:59] {2466} INFO -  at 12.9s,	estimator lgbm's best error=0.4598,	best estimator lgbm's best error=0.4598
    [flaml.automl.logger: 09-01 23:02:59] {2282} INFO - iteration 16, current learner lgbm
    [flaml.automl.logger: 09-01 23:03:00] {2466} INFO -  at 13.9s,	estimator lgbm's best error=0.4372,	best estimator lgbm's best error=0.4372
    [flaml.automl.logger: 09-01 23:03:00] {2282} INFO - iteration 17, current learner lgbm
    [flaml.automl.logger: 09-01 23:03:01] {2466} INFO -  at 14.7s,	estimator lgbm's best error=0.4372,	best estimator lgbm's best error=0.4372
    [flaml.automl.logger: 09-01 23:03:01] {2282} INFO - iteration 18, current learner lgbm
    [flaml.automl.logger: 09-01 23:03:02] {2466} INFO -  at 15.4s,	estimator lgbm's best error=0.4372,	best estimator lgbm's best error=0.4372
    [flaml.automl.logger: 09-01 23:03:02] {2282} INFO - iteration 19, current learner lgbm
    [flaml.automl.logger: 09-01 23:03:05] {2466} INFO -  at 18.5s,	estimator lgbm's best error=0.2455,	best estimator lgbm's best error=0.2455
    [flaml.automl.logger: 09-01 23:03:05] {2282} INFO - iteration 20, current learner lgbm
    [flaml.automl.logger: 09-01 23:03:13] {2466} INFO -  at 26.4s,	estimator lgbm's best error=0.1988,	best estimator lgbm's best error=0.1988
    [flaml.automl.logger: 09-01 23:03:13] {2282} INFO - iteration 21, current learner lgbm
    [flaml.automl.logger: 09-01 23:03:16] {2466} INFO -  at 29.5s,	estimator lgbm's best error=0.1988,	best estimator lgbm's best error=0.1988
    [flaml.automl.logger: 09-01 23:03:16] {2282} INFO - iteration 22, current learner lgbm
    [flaml.automl.logger: 09-01 23:03:24] {2466} INFO -  at 37.5s,	estimator lgbm's best error=0.1988,	best estimator lgbm's best error=0.1988
    [flaml.automl.logger: 09-01 23:03:24] {2282} INFO - iteration 23, current learner lgbm
    [flaml.automl.logger: 09-01 23:03:31] {2466} INFO -  at 45.0s,	estimator lgbm's best error=0.1988,	best estimator lgbm's best error=0.1988
    [flaml.automl.logger: 09-01 23:03:31] {2282} INFO - iteration 24, current learner lgbm
    [flaml.automl.logger: 09-01 23:03:34] {2466} INFO -  at 47.5s,	estimator lgbm's best error=0.1988,	best estimator lgbm's best error=0.1988
    [flaml.automl.logger: 09-01 23:03:34] {2282} INFO - iteration 25, current learner lgbm
    [flaml.automl.logger: 09-01 23:04:02] {2466} INFO -  at 76.2s,	estimator lgbm's best error=0.1816,	best estimator lgbm's best error=0.1816
    [flaml.automl.logger: 09-01 23:04:02] {2282} INFO - iteration 26, current learner lgbm
    [flaml.automl.logger: 09-01 23:04:29] {2466} INFO -  at 102.4s,	estimator lgbm's best error=0.1799,	best estimator lgbm's best error=0.1799
    [flaml.automl.logger: 09-01 23:04:29] {2282} INFO - iteration 27, current learner lgbm
    [flaml.automl.logger: 09-01 23:04:58] {2466} INFO -  at 131.6s,	estimator lgbm's best error=0.1799,	best estimator lgbm's best error=0.1799
    [flaml.automl.logger: 09-01 23:04:58] {2282} INFO - iteration 28, current learner lgbm
    [flaml.automl.logger: 09-01 23:05:13] {2466} INFO -  at 147.3s,	estimator lgbm's best error=0.1799,	best estimator lgbm's best error=0.1799
    [flaml.automl.logger: 09-01 23:05:13] {2282} INFO - iteration 29, current learner lgbm
    [flaml.automl.logger: 09-01 23:05:47] {2466} INFO -  at 181.0s,	estimator lgbm's best error=0.1799,	best estimator lgbm's best error=0.1799
    [flaml.automl.logger: 09-01 23:05:47] {2282} INFO - iteration 30, current learner lgbm
    [flaml.automl.logger: 09-01 23:06:27] {2466} INFO -  at 220.6s,	estimator lgbm's best error=0.1799,	best estimator lgbm's best error=0.1799
    [flaml.automl.logger: 09-01 23:06:27] {2282} INFO - iteration 31, current learner lgbm
    [flaml.automl.logger: 09-01 23:06:48] {2466} INFO -  at 242.2s,	estimator lgbm's best error=0.1799,	best estimator lgbm's best error=0.1799
    [flaml.automl.logger: 09-01 23:06:48] {2282} INFO - iteration 32, current learner lgbm
    [flaml.automl.logger: 09-01 23:07:14] {2466} INFO -  at 268.1s,	estimator lgbm's best error=0.1793,	best estimator lgbm's best error=0.1793
    [flaml.automl.logger: 09-01 23:07:14] {2282} INFO - iteration 33, current learner lgbm
    [flaml.automl.logger: 09-01 23:07:40] {2466} INFO -  at 293.9s,	estimator lgbm's best error=0.1793,	best estimator lgbm's best error=0.1793
    [flaml.automl.logger: 09-01 23:07:40] {2282} INFO - iteration 34, current learner lgbm
    [flaml.automl.logger: 09-01 23:08:20] {2466} INFO -  at 333.4s,	estimator lgbm's best error=0.1793,	best estimator lgbm's best error=0.1793
    [flaml.automl.logger: 09-01 23:08:20] {2282} INFO - iteration 35, current learner lgbm
    [flaml.automl.logger: 09-01 23:08:23] {2466} INFO -  at 336.9s,	estimator lgbm's best error=0.1793,	best estimator lgbm's best error=0.1793
    [flaml.automl.logger: 09-01 23:08:23] {2282} INFO - iteration 36, current learner lgbm
    [flaml.automl.logger: 09-01 23:08:33] {2466} INFO -  at 347.1s,	estimator lgbm's best error=0.1793,	best estimator lgbm's best error=0.1793
    [flaml.automl.logger: 09-01 23:08:33] {2282} INFO - iteration 37, current learner lgbm
    [flaml.automl.logger: 09-01 23:09:04] {2466} INFO -  at 378.1s,	estimator lgbm's best error=0.1790,	best estimator lgbm's best error=0.1790
    [flaml.automl.logger: 09-01 23:09:04] {2282} INFO - iteration 38, current learner lgbm
    [flaml.automl.logger: 09-01 23:09:29] {2466} INFO -  at 402.6s,	estimator lgbm's best error=0.1790,	best estimator lgbm's best error=0.1790
    [flaml.automl.logger: 09-01 23:09:29] {2282} INFO - iteration 39, current learner lgbm
    [flaml.automl.logger: 09-01 23:10:18] {2466} INFO -  at 451.9s,	estimator lgbm's best error=0.1761,	best estimator lgbm's best error=0.1761
    [flaml.automl.logger: 09-01 23:10:18] {2282} INFO - iteration 40, current learner lgbm
    [flaml.automl.logger: 09-01 23:11:33] {2466} INFO -  at 526.7s,	estimator lgbm's best error=0.1761,	best estimator lgbm's best error=0.1761
    [flaml.automl.logger: 09-01 23:11:33] {2282} INFO - iteration 41, current learner lgbm
    [flaml.automl.logger: 09-01 23:11:58] {2466} INFO -  at 552.0s,	estimator lgbm's best error=0.1761,	best estimator lgbm's best error=0.1761
    [flaml.automl.logger: 09-01 23:11:58] {2282} INFO - iteration 42, current learner lgbm
    [flaml.automl.logger: 09-01 23:12:33] {2466} INFO -  at 587.0s,	estimator lgbm's best error=0.1758,	best estimator lgbm's best error=0.1758
    [flaml.automl.logger: 09-01 23:12:33] {2282} INFO - iteration 43, current learner lgbm
    [flaml.automl.logger: 09-01 23:13:23] {2466} INFO -  at 636.5s,	estimator lgbm's best error=0.1758,	best estimator lgbm's best error=0.1758
    [flaml.automl.logger: 09-01 23:13:23] {2282} INFO - iteration 44, current learner lgbm
    [flaml.automl.logger: 09-01 23:14:33] {2466} INFO -  at 706.7s,	estimator lgbm's best error=0.1758,	best estimator lgbm's best error=0.1758
    [flaml.automl.logger: 09-01 23:14:33] {2282} INFO - iteration 45, current learner lgbm
    [flaml.automl.logger: 09-01 23:14:36] {2466} INFO -  at 709.9s,	estimator lgbm's best error=0.1758,	best estimator lgbm's best error=0.1758
    [flaml.automl.logger: 09-01 23:14:36] {2282} INFO - iteration 46, current learner lgbm
    [flaml.automl.logger: 09-01 23:15:02] {2466} INFO -  at 735.5s,	estimator lgbm's best error=0.1758,	best estimator lgbm's best error=0.1758
    [flaml.automl.logger: 09-01 23:15:02] {2282} INFO - iteration 47, current learner lgbm
    [flaml.automl.logger: 09-01 23:15:33] {2466} INFO -  at 767.3s,	estimator lgbm's best error=0.1742,	best estimator lgbm's best error=0.1742
    [flaml.automl.logger: 09-01 23:15:33] {2282} INFO - iteration 48, current learner lgbm
    [flaml.automl.logger: 09-01 23:17:56] {2466} INFO -  at 909.5s,	estimator lgbm's best error=0.1742,	best estimator lgbm's best error=0.1742
    [flaml.automl.logger: 09-01 23:17:56] {2282} INFO - iteration 49, current learner lgbm
    [flaml.automl.logger: 09-01 23:18:03] {2466} INFO -  at 917.0s,	estimator lgbm's best error=0.1742,	best estimator lgbm's best error=0.1742
    [flaml.automl.logger: 09-01 23:18:03] {2282} INFO - iteration 50, current learner lgbm
    [flaml.automl.logger: 09-01 23:19:01] {2466} INFO -  at 975.1s,	estimator lgbm's best error=0.1742,	best estimator lgbm's best error=0.1742
    [flaml.automl.logger: 09-01 23:19:01] {2282} INFO - iteration 51, current learner lgbm
    [flaml.automl.logger: 09-01 23:19:12] {2466} INFO -  at 986.0s,	estimator lgbm's best error=0.1742,	best estimator lgbm's best error=0.1742
    [flaml.automl.logger: 09-01 23:19:12] {2282} INFO - iteration 52, current learner lgbm
    [flaml.automl.logger: 09-01 23:23:12] {2466} INFO -  at 1226.3s,	estimator lgbm's best error=0.1742,	best estimator lgbm's best error=0.1742
    [flaml.automl.logger: 09-01 23:23:12] {2282} INFO - iteration 53, current learner lgbm
    [flaml.automl.logger: 09-01 23:23:17] {2466} INFO -  at 1231.2s,	estimator lgbm's best error=0.1742,	best estimator lgbm's best error=0.1742
    [flaml.automl.logger: 09-01 23:23:17] {2282} INFO - iteration 54, current learner lgbm
    [flaml.automl.logger: 09-01 23:24:04] {2466} INFO -  at 1277.7s,	estimator lgbm's best error=0.1742,	best estimator lgbm's best error=0.1742
    [flaml.automl.logger: 09-01 23:24:04] {2282} INFO - iteration 55, current learner lgbm
    [flaml.automl.logger: 09-01 23:24:28] {2466} INFO -  at 1301.9s,	estimator lgbm's best error=0.1742,	best estimator lgbm's best error=0.1742
    [flaml.automl.logger: 09-01 23:24:28] {2282} INFO - iteration 56, current learner lgbm
    [flaml.automl.logger: 09-01 23:24:42] {2466} INFO -  at 1315.7s,	estimator lgbm's best error=0.1742,	best estimator lgbm's best error=0.1742
    [flaml.automl.logger: 09-01 23:24:42] {2282} INFO - iteration 57, current learner lgbm
    [flaml.automl.logger: 09-01 23:25:40] {2466} INFO -  at 1373.4s,	estimator lgbm's best error=0.1742,	best estimator lgbm's best error=0.1742
    [flaml.automl.logger: 09-01 23:25:40] {2282} INFO - iteration 58, current learner lgbm
    [flaml.automl.logger: 09-01 23:25:52] {2466} INFO -  at 1385.5s,	estimator lgbm's best error=0.1742,	best estimator lgbm's best error=0.1742
    [flaml.automl.logger: 09-01 23:25:52] {2282} INFO - iteration 59, current learner lgbm
    [flaml.automl.logger: 09-01 23:26:41] {2466} INFO -  at 1434.8s,	estimator lgbm's best error=0.1740,	best estimator lgbm's best error=0.1740
    [flaml.automl.logger: 09-01 23:26:41] {2282} INFO - iteration 60, current learner lgbm
    [flaml.automl.logger: 09-01 23:27:26] {2466} INFO -  at 1479.5s,	estimator lgbm's best error=0.1740,	best estimator lgbm's best error=0.1740
    [flaml.automl.logger: 09-01 23:27:26] {2282} INFO - iteration 61, current learner lgbm
    [flaml.automl.logger: 09-01 23:27:44] {2466} INFO -  at 1498.2s,	estimator lgbm's best error=0.1740,	best estimator lgbm's best error=0.1740
    [flaml.automl.logger: 09-01 23:27:44] {2282} INFO - iteration 62, current learner lgbm
    [flaml.automl.logger: 09-01 23:29:37] {2466} INFO -  at 1610.7s,	estimator lgbm's best error=0.1740,	best estimator lgbm's best error=0.1740
    [flaml.automl.logger: 09-01 23:29:37] {2282} INFO - iteration 63, current learner lgbm
    [flaml.automl.logger: 09-01 23:30:02] {2466} INFO -  at 1636.1s,	estimator lgbm's best error=0.1740,	best estimator lgbm's best error=0.1740
    [flaml.automl.logger: 09-01 23:30:02] {2282} INFO - iteration 64, current learner lgbm
    [flaml.automl.logger: 09-01 23:31:18] {2466} INFO -  at 1711.8s,	estimator lgbm's best error=0.1740,	best estimator lgbm's best error=0.1740
    [flaml.automl.logger: 09-01 23:31:18] {2282} INFO - iteration 65, current learner lgbm
    [flaml.automl.logger: 09-01 23:31:55] {2466} INFO -  at 1748.6s,	estimator lgbm's best error=0.1740,	best estimator lgbm's best error=0.1740
    [flaml.automl.logger: 09-01 23:31:55] {2282} INFO - iteration 66, current learner lgbm
    [flaml.automl.logger: 09-01 23:32:36] {2466} INFO -  at 1790.1s,	estimator lgbm's best error=0.1740,	best estimator lgbm's best error=0.1740
    [flaml.automl.logger: 09-01 23:32:36] {2282} INFO - iteration 67, current learner lgbm
    [flaml.automl.logger: 09-01 23:33:10] {2466} INFO -  at 1823.8s,	estimator lgbm's best error=0.1740,	best estimator lgbm's best error=0.1740
    [flaml.automl.logger: 09-01 23:33:10] {2282} INFO - iteration 68, current learner lgbm
    [flaml.automl.logger: 09-01 23:35:17] {2466} INFO -  at 1951.1s,	estimator lgbm's best error=0.1735,	best estimator lgbm's best error=0.1735
    [flaml.automl.logger: 09-01 23:35:17] {2282} INFO - iteration 69, current learner lgbm
    [flaml.automl.logger: 09-01 23:36:07] {2466} INFO -  at 2001.1s,	estimator lgbm's best error=0.1735,	best estimator lgbm's best error=0.1735
    [flaml.automl.logger: 09-01 23:36:07] {2282} INFO - iteration 70, current learner lgbm
    [flaml.automl.logger: 09-01 23:37:30] {2466} INFO -  at 2083.9s,	estimator lgbm's best error=0.1735,	best estimator lgbm's best error=0.1735
    [flaml.automl.logger: 09-01 23:37:30] {2282} INFO - iteration 71, current learner lgbm
    [flaml.automl.logger: 09-01 23:37:59] {2466} INFO -  at 2112.7s,	estimator lgbm's best error=0.1735,	best estimator lgbm's best error=0.1735
    [flaml.automl.logger: 09-01 23:37:59] {2282} INFO - iteration 72, current learner lgbm
    [flaml.automl.logger: 09-01 23:39:03] {2466} INFO -  at 2177.0s,	estimator lgbm's best error=0.1735,	best estimator lgbm's best error=0.1735
    [flaml.automl.logger: 09-01 23:39:03] {2282} INFO - iteration 73, current learner lgbm
    [flaml.automl.logger: 09-01 23:40:51] {2466} INFO -  at 2284.5s,	estimator lgbm's best error=0.1735,	best estimator lgbm's best error=0.1735
    [flaml.automl.logger: 09-01 23:40:51] {2282} INFO - iteration 74, current learner lgbm
    [flaml.automl.logger: 09-01 23:41:43] {2466} INFO -  at 2337.1s,	estimator lgbm's best error=0.1735,	best estimator lgbm's best error=0.1735
    [flaml.automl.logger: 09-01 23:41:43] {2282} INFO - iteration 75, current learner lgbm
    [flaml.automl.logger: 09-01 23:42:46] {2466} INFO -  at 2400.2s,	estimator lgbm's best error=0.1735,	best estimator lgbm's best error=0.1735
    [flaml.automl.logger: 09-01 23:42:46] {2282} INFO - iteration 76, current learner lgbm
    [flaml.automl.logger: 09-01 23:43:21] {2466} INFO -  at 2435.2s,	estimator lgbm's best error=0.1735,	best estimator lgbm's best error=0.1735
    [flaml.automl.logger: 09-01 23:43:21] {2282} INFO - iteration 77, current learner lgbm
    [flaml.automl.logger: 09-01 23:45:41] {2466} INFO -  at 2575.3s,	estimator lgbm's best error=0.1735,	best estimator lgbm's best error=0.1735
    [flaml.automl.logger: 09-01 23:45:41] {2282} INFO - iteration 78, current learner lgbm
    [flaml.automl.logger: 09-01 23:46:29] {2466} INFO -  at 2623.2s,	estimator lgbm's best error=0.1735,	best estimator lgbm's best error=0.1735
    [flaml.automl.logger: 09-01 23:46:29] {2282} INFO - iteration 79, current learner lgbm
    [flaml.automl.logger: 09-01 23:49:30] {2466} INFO -  at 2803.6s,	estimator lgbm's best error=0.1721,	best estimator lgbm's best error=0.1721
    [flaml.automl.logger: 09-01 23:49:30] {2282} INFO - iteration 80, current learner lgbm
    [flaml.automl.logger: 09-01 23:50:23] {2466} INFO -  at 2856.5s,	estimator lgbm's best error=0.1721,	best estimator lgbm's best error=0.1721
    [flaml.automl.logger: 09-01 23:50:23] {2282} INFO - iteration 81, current learner lgbm
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    [flaml.automl.logger: 09-01 03:53:53] {1752} INFO - task = classification
    [flaml.automl.logger: 09-01 03:53:53] {1763} INFO - Evaluation method: holdout
    [flaml.automl.logger: 09-01 03:53:54] {1862} INFO - Minimizing error metric: 1-f1
    [flaml.automl.logger: 09-01 03:53:54] {1979} INFO - List of ML learners in AutoML Run: ['lgbm']
    [flaml.automl.logger: 09-01 03:53:54] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 09-01 03:53:54] {2417} INFO - Estimated sufficient time budget=11583s. Estimated necessary time budget=12s.
    [flaml.automl.logger: 09-01 03:53:54] {2466} INFO -  at 5.5s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 09-01 03:53:54] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 09-01 03:53:54] {2466} INFO -  at 5.6s,	estimator lgbm's best error=1.0000,	best estimator lgbm's best error=1.0000
    [flaml.automl.logger: 09-01 03:53:54] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 09-01 03:53:54] {2466} INFO -  at 5.7s,	estimator lgbm's best error=0.4908,	best estimator lgbm's best error=0.4908
    [flaml.automl.logger: 09-01 03:53:54] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 09-01 03:53:54] {2466} INFO -  at 5.8s,	estimator lgbm's best error=0.4908,	best estimator lgbm's best error=0.4908
    [flaml.automl.logger: 09-01 03:53:54] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 09-01 03:53:54] {2466} INFO -  at 5.9s,	estimator lgbm's best error=0.4908,	best estimator lgbm's best error=0.4908
    [flaml.automl.logger: 09-01 03:53:54] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 09-01 03:53:54] {2466} INFO -  at 6.0s,	estimator lgbm's best error=0.4908,	best estimator lgbm's best error=0.4908
    [flaml.automl.logger: 09-01 03:53:54] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 09-01 03:53:54] {2466} INFO -  at 6.1s,	estimator lgbm's best error=0.4908,	best estimator lgbm's best error=0.4908
    [flaml.automl.logger: 09-01 03:53:54] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 09-01 03:53:55] {2466} INFO -  at 6.9s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 09-01 03:53:55] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 09-01 03:53:56] {2466} INFO -  at 7.7s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 09-01 03:53:56] {2282} INFO - iteration 9, current learner lgbm
    [flaml.automl.logger: 09-01 03:53:57] {2466} INFO -  at 8.6s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 09-01 03:53:57] {2282} INFO - iteration 10, current learner lgbm
    [flaml.automl.logger: 09-01 03:53:57] {2466} INFO -  at 9.4s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 09-01 03:53:57] {2282} INFO - iteration 11, current learner lgbm
    [flaml.automl.logger: 09-01 03:53:58] {2466} INFO -  at 10.2s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 09-01 03:53:58] {2282} INFO - iteration 12, current learner lgbm
    [flaml.automl.logger: 09-01 03:53:59] {2466} INFO -  at 11.1s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 09-01 03:53:59] {2282} INFO - iteration 13, current learner lgbm
    [flaml.automl.logger: 09-01 03:54:00] {2466} INFO -  at 11.9s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 09-01 03:54:00] {2282} INFO - iteration 14, current learner lgbm
    [flaml.automl.logger: 09-01 03:54:01] {2466} INFO -  at 12.8s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 09-01 03:54:01] {2282} INFO - iteration 15, current learner lgbm
    [flaml.automl.logger: 09-01 03:54:02] {2466} INFO -  at 13.7s,	estimator lgbm's best error=0.4813,	best estimator lgbm's best error=0.4813
    [flaml.automl.logger: 09-01 03:54:02] {2282} INFO - iteration 16, current learner lgbm
    [flaml.automl.logger: 09-01 03:54:03] {2466} INFO -  at 14.8s,	estimator lgbm's best error=0.4390,	best estimator lgbm's best error=0.4390
    [flaml.automl.logger: 09-01 03:54:03] {2282} INFO - iteration 17, current learner lgbm
    [flaml.automl.logger: 09-01 03:54:04] {2466} INFO -  at 15.7s,	estimator lgbm's best error=0.4390,	best estimator lgbm's best error=0.4390
    [flaml.automl.logger: 09-01 03:54:04] {2282} INFO - iteration 18, current learner lgbm
    [flaml.automl.logger: 09-01 03:54:05] {2466} INFO -  at 16.5s,	estimator lgbm's best error=0.4390,	best estimator lgbm's best error=0.4390
    [flaml.automl.logger: 09-01 03:54:05] {2282} INFO - iteration 19, current learner lgbm
    [flaml.automl.logger: 09-01 03:54:08] {2466} INFO -  at 19.9s,	estimator lgbm's best error=0.2429,	best estimator lgbm's best error=0.2429
    [flaml.automl.logger: 09-01 03:54:08] {2282} INFO - iteration 20, current learner lgbm
    [flaml.automl.logger: 09-01 03:54:16] {2466} INFO -  at 28.2s,	estimator lgbm's best error=0.2009,	best estimator lgbm's best error=0.2009
    [flaml.automl.logger: 09-01 03:54:16] {2282} INFO - iteration 21, current learner lgbm
    [flaml.automl.logger: 09-01 03:54:20] {2466} INFO -  at 31.5s,	estimator lgbm's best error=0.2009,	best estimator lgbm's best error=0.2009
    [flaml.automl.logger: 09-01 03:54:20] {2282} INFO - iteration 22, current learner lgbm
    [flaml.automl.logger: 09-01 03:54:28] {2466} INFO -  at 40.0s,	estimator lgbm's best error=0.2007,	best estimator lgbm's best error=0.2007
    [flaml.automl.logger: 09-01 03:54:28] {2282} INFO - iteration 23, current learner lgbm
    [flaml.automl.logger: 09-01 03:54:31] {2466} INFO -  at 42.8s,	estimator lgbm's best error=0.2007,	best estimator lgbm's best error=0.2007
    [flaml.automl.logger: 09-01 03:54:31] {2282} INFO - iteration 24, current learner lgbm
    [flaml.automl.logger: 09-01 03:55:06] {2466} INFO -  at 78.1s,	estimator lgbm's best error=0.1762,	best estimator lgbm's best error=0.1762
    [flaml.automl.logger: 09-01 03:55:06] {2282} INFO - iteration 25, current learner lgbm
    [flaml.automl.logger: 09-01 03:55:34] {2466} INFO -  at 106.2s,	estimator lgbm's best error=0.1762,	best estimator lgbm's best error=0.1762
    [flaml.automl.logger: 09-01 03:55:34] {2282} INFO - iteration 26, current learner lgbm
    [flaml.automl.logger: 09-01 03:56:14] {2466} INFO -  at 146.4s,	estimator lgbm's best error=0.1760,	best estimator lgbm's best error=0.1760
    [flaml.automl.logger: 09-01 03:56:14] {2282} INFO - iteration 27, current learner lgbm
    [flaml.automl.logger: 09-01 03:56:36] {2466} INFO -  at 167.4s,	estimator lgbm's best error=0.1760,	best estimator lgbm's best error=0.1760
    [flaml.automl.logger: 09-01 03:56:36] {2282} INFO - iteration 28, current learner lgbm
    [flaml.automl.logger: 09-01 03:57:24] {2466} INFO -  at 215.6s,	estimator lgbm's best error=0.1760,	best estimator lgbm's best error=0.1760
    [flaml.automl.logger: 09-01 03:57:24] {2282} INFO - iteration 29, current learner lgbm
    [flaml.automl.logger: 09-01 03:58:39] {2466} INFO -  at 290.7s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 09-01 03:58:39] {2282} INFO - iteration 30, current learner lgbm
    [flaml.automl.logger: 09-01 03:59:19] {2466} INFO -  at 330.5s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 09-01 03:59:19] {2282} INFO - iteration 31, current learner lgbm
    [flaml.automl.logger: 09-01 04:00:10] {2466} INFO -  at 381.9s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 09-01 04:00:10] {2282} INFO - iteration 32, current learner lgbm
    [flaml.automl.logger: 09-01 04:01:11] {2466} INFO -  at 442.4s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 09-01 04:01:11] {2282} INFO - iteration 33, current learner lgbm
    [flaml.automl.logger: 09-01 04:02:30] {2466} INFO -  at 521.4s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 09-01 04:02:30] {2282} INFO - iteration 34, current learner lgbm
    [flaml.automl.logger: 09-01 04:02:36] {2466} INFO -  at 528.0s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 09-01 04:02:36] {2282} INFO - iteration 35, current learner lgbm
    [flaml.automl.logger: 09-01 04:02:52] {2466} INFO -  at 543.7s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 09-01 04:02:52] {2282} INFO - iteration 36, current learner lgbm
    [flaml.automl.logger: 09-01 04:04:32] {2466} INFO -  at 643.8s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 09-01 04:04:32] {2282} INFO - iteration 37, current learner lgbm
    [flaml.automl.logger: 09-01 04:05:23] {2466} INFO -  at 694.7s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 09-01 04:05:23] {2282} INFO - iteration 38, current learner lgbm
    [flaml.automl.logger: 09-01 04:05:50] {2466} INFO -  at 721.8s,	estimator lgbm's best error=0.1718,	best estimator lgbm's best error=0.1718
    [flaml.automl.logger: 09-01 04:05:50] {2282} INFO - iteration 39, current learner lgbm
    [flaml.automl.logger: 09-01 04:07:25] {2466} INFO -  at 817.0s,	estimator lgbm's best error=0.1711,	best estimator lgbm's best error=0.1711
    [flaml.automl.logger: 09-01 04:07:25] {2282} INFO - iteration 40, current learner lgbm
    [flaml.automl.logger: 09-01 04:08:19] {2466} INFO -  at 871.0s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:08:19] {2282} INFO - iteration 41, current learner lgbm
    [flaml.automl.logger: 09-01 04:08:56] {2466} INFO -  at 907.9s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:08:56] {2282} INFO - iteration 42, current learner lgbm
    [flaml.automl.logger: 09-01 04:10:17] {2466} INFO -  at 989.2s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:10:17] {2282} INFO - iteration 43, current learner lgbm
    [flaml.automl.logger: 09-01 04:11:39] {2466} INFO -  at 1071.4s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:11:39] {2282} INFO - iteration 44, current learner lgbm
    [flaml.automl.logger: 09-01 04:11:45] {2466} INFO -  at 1076.4s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:11:45] {2282} INFO - iteration 45, current learner lgbm
    [flaml.automl.logger: 09-01 04:12:28] {2466} INFO -  at 1119.7s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:12:28] {2282} INFO - iteration 46, current learner lgbm
    [flaml.automl.logger: 09-01 04:13:21] {2466} INFO -  at 1172.5s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:13:21] {2282} INFO - iteration 47, current learner lgbm
    [flaml.automl.logger: 09-01 04:15:52] {2466} INFO -  at 1323.7s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:15:52] {2282} INFO - iteration 48, current learner lgbm
    [flaml.automl.logger: 09-01 04:16:04] {2466} INFO -  at 1336.4s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:16:04] {2282} INFO - iteration 49, current learner lgbm
    [flaml.automl.logger: 09-01 04:17:19] {2466} INFO -  at 1411.2s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:17:19] {2282} INFO - iteration 50, current learner lgbm
    [flaml.automl.logger: 09-01 04:17:40] {2466} INFO -  at 1432.4s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:17:40] {2282} INFO - iteration 51, current learner lgbm
    [flaml.automl.logger: 09-01 04:20:40] {2466} INFO -  at 1612.1s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:20:40] {2282} INFO - iteration 52, current learner lgbm
    [flaml.automl.logger: 09-01 04:20:48] {2466} INFO -  at 1620.0s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:20:48] {2282} INFO - iteration 53, current learner lgbm
    [flaml.automl.logger: 09-01 04:21:52] {2466} INFO -  at 1684.4s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:21:52] {2282} INFO - iteration 54, current learner lgbm
    [flaml.automl.logger: 09-01 04:22:32] {2466} INFO -  at 1723.8s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:22:32] {2282} INFO - iteration 55, current learner lgbm
    [flaml.automl.logger: 09-01 04:23:01] {2466} INFO -  at 1752.6s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:23:01] {2282} INFO - iteration 56, current learner lgbm
    [flaml.automl.logger: 09-01 04:24:13] {2466} INFO -  at 1825.3s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:24:13] {2282} INFO - iteration 57, current learner lgbm
    [flaml.automl.logger: 09-01 04:24:36] {2466} INFO -  at 1847.7s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:24:36] {2282} INFO - iteration 58, current learner lgbm
    [flaml.automl.logger: 09-01 04:25:49] {2466} INFO -  at 1921.0s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:25:49] {2282} INFO - iteration 59, current learner lgbm
    [flaml.automl.logger: 09-01 04:26:43] {2466} INFO -  at 1975.4s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:26:43] {2282} INFO - iteration 60, current learner lgbm
    [flaml.automl.logger: 09-01 04:26:57] {2466} INFO -  at 1989.0s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:26:57] {2282} INFO - iteration 61, current learner lgbm
    [flaml.automl.logger: 09-01 04:28:55] {2466} INFO -  at 2106.7s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:28:55] {2282} INFO - iteration 62, current learner lgbm
    [flaml.automl.logger: 09-01 04:29:18] {2466} INFO -  at 2129.6s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:29:18] {2282} INFO - iteration 63, current learner lgbm
    [flaml.automl.logger: 09-01 04:30:16] {2466} INFO -  at 2187.9s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:30:16] {2282} INFO - iteration 64, current learner lgbm
    [flaml.automl.logger: 09-01 04:30:47] {2466} INFO -  at 2219.1s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:30:47] {2282} INFO - iteration 65, current learner lgbm
    [flaml.automl.logger: 09-01 04:31:13] {2466} INFO -  at 2245.2s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:31:13] {2282} INFO - iteration 66, current learner lgbm
    [flaml.automl.logger: 09-01 04:31:55] {2466} INFO -  at 2287.4s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:31:55] {2282} INFO - iteration 67, current learner lgbm
    [flaml.automl.logger: 09-01 04:34:00] {2466} INFO -  at 2411.7s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:34:00] {2282} INFO - iteration 68, current learner lgbm
    [flaml.automl.logger: 09-01 04:34:21] {2466} INFO -  at 2432.5s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:34:21] {2282} INFO - iteration 69, current learner lgbm
    [flaml.automl.logger: 09-01 04:34:58] {2466} INFO -  at 2469.7s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:34:58] {2282} INFO - iteration 70, current learner lgbm
    [flaml.automl.logger: 09-01 04:35:08] {2466} INFO -  at 2479.9s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:35:08] {2282} INFO - iteration 71, current learner lgbm
    [flaml.automl.logger: 09-01 04:35:38] {2466} INFO -  at 2509.7s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:35:38] {2282} INFO - iteration 72, current learner lgbm
    [flaml.automl.logger: 09-01 04:36:12] {2466} INFO -  at 2543.4s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:36:12] {2282} INFO - iteration 73, current learner lgbm
    [flaml.automl.logger: 09-01 04:36:34] {2466} INFO -  at 2565.8s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:36:34] {2282} INFO - iteration 74, current learner lgbm
    [flaml.automl.logger: 09-01 04:37:02] {2466} INFO -  at 2594.4s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:37:02] {2282} INFO - iteration 75, current learner lgbm
    [flaml.automl.logger: 09-01 04:37:15] {2466} INFO -  at 2606.6s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:37:15] {2282} INFO - iteration 76, current learner lgbm
    [flaml.automl.logger: 09-01 04:38:15] {2466} INFO -  at 2667.1s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:38:15] {2282} INFO - iteration 77, current learner lgbm
    [flaml.automl.logger: 09-01 04:38:40] {2466} INFO -  at 2691.5s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:38:40] {2282} INFO - iteration 78, current learner lgbm
    [flaml.automl.logger: 09-01 04:39:37] {2466} INFO -  at 2748.6s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:39:37] {2282} INFO - iteration 79, current learner lgbm
    [flaml.automl.logger: 09-01 04:39:57] {2466} INFO -  at 2768.6s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:39:57] {2282} INFO - iteration 80, current learner lgbm
    [flaml.automl.logger: 09-01 04:42:17] {2466} INFO -  at 2909.2s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:42:17] {2282} INFO - iteration 81, current learner lgbm
    [flaml.automl.logger: 09-01 04:43:10] {2466} INFO -  at 2961.8s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:43:10] {2282} INFO - iteration 82, current learner lgbm
    [flaml.automl.logger: 09-01 04:44:03] {2466} INFO -  at 3015.2s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:44:03] {2282} INFO - iteration 83, current learner lgbm
    [flaml.automl.logger: 09-01 04:44:50] {2466} INFO -  at 3062.1s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:44:50] {2282} INFO - iteration 84, current learner lgbm
    [flaml.automl.logger: 09-01 04:45:33] {2466} INFO -  at 3105.1s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:45:33] {2282} INFO - iteration 85, current learner lgbm
    [flaml.automl.logger: 09-01 04:46:06] {2466} INFO -  at 3137.8s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:46:06] {2282} INFO - iteration 86, current learner lgbm
    [flaml.automl.logger: 09-01 04:47:14] {2466} INFO -  at 3205.6s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:47:14] {2282} INFO - iteration 87, current learner lgbm
    [flaml.automl.logger: 09-01 04:47:38] {2466} INFO -  at 3229.9s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:47:38] {2282} INFO - iteration 88, current learner lgbm
    [flaml.automl.logger: 09-01 04:48:01] {2466} INFO -  at 3253.2s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:48:01] {2282} INFO - iteration 89, current learner lgbm
    [flaml.automl.logger: 09-01 04:48:39] {2466} INFO -  at 3290.7s,	estimator lgbm's best error=0.1697,	best estimator lgbm's best error=0.1697
    [flaml.automl.logger: 09-01 04:48:39] {2282} INFO - iteration 90, current learner lgbm
    [flaml.automl.logger: 09-01 04:49:42] {2466} INFO -  at 3353.9s,	estimator lgbm's best error=0.1696,	best estimator lgbm's best error=0.1696
    [flaml.automl.logger: 09-01 04:49:42] {2282} INFO - iteration 91, current learner lgbm
    [flaml.automl.logger: 09-01 04:50:05] {2466} INFO -  at 3376.7s,	estimator lgbm's best error=0.1696,	best estimator lgbm's best error=0.1696
    [flaml.automl.logger: 09-01 04:50:05] {2282} INFO - iteration 92, current learner lgbm
    [flaml.automl.logger: 09-01 04:51:11] {2466} INFO -  at 3442.7s,	estimator lgbm's best error=0.1696,	best estimator lgbm's best error=0.1696
    [flaml.automl.logger: 09-01 04:51:11] {2282} INFO - iteration 93, current learner lgbm
    [flaml.automl.logger: 09-01 04:51:41] {2466} INFO -  at 3473.0s,	estimator lgbm's best error=0.1696,	best estimator lgbm's best error=0.1696
    [flaml.automl.logger: 09-01 04:51:41] {2282} INFO - iteration 94, current learner lgbm
    [flaml.automl.logger: 09-01 04:51:57] {2466} INFO -  at 3488.7s,	estimator lgbm's best error=0.1696,	best estimator lgbm's best error=0.1696
    [flaml.automl.logger: 09-01 04:51:57] {2282} INFO - iteration 95, current learner lgbm
    [flaml.automl.logger: 09-01 04:53:28] {2466} INFO -  at 3580.4s,	estimator lgbm's best error=0.1696,	best estimator lgbm's best error=0.1696
    [flaml.automl.logger: 09-01 04:54:35] {2724} INFO - retrain lgbm for 67.0s
    [flaml.automl.logger: 09-01 04:54:35] {2727} INFO - retrained model: LGBMClassifier(colsample_bytree=np.float64(0.9507638075713548),
                   learning_rate=np.float64(0.07145698689510153), max_bin=511,
                   min_child_samples=23, n_estimators=1196, n_jobs=-1,
                   num_leaves=101, reg_alpha=np.float64(0.04824059831356815),
                   reg_lambda=np.float64(0.0012071539876428944),
                   scale_pos_weight=np.float64(14.108642353662274), verbose=-1)
    [flaml.automl.logger: 09-01 04:54:35] {2009} INFO - fit succeeded
    [flaml.automl.logger: 09-01 04:54:35] {2010} INFO - Time taken to find the best model: 3353.878039121628
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
    fitted_voter, y_pred_with_top_k_models = make_predict_with_top_k_models(
        X_train_final,
        y_train,
        X_test_final,
        # path_to_log_file="./log_flaml_tuning_01_09_25T0008.log", # 0.7511
        # path_to_log_file="./logs/log_flaml_tuning_020925_T001537.log",
        top_k=3,
        path_to_log_dir="./logs",
        predict_proba=False,
        threshold=0.5,
        n_jobs=-1,
    )
    return fitted_voter, y_pred_with_top_k_models


@app.cell
def _(mo):
    mo.md(
        r"""
    Start ...
    ("MODEL CONFIG: {'n_estimators': 4248, 'num_leaves': 126, 'min_child_samples': "
     "7, 'learning_rate': 0.112704414438969, 'log_max_bin': 6, 'colsample_bytree': "
     "0.662929128184499, 'reg_alpha': 0.040139570097457, 'reg_lambda': "
     "0.07221287260078, 'scale_pos_weight': 14.108642353662274, 'random_state': "
     "42, 'deterministic': True, 'force_col_wise': True, 'FLAML_sample_size': "
     '157758}')
    Get 1 models ...
    Start model fitting with val-score-best-config ...
    [LGBMClassifier] 0/1: 1042 / 21718
    PREDICT PROBA MODE: False
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # 0.7511 with path_to_log_file="./log_flaml_tuning_01_09_25T0008.log", # 0.7511

    Start ...
    ("MODEL CONFIG: {'n_estimators': 32767, 'num_leaves': 114, "
     "'min_child_samples': 40, 'learning_rate': 0.020656724959742002, "
     "'log_max_bin': 8, 'colsample_bytree': 0.47390240515872806, 'reg_alpha': "
     "0.769818669290351, 'reg_lambda': 0.12433362683346101, 'scale_pos_weight': "
     "14.108642353662274, 'FLAML_sample_size': 157758}")
    ("MODEL CONFIG: {'n_estimators': 11417, 'num_leaves': 193, "
     "'min_child_samples': 20, 'learning_rate': 0.024958544946010002, "
     "'log_max_bin': 8, 'colsample_bytree': 0.49732123717601606, 'reg_alpha': "
     "0.371681407627018, 'reg_lambda': 0.8324410895440431, 'scale_pos_weight': "
     "14.108642353662274, 'FLAML_sample_size': 157758}")
    ("MODEL CONFIG: {'n_estimators': 32767, 'num_leaves': 60, 'min_child_samples': "
     "14, 'learning_rate': 0.027295521181665002, 'log_max_bin': 8, "
     "'colsample_bytree': 0.623823161560474, 'reg_alpha': 0.518043245961315, "
     "'reg_lambda': 0.650612422516965, 'scale_pos_weight': 14.108642353662274, "
     "'FLAML_sample_size': 157758}")
    ("MODEL CONFIG: {'n_estimators': 15167, 'num_leaves': 205, "
     "'min_child_samples': 50, 'learning_rate': 0.009029194684408002, "
     "'log_max_bin': 9, 'colsample_bytree': 0.5716254981602741, 'reg_alpha': "
     "0.7139083971663831, 'reg_lambda': 0.012989543750033, 'scale_pos_weight': "
     "14.108642353662274, 'FLAML_sample_size': 157758}")
    ("MODEL CONFIG: {'n_estimators': 13573, 'num_leaves': 223, "
     "'min_child_samples': 32, 'learning_rate': 0.031442552811430005, "
     "'log_max_bin': 10, 'colsample_bytree': 0.510365976661978, 'reg_alpha': "
     "0.234474023622873, 'reg_lambda': 0.152831669957864, 'scale_pos_weight': "
     "14.108642353662274, 'FLAML_sample_size': 157758}")
    Get 5 models ...
    Start voter fitting ...
    [VotingClassifier] 0/1: 1139 / 21621
    [clf-58] 0/1: 1176 / 21584
    [clf-54] 0/1: 1125 / 21635
    [clf-56] 0/1: 1160 / 21600
    [clf-63] 0/1: 1154 / 21606
    [clf-53] 0/1: 1104 / 21656
    PREDICT PROBA MODE: False
    """
    )
    return


@app.cell
def _(pd, y_pred_with_top_k_models):
    poss_to_negs = pd.Series(y_pred_with_top_k_models).value_counts()
    poss_to_negs
    return


@app.cell
def _(X_test_final, save_submission, y_pred_with_top_k_models):
    save_submission(y_pred_with_top_k_models, X_test_final)
    return


@app.cell
def _(
    SEED,
    StratifiedKFold,
    TunedThresholdClassifierCV,
    X_train_final,
    fitted_voter,
    y_train,
):
    voter_with_tuned_threshold = TunedThresholdClassifierCV(
        fitted_voter,
        scoring="f1",
        response_method="predict_proba",
        thresholds=50,
        cv=StratifiedKFold(n_splits=3),
        n_jobs=1,
        random_state=SEED,
    ).fit(X_train_final, y_train)
    return (voter_with_tuned_threshold,)


@app.cell
def _(pd):
    pd.read_csv("./submission.csv", index_col=0)["prediction"].value_counts()
    return


@app.cell
def _(voter_with_tuned_threshold):
    voter_with_tuned_threshold.best_threshold_
    return


@app.cell
def _(X_test_final, pd, voter_with_tuned_threshold):
    pd.Series(voter_with_tuned_threshold.predict(X_test_final)).value_counts()
    return


@app.cell
def _(X_test_final, save_submission, voter_with_tuned_threshold):
    save_submission(voter_with_tuned_threshold.predict(X_test_final), X_test_final)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
