import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Разведочный анализ данных по треку "Контроль качества: автоматическое выявление поддельных товаров"
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import typing as t
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import shap
    import torch
    import os
    import itertools

    from tqdm import tqdm
    from PIL import Image
    from transformers import ViTFeatureExtractor, ViTModel
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    from category_encoders.cat_boost import CatBoostEncoder
    from flaml.automl.ml import sklearn_metric_loss_score
    return (
        ColumnTransformer,
        HistGradientBoostingClassifier,
        Image,
        Pipeline,
        ViTFeatureExtractor,
        ViTModel,
        mo,
        np,
        os,
        pd,
        plt,
        shap,
        sklearn_metric_loss_score,
        t,
        torch,
        tqdm,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Константы""")
    return


@app.cell
def _():
    SEED = 3535881
    TARGET_COL_NAME = "resolution"
    CLASS_WEIGHT_BALANCED = "balanced"
    EMPTY_LIST = []
    return CLASS_WEIGHT_BALANCED, SEED, TARGET_COL_NAME


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Вспомогательные функции""")
    return


@app.cell
def _(
    CLASS_WEIGHT_BALANCED,
    ColumnTransformer,
    HistGradientBoostingClassifier,
    Pipeline,
    SEED,
    TARGET_COL_NAME,
    pd,
    plt,
    shap,
    sklearn_metric_loss_score,
    t,
    train_test_split,
):
    def _get_col_names_wo_target_name(
        train: pd.DataFrame,
        use_only_numeric_cols: bool = True,
    ) -> list[str]:
        if use_only_numeric_cols:
            columns: list[str] = train.columns.tolist()
            columns.remove(TARGET_COL_NAME)

        return train[columns].select_dtypes(exclude=["object"]).columns.tolist()

    def _stratified_background_sample(
        train: pd.DataFrame,
        n_samples=100
    ) -> pd.DataFrame:
        columns: list[str] = _get_col_names_wo_target_name(train)
        X, y = train[columns], train[TARGET_COL_NAME]

        n_minority = sum(y == 1)
        n_majority = sum(y == 0)

        sample_majority = X[y == 0].sample(
            n=min(n_samples // 2, n_majority), 
            random_state=42
        )
        sample_minority = X[y == 1].sample(
            n=min(n_samples // 2, n_minority),
            random_state=42
        )

        return pd.concat([sample_majority, sample_minority])

    def get_shap_values(
        *,
        train: pd.DataFrame,
        model_params: t.Optional[dict] = None,
        seed: int = SEED,
        target_col_name: str = TARGET_COL_NAME,
        val_size: float = 0.25,
        metric: str = "f1",
        use_categoric_cols: bool = False,
        figsize: tuple[int, int] = (80, 180),
    ) -> None:
        numerical_cols: list[str] = _get_col_names_wo_target_name(train)
        X_train, y_train = train[numerical_cols], train[TARGET_COL_NAME]

        X_sub_train, X_val, y_sub_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            stratify=y_train,
            random_state=seed,
        )

        preprocessor = ColumnTransformer(
            transformers=[("num", "passthrough", numerical_cols)]
        )

        base_model_params = {
            "max_iter": 100,
            "learning_rate": 0.3,
            "max_depth": 5,
            "max_features": 0.8,
            "class_weight": CLASS_WEIGHT_BALANCED,
            "early_stopping": True,
        }

        if model_params is not None:
            base_model_params.update(model_params)

        classifier = HistGradientBoostingClassifier(**base_model_params)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_sub_train, y_sub_train)

        loss_score_train = sklearn_metric_loss_score(metric, pipeline.predict(X_sub_train), y_sub_train)
        loss_score_val = sklearn_metric_loss_score(metric, pipeline.predict(X_val), y_val)

        print(f"{metric.upper()}_train: {loss_score_train:.3g}")
        print(f"{metric.upper()}_val: {loss_score_val:.3g}")

        background_data = _stratified_background_sample(train, n_samples=100)

        explainer = shap.TreeExplainer(
            pipeline.named_steps["classifier"], 
            data=background_data,
            model_output="probability"
        )

        shap_values_val = explainer.shap_values(pipeline.named_steps["preprocessor"].transform(X_val))

        fig = plt.figure(figsize=figsize)
        plt.subplot(3, 1, 1)
        shap.summary_plot(
            shap_values_val, 
            X_val,
            show=False,
        )
        plt.title("Both Classes")

        fig = plt.figure(figsize=figsize)
        plt.subplot(3, 1, 2)
        neg_mask = y_val == 0
        shap.summary_plot(
            shap_values_val[neg_mask], 
            X_val[neg_mask],
            show=False,
        )
        plt.title("Majority Class")

        fig = plt.figure(figsize=figsize)
        plt.subplot(3, 1, 3)
        pos_mask = y_val == 1
        shap.summary_plot(
            shap_values_val[pos_mask], 
            X_val[pos_mask],
            show=False,
        )
        plt.title("Minority Class")

        plt.tight_layout()
        plt.show()
    return (get_shap_values,)


@app.cell
def _(Image, ViTFeatureExtractor, ViTModel, np, os, pd, torch, tqdm):
    def _get_image_paths(archive_path, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
        """Возвращает пути к изображениям и их числовые ID из имен файлов"""
        image_paths = []
        image_ids = []

        if os.path.isdir(archive_path):
            for root, _, files in os.walk(archive_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        # Извлекаем числовой ID из имени файла
                        file_name = os.path.splitext(file)[0]  # Убираем расширение
                        try:
                            image_id = int(file_name)  # Пробуем преобразовать в число
                            image_paths.append(os.path.join(root, file))
                            image_ids.append(image_id)
                        except ValueError:
                            print(f"Пропускаем файл {file}: имя не является числом")
                            continue
        return image_paths, image_ids

    def _process_images_batch(model, feature_extractor, image_paths, image_ids, batch_size=32, device='cuda'):
        """Обрабатывает изображения и сохраняет эмбеддинги с индексами"""

        model = model.to(device)
        model.eval()

        # Словарь для хранения эмбеддингов: {image_id: embedding}
        embeddings_dict = {}
        processed_ids = []

        # Обрабатываем пакетами
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i+batch_size]
            batch_ids = image_ids[i:i+batch_size]
            batch_images = []
            valid_ids = []

            # Загружаем изображения
            for path, img_id in zip(batch_paths, batch_ids):
                try:
                    image = Image.open(path).convert('RGB')
                    batch_images.append(image)
                    valid_ids.append(img_id)
                except Exception as e:
                    print(f"Skipping {path}: {e}")
                    continue

            if not batch_images:
                continue

            # Обрабатываем пакет
            try:
                inputs = feature_extractor(
                    images=batch_images, 
                    return_tensors="pt",
                    padding=True
                )
                pixel_values = inputs['pixel_values'].to(device)

                with torch.no_grad():
                    outputs = model(pixel_values=pixel_values)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                    # Сохраняем эмбеддинги с соответствующими ID
                    for img_id, embedding in zip(valid_ids, embeddings):
                        embeddings_dict[img_id] = embedding
                        processed_ids.append(img_id)

            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue

        return embeddings_dict, processed_ids

    def _save_embeddings_with_ids(embeddings_dict, output_path, hugging_face_model_path: str):
        """Сохраняет эмбеддинги с сохранением индексов"""

        # Создаем матрицу эмбеддингов и массив индексов
        ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[img_id] for img_id in ids])

        # Сохраняем в npz файл
        np.savez(output_path, embeddings=embeddings, ids=ids, hugging_face_model_path=hugging_face_model_path)

        # Также сохраняем в текстовом формате для читаемости
        with open(output_path.replace('.npz', '_mapping.txt'), 'w') as f:
            for img_id in ids:
                f.write(f"{img_id}\n")

        return embeddings, ids

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

    def get_embeddings(
        path_to_images: str,
        output_path: str,
        hugging_face_model_path: str = "google/vit-base-patch16-224-in21k",
        batch_size: int = 16,
    ):
        # Загрузка модели и feature extractor
        feature_extractor = ViTFeatureExtractor.from_pretrained(hugging_face_model_path)
        model = ViTModel.from_pretrained(hugging_face_model_path)

        # Получаем пути к изображениям
        image_paths, image_ids = _get_image_paths(path_to_images)
        print(f"Found {len(image_paths)} images")

        if not image_paths:
            print("No images found!")
            return

           # Обрабатываем изображения
        embeddings_dict, processed_ids = _process_images_batch(
            model=model,
            feature_extractor=feature_extractor,
            image_paths=image_paths,
            image_ids=image_ids,
            batch_size=16,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        embeddings_array, ids_array = _save_embeddings_with_ids(embeddings_dict, output_path, hugging_face_model_path)

        print(f"Processed {len(processed_ids)} images")
        print(f"Embeddings shape: {embeddings_array.shape}")
        print(f"Unique IDs: {len(ids_array)}")
        print(f"Example IDs: {ids_array[:5]}")
    return get_embeddings, load_embeddings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Загрузка данных""")
    return


@app.cell
def _(pd):
    train = pd.read_csv("./ml_ozon_сounterfeit_data/ml_ozon_сounterfeit_train.csv", index_col=0)
    test = pd.read_csv("./ml_ozon_сounterfeit_data/ml_ozon_сounterfeit_test.csv", index_col=0)
    return (train,)


@app.cell
def _(train):
    train.shape
    return


@app.cell
def _(train):
    train.dtypes
    return


@app.cell
def _(train):
    train
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Важность признаков по Шепли""")
    return


@app.cell
def _(CLASS_WEIGHT_BALANCED, get_shap_values, train):
    get_shap_values(
        model_params=dict(
            max_iter=1_000,
            learning_rate=0.01,
            max_depth=5,
            max_features=0.3,
            class_weight=CLASS_WEIGHT_BALANCED,
            early_stopping=True,
        ),
        train=train,
        use_categoric_cols=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Подготовка эмбедингов по изображением""")
    return


@app.cell
def _(get_embeddings):
    get_embeddings(
        path_to_images = "./ml_ozon_сounterfeit_train_images/",
        output_path="image_embeddings_vit-base-patch16-224-in21k.npz",
        batch_size = 32,
    )
    return


@app.cell
def _(load_embeddings):
    image_ebmds = load_embeddings("./image_embeddings_vit-base-patch16-224-in21k.npz")
    return


@app.cell
def _(embds, pd):
    pd.DataFrame(embds["embeddings"], index=embds["ids"]).iloc[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Анализ только контрафактных товаров""")
    return


@app.cell
def _(train):
    train["rating_1_count"].median()
    return


@app.cell
def _(train):
    train[train["resolution"] == 1].describe()
    return


@app.cell
def _(train):
    train[train["resolution"] == 0].describe()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
