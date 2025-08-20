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


SEED = 3535881
TARGET_COL_NAME = "resolution"
CLASS_WEIGHT_BALANCED = "balanced"
EMPTY_LIST = []


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

if __name__ == "__main__":
    get_embeddings(
        path_to_images = "./ml_ozon_сounterfeit_train_images/",
        output_path="image_embeddings_vit-base-patch16-224-in21k.npz",
        batch_size = 32,
    )

