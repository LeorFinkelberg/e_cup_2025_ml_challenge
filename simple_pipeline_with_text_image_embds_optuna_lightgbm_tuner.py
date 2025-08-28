import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ```bash
    $ uv run optuna-dashboard sqlite:///DB_NAME.db
    ```
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
    import typing as t

    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, roc_curve, auc
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.calibration import calibration_curve

    from flaml.automl.automl import AutoML
    from flaml import tune

    import lightgbm as lgb
    from lightgbm import early_stopping, log_evaluation

    import optuna
    import optuna_integration
    import optuna_integration.lightgbm as optuna_lgb
    return (
        AutoML,
        PCA,
        QuantileTransformer,
        StratifiedKFold,
        auc,
        calibration_curve,
        ce,
        early_stopping,
        f1_score,
        lgb,
        log_evaluation,
        mo,
        np,
        optuna,
        optuna_lgb,
        pd,
        plt,
        roc_curve,
        t,
        train_test_split,
    )


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
def _(lgb):
    def save_best_automl(automl, path_to_model: str) -> None:
        automl.model.estimator.booster_.save_model(path_to_model)

    def load_best_automl(path_to_model: str) -> lgb.Booster:
        return lgb.Booster(model_file=path_to_model)
    return load_best_automl, save_best_automl


@app.cell
def _(
    SEED,
    StratifiedKFold,
    auc,
    calibration_curve,
    f1_score,
    np,
    optuna,
    optuna_lgb,
    pd,
    plt,
    roc_curve,
    t,
):
    from optuna.study import Study
    from optuna.trial import FrozenTrial

    class OptunaLightGBMTunerWrapper:
        def __init__(
            self,
            *,
            params: dict,
            X_train: pd.DataFrame,
            y_train: np.typing.ArrayLike,
            X_val: pd.DataFrame,
            y_val: np.typing.ArrayLike,
            time_budget: int = 3600,
            study_name: t.Optional[str] = None,
            valid_names: list[str] = ["validation"],
            n_splits: int = 3,
            cv: bool = False,
            callbacks: t.Optional[list[t.Callable[[Study, FrozenTrial], None]]] = None,
            show_progress_bar: bool = False, 
            model_dir: t.Optional[str] = None, 
            optuna_seed: int = SEED,
        ) -> None:
            self.dtrain = optuna_lgb.Dataset(X_train, label=y_train)
            self.dval = optuna_lgb.Dataset(X_val, label=y_val, reference=self.dtrain)
        
            self.n_splits = n_splits
            self.cv = cv
            self.time_budget = time_budget
            self.params = params
            self.valid_names = valid_names
            self.callbacks = callbacks
            self.study_name = study_name
            _storage_url = f"sqlite:///{study_name}.db" if (study_name is not None) else None
            self.study = optuna.create_study(
                study_name=study_name,
                storage=_storage_url,
                direction="minimize",
                load_if_exists=False
            )
            self.show_progress_bar = show_progress_bar
            self.model_dir = model_dir
            self.optuna_seed = optuna_seed,
        
            self._tuner = None
            self.best_params = None
            self.best_iteration = None

        """
        def f1_loss(self, y_pred, data, threshold: float = 0.5):
            y_true = data.get_label()
            y_pred_binary = (y_pred > threshold).astype(int)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0, average="macro")
        
            return "1 - f1", 1 - f1, False
        """
        
        def f1_metric(self, y_pred, data, threshold: float = 0.5):
            y_true = data.get_label()
            y_pred_binary = (y_pred > threshold).astype(int)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0, average="macro")
            return "f1", f1, True

        def train(self) -> None:
            if self.cv:
                self._tuner = optuna_lgb.LightGBMTunerCV(
                    params=self.params,
                    train_set=self.dtrain,
                    folds=StratifiedKFold(n_splits=self.n_splits),
                    time_budget=self.time_budget,
                    callbacks=self.callbacks,
                    study=self.study,
                    return_cvbooster=True,
                    show_progress_bar=self.show_progress_bar,
                    model_dir=self.model_dir,
                    optuna_seed=self.optuna_seed,
                )
            else:
                self._tuner = optuna_lgb.train(
                    params=self.params,
                    train_set=self.dtrain,
                    valid_sets=[self.dval],
                    valid_names=self.valid_names,
                    time_budget=self.time_budget,
                    feval=self.f1_metric,
                    callbacks=self.callbacks,
                    study=self.study,
                    show_progress_bar=self.show_progress_bar,
                    model_dir=self.model_dir,
                    optuna_seed=self.optuna_seed,
                )

            self.best_params: dict = self._tuner.params
            self.best_iteration: int = self._tuner.best_iteration

        def predict(self, X_test: pd.DataFrame) -> np.array:
            return np.rint(self._tuner.predict(X_test, num_iteration=self.best_iteration))
        
        def plot_calibration_threshold_analysis(self, X_val, y_val):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    
            # 1. Кривая калибровки
            y_pred_proba = self._tuner.predict(X_val)
            prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
    
            ax1.plot([0, 1], [0, 1], "k:", label="Идеально")
            ax1.plot(prob_pred, prob_true, "s-", label="Калиброванная модель")
            ax1.set_xlabel("Предсказанная вероятность")
            ax1.set_ylabel("Истинная доля положительных")
            ax1.set_title("Кривая калибровки")
            ax1.legend()
            ax1.grid(True)
    
            # 2. Зависимость F1 от порога
            thresholds = np.linspace(0.1, 0.9, 50)
            f1_scores = []
    
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1_scores.append(f1_score(y_val, y_pred, average="macro"))
        
            ax2.plot(thresholds, f1_scores, 'b-', linewidth=2)
            # ax2.axvline(model.optimal_threshold, color='red', linestyle='--', 
                        # label=f'Оптимальный порог: {model.optimal_threshold:.3f}')
            ax2.set_xlabel('Порог')
            ax2.set_ylabel('F1-score')
            ax2.set_title('Зависимость F1-score от порога')
            ax2.legend()
            ax2.grid(True)
    
            # 3. Кривая ROC-AUC откалиброванной модели
            fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
            auc_raw = auc(fpr, tpr)
        
            ax3.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC кривая (AUC = {auc_raw:.2f})')
            ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный классификатор')
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('Кривая ROC-AUC откалиброванной модели')
            ax3.legend(loc="lower right")
            ax3.grid(True)
        
            plt.tight_layout()
            plt.show()
            
    return (OptunaLightGBMTunerWrapper,)


@app.cell
def _(
    SEED,
    calibration_curve,
    early_stopping,
    f1_score,
    lgb,
    log_evaluation,
    np,
    optuna,
    plt,
    t,
):
    class FLAMLOptunaRefiner:
        def __init__(
            self,
            automl_model,
            X_train,
            y_train,
            X_val,
            y_val,
            n_startup_trials: int = 10,
            stopping_rounds: int = 50,
            random_state: int = SEED,
        ):
            self.automl = automl_model
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.best_params = None
            self.refined_model = None
            self.n_startup_trials = n_startup_trials
            self.stopping_rounds = stopping_rounds
            self.random_state = random_state
        
        def get_flaml_params(self):
            """Извлекает и преобразует параметры из FLAML"""
            flaml_params = self.automl.model.estimator.get_params()
        
            # Преобразуем в формат LightGBM
            lgbm_params = {
                "n_estimators": flaml_params.get("n_estimators", 1_000),
                "num_leaves": flaml_params.get("num_leaves", 31),
                "min_child_samples": flaml_params.get("min_child_samples", 20),
                "learning_rate": flaml_params.get("learning_rate", 0.1),
                "colsample_bytree": flaml_params.get("colsample_bytree", 0.8),
                "reg_alpha": flaml_params.get("reg_alpha", 0.0),
                "reg_lambda": flaml_params.get("reg_lambda", 0.0),
                "feature_fraction": flaml_params.get("feature_fraction", 1.0),
                "bagging_fraction": flaml_params.get("bagging_fraction", 1.0),
                "bagging_freq": flaml_params.get("bagging_freq", 0),
            }
            return lgbm_params
    
        def objective(self, trial):
            """Objective функция для тонкой настройки вокруг FLAML параметров"""
            base_params = self.get_flaml_params()
            print(f"Base params from FLAML: {base_params}")
        
            # Тонкая настройка вокруг лучших параметров FLAML
            tuning_params = {
                'num_leaves': trial.suggest_int(
                    'num_leaves',
                    max(7, int(base_params['num_leaves'] * 0.3)),  
                    min(255, int(base_params['num_leaves'] * 1.7))
                ),
                'min_child_samples': trial.suggest_int(
                    'min_child_samples',
                    max(1, int(base_params['min_child_samples'] * 0.3)),
                    min(200, int(base_params['min_child_samples'] * 2.0))
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    max(0.005, base_params['learning_rate'] * 0.3),
                    min(0.2, base_params['learning_rate'] * 3.0), 
                    log=True
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    max(0.3, base_params["colsample_bytree"] - 0.3),
                    min(1.0, base_params["colsample_bytree"] + 0.1),
                ),
                'feature_fraction': trial.suggest_float(
                    'feature_fraction',
                    max(0.4, base_params['feature_fraction'] - 0.3),
                    min(1.0, base_params['feature_fraction'] + 0.1)
                ),
                'bagging_fraction': trial.suggest_float(
                    'bagging_fraction',
                    max(0.4, base_params['bagging_fraction'] - 0.3),
                    min(1.0, base_params['bagging_fraction'] + 0.1)
                ),
                'bagging_freq': trial.suggest_int(
                    'bagging_freq',
                    max(1, int(base_params['bagging_freq'] - 3)),
                    min(5, int(base_params['bagging_freq'] + 3)),
                ),
                'reg_alpha': trial.suggest_float(
                    'reg_alpha',
                    max(1e-8, base_params['reg_alpha'] * 0.1),
                    min(5.0, base_params['reg_alpha'] * 10.0), 
                    log=True
                ),
                'reg_lambda': trial.suggest_float(
                    'reg_lambda', 
                    max(1e-8, base_params['reg_lambda'] * 0.1),
                    min(5.0, base_params['reg_lambda'] * 10.0),
                    log=True
                )
            }
        
            # Обучаем модель
            train_data = lgb.Dataset(self.X_train, label=self.y_train)
            val_data = lgb.Dataset(self.X_val, label=self.y_val)

            final_params = base_params.copy()
            final_params.update(tuning_params)
            final_params.update({
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "random_state": self.random_state,
                "is_unbalance": True,
            })
            model = lgb.train(
                final_params,
                train_data,
                valid_sets=[val_data],
                valid_names=['validation'],
                callbacks=[
                    early_stopping(stopping_rounds=self.stopping_rounds, verbose=True),
                    log_evaluation(period=50),
                ],
            )

            print(
                f"Trial {trial.number}: Best iteration={model.best_iteration} ",
                f"Best score={model.best_score['validation']['binary_logloss']:.6f}"
            )

            y_pred = model.predict(self.X_val)
            f1 = f1_score(self.y_val, (y_pred > 0.5).astype(int))
        
            # return model.best_score['validation']['binary_logloss']
            return 1 - f1
    
        def refine(self, study_name: t.Optional[str] = None, time_budget=1800, n_trials=50):
            """Запуск тонкой настройки"""
            storage_url = f"sqlite:///{study_name}.db" if (study_name is not None) else None

            if study_name and storage_url:
                try:
                    optuna.delete_study(study_name=study_name, storage=storage_url)
                except:
                    pass
            
            # Создаем study
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction='minimize',
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=self.n_startup_trials,
                    seed=self.random_state,
                ),
                load_if_exists=False
            )
        
            # Запускаем оптимизацию
            study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=time_budget,
                show_progress_bar=True
            )

            base_params = self.get_flaml_params()
            self.best_params = base_params.copy()
            self.best_params.update(study.best_params)  # Добавляем tuning параметры
            self.best_params.update({
                'objective': 'binary',
                'metric': 'binary_logloss', 
                'verbosity': -1,
                'random_state': self.random_state,
                'is_unbalance': True,
            })
            print("Final best_params keys:", list(self.best_params.keys()))
        
            # Обучаем финальную модель
            self._train_final_model()
        
            return self.best_params, study
    
        def _train_final_model(self):
            """Обучает финальную модель с лучшими параметрами"""
            train_data = lgb.Dataset(self.X_train, label=self.y_train)
            val_data = lgb.Dataset(self.X_val, label=self.y_val)
        
            self.refined_model = lgb.train(
                self.best_params,
                train_data,
                num_boost_round=10_000,
                valid_sets=[val_data],
                callbacks=[
                    early_stopping(stopping_rounds=self.stopping_rounds, verbose=True),
                    log_evaluation(period=50),
                ],
            )
    
        def predict(self, X):
            """Предсказания от донастроенной модели"""
            if self.refined_model is None:
                raise ValueError("Сначала запустите refine()")
            return self.refined_model.predict(X)
    
        def compare_metrics(self, X_val, y_val):
            """Сравнение метрик до и после донастройки"""
            # Предсказания FLAML
            y_pred_flaml = self.automl.predict_proba(X_val)[:, 1]
            flaml_f1 = f1_score(y_val, (y_pred_flaml > 0.5).astype(int), average="macro")
        
            # Предсказания refined модели
            y_pred_refined = self.predict(X_val)
            refined_f1 = f1_score(y_val, (y_pred_refined > 0.5).astype(int), average="macro")
        
            return {
                'flaml_f1': flaml_f1,
                'refined_f1': refined_f1,
                'improvement': refined_f1 - flaml_f1
            }

        def plot_calibration_threshold_analysis(self, X_val, y_val):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    
            # 1. Кривая калибровки
            y_pred_proba = self.refined_model.predict(X_val)
            prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
    
            ax1.plot([0, 1], [0, 1], "k:", label="Идеально")
            ax1.plot(prob_pred, prob_true, "s-", label="Калиброванная модель")
            ax1.set_xlabel("Предсказанная вероятность")
            ax1.set_ylabel("Истинная доля положительных")
            ax1.set_title("Кривая калибровки")
            ax1.legend()
            ax1.grid(True)
    
            # 2. Зависимость F1 от порога
            thresholds = np.linspace(0.1, 0.9, 50)
            f1_scores = []
    
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1_scores.append(f1_score(y_val, y_pred, average="macro"))
        
            ax2.plot(thresholds, f1_scores, 'b-', linewidth=2)
            # ax2.axvline(model.optimal_threshold, color='red', linestyle='--', 
                        # label=f'Оптимальный порог: {model.optimal_threshold:.3f}')
            ax2.set_xlabel('Порог')
            ax2.set_ylabel('F1-score')
            ax2.set_title('Зависимость F1-score от порога')
            ax2.legend()
            ax2.grid(True)
    return (FLAMLOptunaRefiner,)


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
def _(SEED, X_train_final, train_test_split, y_train):
    X_sub_train_final, X_val_final, y_sub_train, y_val = train_test_split(
        X_train_final, 
        y_train,
        stratify=y_train,
        test_size=0.20,
        random_state=SEED,
    )
    return X_sub_train_final, X_val_final, y_sub_train, y_val


@app.cell
def _(SEED, X_sub_train_final, train_test_split, y_sub_train):
    X_train_flaml, X_train_optuna, y_train_flaml, y_train_optuna = train_test_split(
        X_sub_train_final,
        y_sub_train,
        stratify=y_sub_train,
        test_size=0.20,
        random_state=SEED,
    )
    return X_train_flaml, X_train_optuna, y_train_flaml, y_train_optuna


@app.cell
def _(X_sub_train_final):
    X_sub_train_final.shape
    return


@app.cell
def _(X_test_final):
    X_test_final.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## _Проверка гипотез_""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Запуск FLAML + Optuna_""")
    return


@app.cell
def _(AutoML, SEED, TASK, X_train_flaml, np, y_train, y_train_flaml):
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

    automl = AutoML()
    automl.fit(
        X_train_flaml,
        y_train_flaml,
        task=TASK,
        time_budget=5400,
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
def _(automl):
    automl.best_config
    return


@app.cell
def _(automl, save_best_automl):
    save_best_automl(automl, path_to_model = "./flaml_after_optuna_tune.txt")
    return


@app.cell
def _(load_best_automl):
    best_model = load_best_automl(path_to_model = "./flaml_after_optuna_tune.txt")
    return


@app.cell
def _(automl, pd):
    best_booster = automl.best_model_for_estimator(estimator_name="lgbm")
    pd.Series(best_booster.feature_importances_, index=best_booster.feature_names_in_).sort_values(ascending=False)
    return


@app.cell
def _(
    FLAMLOptunaRefiner,
    X_train_optuna,
    X_val_final,
    automl,
    y_train_optuna,
    y_val,
):
    refiner = FLAMLOptunaRefiner(automl, X_train_optuna, y_train_optuna, X_val_final, y_val)
    best_params, study = refiner.refine(study_name="optuna_tuning_flaml", time_budget=3600, n_trials=500)

    # Сравниваем результаты
    metrics_comparison = refiner.compare_metrics(X_val_final, y_val)
    print(f"FLAML F1: {metrics_comparison['flaml_f1']:.4f}")
    print(f"Refined F1: {metrics_comparison['refined_f1']:.4f}")
    print(f"Improvement: {metrics_comparison['improvement']:.4f}")
    return (refiner,)


@app.cell
def _(X_val_final, refiner, y_val):
    refiner.plot_calibration_threshold_analysis(X_val_final, y_val)
    return


@app.cell
def _(X_test_final, np, pd, refiner):
    pd.DataFrame({
        "id": X_test_final.index, 
        "prediction": np.rint(refiner.predict(X_test_final)),
    }).to_csv("./submission.csv", index=False)
    return


@app.cell
def _(pd):
    pd.read_csv("./submission.csv", index_col=0)["prediction"].value_counts()
    return


@app.cell
def _(pd):
    pd.read_csv("./submission_f1=0.7333.csv", index_col=0)["prediction"].value_counts()
    return


@app.cell
def _(mo):
    mo.md(r"""### _FLAML ZeroShotAutoML_""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### _Запуск тюнера LightGBMTuner_""")
    return


@app.cell
def _(
    OptunaLightGBMTunerWrapper,
    X_sub_train_final,
    X_val_final,
    early_stopping,
    log_evaluation,
    y_sub_train_final,
    y_val_final,
):
    tuner = OptunaLightGBMTunerWrapper(
        X_train=X_sub_train_final,
        y_train=y_sub_train_final,
        X_val=X_val_final,
        y_val=y_val_final,
        time_budget=7200,
        params={
            "objective": "binary",
            # "metric": "binary_logloss",
            "is_unbalance": True,
            "boosting_type": "gbdt",
            "verbosity": -1,
        },
        cv=False,
        callbacks=[
            early_stopping(50),
            log_evaluation(100),
        ],
        study_name="lightgbm_tuning",
        show_progress_bar=False,
    )

    tuner.train()
    return (tuner,)


@app.cell
def _(tuner):
    tuner.best_params
    return


@app.cell
def _(tuner):
    tuner.best_iteration
    return


@app.cell
def _(X_val_final, tuner, y_val_final):
    tuner.plot_calibration_threshold_analysis(X_val_final, y_val_final)
    return


if __name__ == "__main__":
    app.run()
