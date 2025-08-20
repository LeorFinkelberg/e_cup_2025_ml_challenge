class TextEmbedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device=None):
        """
        Инициализация модели для создания эмбеддингов
        
        Args:
            model_name: название модели с HuggingFace
            device: 'cuda' для GPU или 'cpu' (автоопределение если None)
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Модель загружена: {model_name}")
        print(f"Размерность эмбеддингов: {self.embedding_dim}")
    
    def create_embeddings(self, texts, batch_size=32, show_progress=True):
        """
        Создание эмбеддингов для списка текстов
        
        Args:
            texts: список текстов для обработки
            batch_size: размер батча для обработки
            show_progress: показывать прогресс-бар
            
        Returns:
            numpy array с эмбеддингами формы (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
            
        # Обрабатываем тексты батчами
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Нормализуем для косинусного сходства
        )
        
        return embeddings
    
    def add_embeddings_to_dataframe(self, df, text_column, output_prefix='embedding_', batch_size=32):
        """
        Добавляет эмбеддинги к pandas DataFrame
        
        Args:
            df: исходный DataFrame
            text_column: название колонки с текстом
            output_prefix: префикс для названий колонок с эмбеддингами
            batch_size: размер батча
            
        Returns:
            DataFrame с добавленными колонками эмбеддингов
        """
        if text_column not in df.columns:
            raise ValueError(f"Колонка '{text_column}' не найдена в DataFrame")
        
        # Получаем тексты
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # Создаем эмбеддинги
        embeddings = self.create_embeddings(texts, batch_size=batch_size)
        
        # Создаем колонки для эмбеддингов
        embedding_columns = [f'{output_prefix}{i}' for i in range(self.embedding_dim)]
        
        # Добавляем эмбеддинги в DataFrame
        embedding_df = pd.DataFrame(embeddings, columns=embedding_columns, index=df.index)
        result_df = pd.concat([df, embedding_df], axis=1)
        
        return result_df

if __name__ == "__main__":
    train = pd.read_csv("./ml_ozon_сounterfeit_data/ml_ozon_сounterfeit_train.csv", index_col=0)
    test = pd.read_csv("./ml_ozon_сounterfeit_data/ml_ozon_сounterfeit_test.csv", index_col=0)

    embedder = TextEmbedder()
    train_with_embeddings = embedder.add_embeddings_to_dataframe(
        df=train,
        text_column="description",
        output_prefix="desc_embed_",
        batch_size=32
    )
    train_with_embeddings = embedder.add_embeddings_to_dataframe(
        df=train_with_embeddings,
        text_column="name_rus",
        output_prefix="name_rus_embed_",
        batch_size=32
    )
    train_with_embeddings.to_csv("train_with_text_embeddings_all-MiniLM-L6-v2.csv", index=True)

    test_with_embeddings = embedder.add_embeddings_to_dataframe(
        df=test,
        text_column="description",
        output_prefix="desc_embed_",
        batch_size=32
    )
    test_with_embeddings = embedder.add_embeddings_to_dataframe(
        df=test_with_embeddings,
        text_column="name_rus",
        output_prefix="name_rus_embed_",
        batch_size=32
    )
    test_with_embeddings.to_csv("test_with_text_embeddings_all-MiniLM-L6-v2.csv", index=True)

