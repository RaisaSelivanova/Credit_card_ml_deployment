"""
Модуль предобработки данных для прогнозирования дефолта по кредитным картам

Основан на выводах EDA:
1. Аномалии: EDUCATION (0,5,6) и MARRIAGE (0) требуют обработки
2. PAY_* (-2) - значимый позитивный сигнал, сохранён как отдельная категория
3. Удалены признаки с мультиколлинеарностью
5. Созданы 6 новых признаков с подтверждённой теснотой связи с таргетом
6. StandardScaler для нормализации (огромный разброс сумм: 0 - 1.6M)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Класс для загрузки, очистки и предобработки данных кредитного скоринга.
    
    Пайплайн:
    1. Загрузка CSV
    2. Обработка аномалий (EDUCATION, MARRIAGE)
    3. Создание дополнительных признаков
    4. Удаление слабых/дублирующих признаков
    5. Масштабирование (StandardScaler)
    6. Разделение на train/test (стратифицированное, есть дисбаланс классов)
    """
    
    def __init__(self, data_path='C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/data/UCI_Credit_Card.csv'):
       
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.feature_columns = None  # Сохраняем порядок признаков для API
        self.full_pipeline = None    # Полный пайплайн для сохранения
        
    def load_data(self):
        """
        Загрузка данных из CSV-файла.

        """
        logger.info(f"Загрузка данных из {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info(f"Данные загружены. Размер: {df.shape}")
        return df
    
    def clean_anomalies(self, df):
        """
        Очистка аномальных значений на основе выводов EDA:
        
        - EDUCATION: значения 0, 5, 6 объединяем с категорией 4 ('others')
            
        - MARRIAGE: значение 0 отсутствует в документации, объединяем с категорией 3 
        ('others')
         """
        logger.info("Очистка аномалий...")
        
        # EDUCATION:
        education_anomalies = df['EDUCATION'].isin([0, 5, 6]).sum()
        df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
        logger.info(f"  EDUCATION: {education_anomalies} аномалий (0,5,6) заменены на 4 (others)")
        
        # MARRIAGE: 
        marriage_anomalies = (df['MARRIAGE'] == 0).sum()
        df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
        logger.info(f"  MARRIAGE: {marriage_anomalies} аномалий (0) заменены на 3 (others)")
        
        return df
    
    def create_features(self, df):
        """
        Создание дополнительных признаков на основе EDA.
        
        Выводы EDA, обосновывающие создание признаков:
        
        1. PAY_DELAY_COUNT (корр. +0.398) -
           Клиенты с просрочками в нескольких месяцах - высокий риск.
           Агрегирует информацию из всех 6 PAY_* признаков.
           
        2. PAY_MAX (корр. +0.331) -
           Максимальная задержка показывает худшее поведение клиента.
           Важнее среднего, т.к. единичная сильная просрочка - красный флаг.
           
        3. PAY_TREND (корр. +0.129) - ухудшение статуса
           Разница между свежим статусом (PAY_0) и старым (PAY_6).
           Положительное значение = ситуация ухудшается.
           
        4. PAY_AMT_MEAN (корр. -0.102) - средний платёж
           Используем среднее вместо 6 отдельных (избегаем шума).
           
        5. CREDIT_USAGE (корр. +0.086) - использование лимита
                  
        6. BILL_AMT1 - оставлен как единственный представитель BILL_AMT*
           Остальные BILL_AMT2-6 удалены из-за мультиколлинеарности (>0.85).
            
        """
        logger.info("Создание дополнительных признаков...")
        
        # Исходные PAY_* колонки
        pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        
        # 1. Количество месяцев с просрочкой (>0)
       
        df['PAY_DELAY_COUNT'] = (df[pay_cols] > 0).sum(axis=1)
        logger.info(f"  + PAY_DELAY_COUNT (корр. +0.398)")
        
        # 2. Максимальная задержка за полгода
       
        df['PAY_MAX'] = df[pay_cols].max(axis=1)
        logger.info(f"  + PAY_MAX (корр. +0.331)")
        
        # 3. Тренд ухудшения: PAY_0 - PAY_6
       
        df['PAY_TREND'] = df['PAY_0'] - df['PAY_6']
        logger.info(f"  + PAY_TREND (корр. +0.129)")
        
        # 4. Средний платёж за полгода
       
        pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                        'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        df['PAY_AMT_MEAN'] = df[pay_amt_cols].mean(axis=1)
        logger.info(f"  + PAY_AMT_MEAN (корр. -0.102)")
        
        # 5. Использование кредитного лимита
     
        df['CREDIT_USAGE'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
        logger.info(f"  + CREDIT_USAGE (корр. +0.086)")
        
        return df
    
    def select_features(self, df):
        """
        Отбор финального набора признаков для модели.
        
        Критерии отбора (на основе EDA):
        - Корреляция с таргетом > 0.05 (или бизнес-значимость)
        - Отсутствие мультиколлинеарности (>0.85)
        - Отсутствие дублирования с новыми признаками
        
        ОСТАВЛЯЕМ (13 признаков):
        Исходные (7):
          - LIMIT_BAL, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
        Новые (6):
          - PAY_DELAY_COUNT, PAY_MAX, PAY_TREND
          - PAY_AMT_MEAN, CREDIT_USAGE
          - BILL_AMT1 (единственный оставленный из BILL_AMT*)
        
        """
        logger.info("Отбор признаков...")
        
        # Целевая переменная
        target_column = 'default.payment.next.month'
        y = df[target_column]
        
        # Финальный список признаков (13)
        selected_features = [
            # Кредитный лимит
            'LIMIT_BAL',
            # Статусы платежей (исходные, 6 признаков)
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            # Статусы платежей (агрегированные, 3 признака)
            'PAY_DELAY_COUNT', 'PAY_MAX', 'PAY_TREND',
            # Суммы (агрегированные, 2 признака)
            'PAY_AMT_MEAN', 'CREDIT_USAGE',
            # Счета (только свежий, 1 признак)
            'BILL_AMT1'
        ]
        
        X = df[selected_features].copy()
        
        # Сохраняем список и порядок признаков для API
        self.feature_columns = selected_features
        
        logger.info(f"  Отобрано {len(selected_features)} признаков")
        logger.info(f"  Признаки: {selected_features}")
        logger.info(f"  Удалены: ID, SEX, EDUCATION, MARRIAGE, AGE, BILL_AMT2-6, PAY_AMT1-6")
        
        return X, y
    
    def scale_features(self, X):
        """
        Масштабирование признаков с помощью StandardScaler.
        Причина: признаки имеют разный масштаб
        
        StandardScaler приводит всё к единому масштабу, что критично для ряда алгоритмов
        """
        logger.info("Масштабирование признаков (StandardScaler)...")
        X_scaled = self.scaler.fit_transform(X)
        logger.info(f"  Размер после масштабирования: {X_scaled.shape}")
        return X_scaled
    
    def build_pipeline(self):
        """
        Создание единого пайплайна для сохранения.
        
        Пайплайн включает все шаги предобработки:
        1. Очистка аномалий
        2. Создание признаков
        3. Отбор признаков
        4. Масштабирование
     
        """
        logger.info("Создание полного пайплайна...")
        
        # Создаём трансформеры из функций
        cleaner = FunctionTransformer(
            self.clean_anomalies, 
            validate=False
        )
        feature_creator = FunctionTransformer(
            self.create_features, 
            validate=False
        )
        
        # Отбор признаков
        selector = FunctionTransformer(
            self._select_features_only, 
            validate=False
        )
        self.full_pipeline = Pipeline([
            ('cleaner', cleaner),
            ('feature_creator', feature_creator),
            ('selector', selector),
            ('scaler', self.scaler)
        ])
        
        logger.info("Пайплайн создан!")
        return self.full_pipeline
    
    def _select_features_only(self, df):
        """
        Отбор признаков без возврата целевой переменной.
        
        """
        return df[self.feature_columns]
        
        
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Стратифицированное разделение на train/test.
        Стратификация критична из-за дисбаланса классов
        """
        logger.info(f"Разделение на train/test (test_size={test_size}, stratify=y)...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"  Train: {X_train.shape}, дефолтов: {y_train.mean():.2%}")
        logger.info(f"  Test:  {X_test.shape}, дефолтов: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def save_artifacts(self, path='C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/models/'):
        """
        Сохранение артефактов:
        1. full_pipeline.joblib - пайплайн (для API)
        2. preprocessor_info.joblib - метаинформация (для совместимости)
        
        """
        os.makedirs(path, exist_ok=True)
        
        # Сохраняем пайплайн 
        pipeline_path = os.path.join(path, 'full_pipeline.joblib')
        joblib.dump(self.full_pipeline, pipeline_path)
        logger.info(f"Пайплайн сохранён: {pipeline_path}")
        
        # Сохраняем метаинформацию (для отладки)
        info_path = os.path.join(path, 'preprocessor_info.joblib')
        info = {
            'feature_columns': self.feature_columns,
            'features_count': len(self.feature_columns)
        }
        joblib.dump(info, info_path)
        logger.info(f"Метаинформация сохранена: {info_path}")
    
    def run_pipeline(self):
        """
        Запуск пайплайна предобработки.
        
        Шаги:
        1. Загрузка данных
        2. Очистка аномалий 
        3. Создание признаков 
        4. Отбор признаков 
        5. Масштабирование (StandardScaler)
        6. Разделение train/test (стратифицированное)
        7. Построение полного пайплайна
        8. Сохранение артефактов
  
        """
       
        logger.info("\nЗАПУСК ПАЙПЛАЙНА ПРЕДОБРАБОТКИ\n")
     
        
        # Шаг 1: Загрузка
        df = self.load_data()
        
        # Шаг 2: Очистка аномалий
        df = self.clean_anomalies(df)
        
        # Шаг 3: Создание признаков
        df = self.create_features(df)
        
        # Шаг 4: Отбор признаков
        X, y = self.select_features(df)
        
        # Шаг 5: Масштабирование
        X_scaled = self.scale_features(X)
        
        # Шаг 6: Разделение
        X_train, X_test, y_train, y_test = self.split_data(X_scaled, y)
        
        # Шаг 7: Построение полного пайплайна
        self.build_pipeline()
        
        # Шаг 8: Сохранение артефактов
        os.makedirs('C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/data/processed', exist_ok=True)
        np.save('C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/data/processed/X_train.npy', X_train)
        np.save('C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/data/processed/X_test.npy', X_test)
        np.save('C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/data/processed/y_train.npy', y_train.values)
        np.save('C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/data/processed/y_test.npy', y_test.values)
        logger.info("Обработанные данные сохранены в data/processed/")
        
        self.save_artifacts('C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/models/')
        
        logger.info("\nПАЙПЛАЙН УСПЕШНО ЗАВЕРШЁН\n")
        
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.run_pipeline()