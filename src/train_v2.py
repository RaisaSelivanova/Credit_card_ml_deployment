"""
Модуль обучения модели прогнозирования дефолта по кредитным картам

Создадим вторую версию модели, в которой изменим гиперпараметры относительно исходной
Все остальное неизменно
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
import joblib
import os
import json
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Класс для обучения и оценки модели кредитного скоринга.
    
    Пайплайн:
    1. Загрузка обработанных данных
    2. Создание модели RandomForest
    3. Обучение на train
    4. Оценка на test
    5. Сохранение модели и метрик
    """
    
    def __init__(self, model_params=None):
        """
        Инициализация тренера модели.
        
        """
        self.model = None
        self.metrics = {}
        self.feature_importance = None
        
        # Параметры по умолчанию 
        self.default_params = {
            'n_estimators': 150,        # для второй версии модели увеличим кол-во деревьев со 100 до 150
            'max_depth': 12,            # сделаем глубже деревья (было в первой версии 10)
            'min_samples_split': 5,     # меньше образцов (было в v1 10)
            'min_samples_leaf': 2,      # меньше (в первой версии было 5)
            'class_weight': 'balanced',  # Компенсация дисбаланса 
            'random_state': 42,         # для воспроизводимости
            'n_jobs': -1                # Все ядра процессора
        }
        
        self.model_params = model_params if model_params else self.default_params
        
    def load_data(self):
        """
        Загрузка предобработанных данных.
        """
        logger.info("Загрузка обработанных данных...")
        
        X_train = np.load('C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/data/processed/X_train.npy')
        X_test = np.load('C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/data/processed/X_test.npy')
        y_train = np.load('C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/data/processed/y_train.npy')
        y_test = np.load('C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/data/processed/y_test.npy')
        
        logger.info(f"Train: {X_train.shape}, дефолтов: {y_train.mean():.2%}")
        logger.info(f"Test:  {X_test.shape}, дефолтов: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def create_model(self):
        """
        Создание модели RandomForestClassifier.
       
        """
        logger.info("Создание модели RandomForestClassifier...")
        logger.info(f"Параметры: {self.model_params}")
        
        self.model = RandomForestClassifier(**self.model_params)
        
        return self.model
    
    def train(self, X_train, y_train):
        """
        Обучение модели.
        
        """
        logger.info("Обучение модели...")
        self.model.fit(X_train, y_train)
        logger.info("Модель обучена!")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Оценка модели на тестовой выборке.
        
        Рассчитываемые метрики:
        - F1-score (основная) - баланс precision и recall
        - Precision - доля правильно предсказанных дефолтов
        - Recall - доля найденных дефолтов
        - ROC-AUC - качество ранжирования
        - Confusion Matrix - матрица ошибок
        
        Почему F1-score основная, а не Accuracy:
        Accuracy при дисбалансе 78/22 бессмысленна:
        Модель, всегда говорящая "нет дефолта", даст Accuracy 78%,
        но не найдёт ни одного реального дефолта.
        F1 учитывает оба типа ошибок
        
        
        """
        logger.info("Оценка модели на тестовой выборке...")
        
        # Предсказания
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Расчёт метрик
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Сохраняем метрики
        self.metrics = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Вывод результатов
        
        logger.info("\nМЕТРИКИ МОДЕЛИ\n")
        
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info(f"Precision: {precision:.4f} (доля верных предсказаний дефолта)")
        logger.info(f"Recall:    {recall:.4f} (доля найденных дефолтов)")
        logger.info(f"ROC-AUC:   {roc_auc:.4f} (качество ранжирования)")
        logger.info(f"\nМатрица ошибок:")
        logger.info(f"  TN: {tn} (верно предсказано 'нет дефолта')")
        logger.info(f"  FP: {fp} (ложная тревога)")
        logger.info(f"  FN: {fn} (пропущенный дефолт)")
        logger.info(f"  TP: {tp} (верно предсказанный дефолт)")
        logger.info(f"\nClassification Report:")
        logger.info(f"\n{self.metrics['classification_report']}")
        
        return self.metrics
    
    def get_feature_importance(self, feature_names=None):
        """
        Извлечение важности признаков из модели.
        
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train().")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Сортируем по убыванию важности
        indices = np.argsort(importances)[::-1]
        
       
        logger.info("\nВАЖНОСТЬ ПРИЗНАКОВ\n")
        
        for i in range(min(20, len(feature_names))):
            idx = indices[i]
            logger.info(f"{i+1:2}. {feature_names[idx]:<25} {importances[idx]:.4f}")
        
        self.feature_importance = {
            'features': feature_names,
            'importances': importances.tolist()
        }
        
        return self.feature_importance
    
    def save_model(self, path='C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/models/model_2.joblib'):
        """
        Сохранение модели и метрик.
        
        Сохраняем:
        - model: обученный RandomForest
        - model_params: параметры модели
        - metrics: метрики качества
        - feature_importance: важность признаков
        - timestamp: время создания
        - version: версия модели (v2)

        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_artifact = {
            'model': self.model,
            'model_params': self.model_params,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now().isoformat(),
            'version': 'v2'
        }
        
        joblib.dump(model_artifact, path)
        logger.info(f"\nМодель сохранена: {path}")
        logger.info(f"Версия: v2")
        logger.info(f"Время: {datetime.now().isoformat()}")
    
    def save_metrics_json(self, path='C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/models/metrics_2.json'):
        """
        Сохранение метрик в JSON для отчётов.
   
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Убираем текстовый classification_report для JSON
        json_metrics = {
            'f1_score': self.metrics['f1_score'],
            'precision': self.metrics['precision'],
            'recall': self.metrics['recall'],
            'roc_auc': self.metrics['roc_auc'],
            'confusion_matrix': self.metrics['confusion_matrix'],
            'model_params': self.model_params,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(json_metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Метрики сохранены: {path}")
    
    def run_pipeline(self):
        """
        Запуск полного пайплайна обучения.
        
        Шаги:
        1. Загрузка данных
        2. Создание модели
        3. Обучение
        4. Оценка
        5. Анализ важности признаков
        6. Сохранение артефактов
        """
       
        logger.info("\nЗАПУСК ПАЙПЛАЙНА ОБУЧЕНИЯ\n")
              
        # Шаг 1: Загрузка данных
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Шаг 2: Создание модели
        self.create_model()
        
        # Шаг 3: Обучение
        self.train(X_train, y_train)
        
        # Шаг 4: Оценка
        self.evaluate(X_test, y_test)
        
        # Шаг 5: Важность признаков
        # Загружаем названия признаков из препроцессора
        preprocessor_info = joblib.load('C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/models/preprocessor_info.joblib')
        feature_names = preprocessor_info['feature_columns']
        self.get_feature_importance(feature_names)
        
        # Шаг 6: Сохранение
        self.save_model('C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/models/model_2.joblib')
        self.save_metrics_json('C:/Users/U_M1P8G/Desktop/IDE/Внедрение моделей ML/Сессионный проект/credit_card_ml_deployment/models/metrics_2.json')
        
        logger.info("\nПАЙПЛАЙН ОБУЧЕНИЯ УСПЕШНО ЗАВЕРШЁН\n")
        
        
        return self.model, self.metrics


if __name__ == "__main__":
    trainer = ModelTrainer()
    model, metrics = trainer.run_pipeline()