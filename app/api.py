"""
Flask API для сервиса прогнозирования дефолта по кредитным картам

Эндпоинты:
- POST /predict - прогноз дефолта для одного клиента (поддержка версий v1/v2)
- GET /health - проверка работоспособности сервиса
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pythonjsonlogger import jsonlogger
import sys
import os
import random

# Добавляем путь к src для импорта DataPreprocessor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import DataPreprocessor  # Нужно для десериализации пайплайна

import __main__
__main__.DataPreprocessor = DataPreprocessor

# Настройка JSON-логирования
logger = logging.getLogger(__name__)
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

app = Flask(__name__)

# Глобальные переменные
full_pipeline = None
models = {}                 # словарь {version: model}

def load_artifacts():
    """
    Загрузка пайплайна и моделей при старте сервиса.
    """
    global full_pipeline, models

    logger.info("Загрузка артефактов...")

    try:
        # Загрузка пайплайна
        full_pipeline = joblib.load('models/full_pipeline.joblib')
        logger.info("Пайплайн загружен")

        # Загрузка моделей v1 и v2
        for version in ['v1', 'v2']:
            if version == 'v1':
                filename = 'models/model.joblib'
            else:
                filename = 'models/model_2.joblib'

            try:
                model_artifact = joblib.load(filename)
                models[version] = model_artifact['model']
                logger.info(f"Модель {version} загружена из {filename}")
            except FileNotFoundError:
                logger.warning(f"Модель {version} не найдена по пути {filename}, пропускаем")

        if not models:
            raise FileNotFoundError("Не найдено ни одной модели (model.joblib или model_2.joblib)")


    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
        raise


# Валидация входных данных
REQUIRED_FEATURES = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

def validate_input(data):
    """
    Валидация входных данных.
    Проверяем:
    1. Все ли признаки присутствуют
    2. Нет ли пропущенных значений
    3. Числовые ли значения
    """
    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        return False, f"Отсутствуют поля: {missing}"

    for field in REQUIRED_FEATURES:
        value = data[field]
        if value is None:
            return False, f"Поле {field} не может быть None"
        try:
            float(value)
        except (ValueError, TypeError):
            return False, f"Поле {field} должно быть числом"

    return True, "OK"


def preprocess_input(data):
    """
    Предобработка одного клиента через пайплайн.
    """
    df = pd.DataFrame([data])
    X_scaled = full_pipeline.transform(df)
    return X_scaled


# Эндпоинты
@app.route('/health', methods=['GET'])
def health_check():
    """
    Проверка работоспособности сервиса.
    """
    logger.info("Health check")

    return jsonify({
        'status': 'healthy',
        'available_versions': list(models.keys()),
        'selection_mode': 'random (50/50) when version not specified',
        'model_loaded': len(models) > 0,
        'pipeline_loaded': full_pipeline is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Предсказание дефолта для клиента.
    """
    request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')

    logger.info("Predict запрос", extra={
        'request_id': request_id,
        'remote_addr': request.remote_addr
    })

    try:
        # 1. Получаем JSON
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Тело запроса должно быть JSON',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Извлечение версии модели (с поддержкой A/B-теста)
        requested_version = data.pop('version', None)
        if requested_version and requested_version in models:
            current_model = models[requested_version]
            current_version = requested_version
        else:
            # Если версия не указана, случайным образом выбираем v1 или v2 (50/50)
            current_version = random.choice(list(models.keys()))
            current_model = models[current_version]

        # 2. Валидация
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            logger.warning(f"Валидация не пройдена: {error_msg}",
                          extra={'request_id': request_id})
            return jsonify({
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }), 400

        # 3. Предобработка
        X_scaled = preprocess_input(data)

        # 4. Предсказание выбранной моделью
        prediction = int(current_model.predict(X_scaled)[0])
        probability = float(current_model.predict_proba(X_scaled)[0, 1])

        # 5. Ответ
        response = {
            'prediction': prediction,
            'probability': round(probability, 4),
            'version': current_version,
            'timestamp': datetime.now().isoformat()
        }

        logger.info("Predict результат", extra={
            'request_id': request_id,
            'prediction': prediction,
            'probability': probability,
            'model_version': current_version
        })

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}", extra={'request_id': request_id}, exc_info=True)
        return jsonify({
            'error': f'Внутренняя ошибка: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500


def create_app():
    """
    Создаёт и настраивает Flask-приложение.
    Загружает артефакты и возвращает готовое приложение для тестов или продакшена.
    """
    load_artifacts()
    return app


if __name__ == '__main__':
    app = create_app()
    logger.info("Запуск Flask сервера на порту 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)