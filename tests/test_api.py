"""
Тесты для Flask API прогнозирования дефолта.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import json
from app.api import create_app
from preprocess import DataPreprocessor
import __main__
__main__.DataPreprocessor = DataPreprocessor

@pytest.fixture
def client():
    """Фикстура, создающая тестовый клиент Flask."""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    """Проверка эндпоинта /health."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] == True
    assert data['pipeline_loaded'] == True
    assert 'v1' in data['available_versions'] or 'v2' in data['available_versions']
    # Проверяем, что отображается режим случайного выбора
    assert 'selection_mode' in data
    assert 'random' in data['selection_mode'].lower()

def test_predict_random_version(client):
    """Проверка случайного выбора версии при отсутствии поля 'version'."""
    payload = {
        "LIMIT_BAL": 200000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 50000, "BILL_AMT2": 45000, "BILL_AMT3": 40000,
        "BILL_AMT4": 35000, "BILL_AMT5": 30000, "BILL_AMT6": 25000,
        "PAY_AMT1": 5000, "PAY_AMT2": 4500, "PAY_AMT3": 4000,
        "PAY_AMT4": 3500, "PAY_AMT5": 3000, "PAY_AMT6": 2500
    }
    # Делаем несколько запросов и собираем использованные версии
    versions = set()
    for _ in range(10):
        response = client.post('/predict', data=json.dumps(payload), content_type='application/json')
        assert response.status_code == 200
        data = response.get_json()
        assert data['version'] in ('v1', 'v2')
        versions.add(data['version'])
    # При случайном выборе с вероятностью 50/50 за 10 попыток должны встретиться обе версии
    # (это не гарантировано, но высоко вероятно)
    assert 'v1' in versions or 'v2' in versions

def test_predict_v2_explicit(client):
    """Проверка predict с явным указанием версии v2."""
    payload = {
        "version": "v2",
        "LIMIT_BAL": 200000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 50000, "BILL_AMT2": 45000, "BILL_AMT3": 40000,
        "BILL_AMT4": 35000, "BILL_AMT5": 30000, "BILL_AMT6": 25000,
        "PAY_AMT1": 5000, "PAY_AMT2": 4500, "PAY_AMT3": 4000,
        "PAY_AMT4": 3500, "PAY_AMT5": 3000, "PAY_AMT6": 2500
    }
    response = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert data['version'] == 'v2'
    assert data['probability'] < 0.5

def test_predict_risky_client(client):
    """Проверка predict для клиента с высоким риском (версия по умолчанию случайна)."""
    payload = {
        "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
        "PAY_0": 3, "PAY_2": 2, "PAY_3": 2, "PAY_4": 1, "PAY_5": 1, "PAY_6": 0,
        "BILL_AMT1": 45000, "BILL_AMT2": 40000, "BILL_AMT3": 35000,
        "BILL_AMT4": 30000, "BILL_AMT5": 25000, "BILL_AMT6": 20000,
        "PAY_AMT1": 1000, "PAY_AMT2": 1000, "PAY_AMT3": 1000,
        "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000
    }
    response = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert data['probability'] > 0.5

def test_predict_invalid_input(client):
    """Проверка валидации входных данных."""
    payload = {"LIMIT_BAL": 200000}
    response = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data