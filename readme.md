# Прогнозирование дефолта по кредитным картам

Сервис машинного обучения для оценки вероятности дефолта клиента на основе кредитной истории.  
Реализован полный цикл: от анализа данных и обучения модели до контейнеризации и A/B-тестирования.

## Версионность модели
Сервис поддерживает две версии модели, что позволяет проводить A/B-тестирование:
- **v1** (базовая): RandomForest, 100 деревьев, max_depth=10
- **v2** (модель для сравнения): RandomForest, 150 деревьев, max_depth=12

Версию можно указать в запросе (`"version": "v2"`). Если версия не указана, сервис случайным образом (с вероятностью 50/50) выбирает одну из загруженных моделей (v1 или v2), что автоматически обеспечивает разделение трафика для A/B-теста.
Подробнее о сравнении версий и плане тестирования в AB_testing.md

## Стек
- Python 3.9, scikit-learn, Flask, Docker, Docker Compose, Nginx

## Структура проекта
- `data/` - исходные и обработанные данные
- `notebooks/` - разведочный анализ данных и Feature Engineering
- `src/` - предобработка (`preprocess.py`) и обучение модели в двух версиях (`train.py`, `train_v2.py`)
- `app/` - Flask API (`api.py`)
- `models/` - сохранённые пайплайн и модели (v1, v2)
- `docker/` - Dockerfile
- `nginx/` - конфигурация для Docker Compose
- `tests/` - автоматические тесты API
- `screenshots/` - демонстрация работы
Дополнительная документация:
- Архитектура сервиса Architecture.md
- План A/B-тестирования AB_testing.md

## Локальный запуск
1. Клонируйте репозиторий: в терминале (bash) введите команды
   
   git clone https://github.com/raisas/credit-card-ml-deployment.git
   cd credit-card-ml-deployment

2. Создайте и активируйте виртуальное окружение:

   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate

3. Установите зависимости:
  
   pip install -r requirements.txt
4. Подготовьте данные и обучите модели:

   python src/preprocess.py
   python src/train.py
   python src/train_v2.py
5. Запустите сервис:

   python app/api.py

Запуск в Docker: 
  
   docker build -t credit-default-predictor:v1 -f docker/Dockerfile .
   docker run -d -p 5000:5000 --name credit-api credit-default-predictor:v1

Запуск через Docker Compose (с Nginx)
   
   docker-compose up -d
После этого API доступен на порту 80 (без указания порта).

### Docker Hub
Образ доступен по адресу: https://hub.docker.com/r/raisas/credit-default-predictor

## API
### GET /health
Проверка работоспособности:
  powershell:
  Invoke-RestMethod -Uri http://127.0.0.1:5000/health
Ответ (в терминале):

available_versions : {v1, v2}
model_loaded       : True
pipeline_loaded    : True
selection_mode     : random (50/50) when version not specified
status             : healthy
timestamp          : 2026-06-14T12:01:59.031816

или:
  curl.exe http://127.0.0.1:5000/health
Ответ (JSON):  
{"available_versions":["v1","v2"],"model_loaded":true,"pipeline_loaded":true,"selection_mode":"random (50/50) when version not specified","status":"healthy","timestamp":"2026-06-14T11:44:05.333302"}


### POST /predict
Предсказание дефолта. Тело запроса - JSON с признаками клиента. Все признаки обязательны. Можно указать версию модели (поле version).

Пример запроса (PowerShell, Invoke-RestMethod):

powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{
  "LIMIT_BAL":200000,"SEX":1,"EDUCATION":2,"MARRIAGE":1,"AGE":35,
  "PAY_0":0,"PAY_2":0,"PAY_3":0,"PAY_4":0,"PAY_5":0,"PAY_6":0,
  "BILL_AMT1":50000,"BILL_AMT2":45000,"BILL_AMT3":40000,
  "BILL_AMT4":35000,"BILL_AMT5":30000,"BILL_AMT6":25000,
  "PAY_AMT1":5000,"PAY_AMT2":4500,"PAY_AMT3":4000,
  "PAY_AMT4":3500,"PAY_AMT5":3000,"PAY_AMT6":2500,
  "version":"v2"
}'


Пример запроса curl: 
curl.exe -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":200000,"SEX":1,"EDUCATION":2,"MARRIAGE":1,"AGE":35,"PAY_0":0,"PAY_2":0,"PAY_3":0,"PAY_4":0,"PAY_5":0,"PAY_6":0,"BILL_AMT1":50000,"BILL_AMT2":45000,"BILL_AMT3":40000,"BILL_AMT4":35000,"BILL_AMT5":30000,"BILL_AMT6":25000,"PAY_AMT1":5000,"PAY_AMT2":4500,"PAY_AMT3":4000,"PAY_AMT4":3500,"PAY_AMT5":3000,"PAY_AMT6":2500,"version":"v2"}'

Ответ json:

{
  "prediction": 0,
  "probability": 0.1182,
  "version": "v2",
  "timestamp": "2026-06-13T15:01:50"
}

### Логи API (пример):

PS C:\Users\U_M1P8G\Desktop\IDE\Внедрение моделей ML\Сессионный проект\credit_card_ml_deployment> python app/api.py 
{"asctime": "2026-06-14 11:43:18,747", "name": "__main__", "levelname": "INFO", "message": "\u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u0430\u0440\u0442\u0435\u0444\u0430\u043a\u0442\u043e\u0432..."}
2026-06-14 11:43:18,747 - INFO - Загрузка артефактов...
{"asctime": "2026-06-14 11:43:18,749", "name": "__main__", "levelname": "INFO", "message": "\u041f\u0430\u0439\u043f\u043b\u0430\u0439\u043d \u0437\u0430\u0433\u0440\u0443\u0436\u0435\u043d"}
2026-06-14 11:43:18,749 - INFO - Пайплайн загружен
{"asctime": "2026-06-14 11:43:18,925", "name": "__main__", "levelname": "INFO", "message": "\u041c\u043e\u0434\u0435\u043b\u044c v1 \u0437\u0430\u0433\u0440\u0443\u0436\u0435\u043d\u0430 \u0438\u0437 models/model.joblib"}
2026-06-14 11:43:18,925 - INFO - Модель v1 загружена из models/model.joblib
{"asctime": "2026-06-14 11:43:18,999", "name": "__main__", "levelname": "INFO", "message": "\u041c\u043e\u0434\u0435\u043b\u044c v2 \u0437\u0430\u0433\u0440\u0443\u0436\u0435\u043d\u0430 \u0438\u0437 models/model_2.joblib"}
2026-06-14 11:43:18,999 - INFO - Модель v2 загружена из models/model_2.joblib
{"asctime": "2026-06-14 11:43:18,999", "name": "__main__", "levelname": "INFO", "message": "\u0417\u0430\u043f\u0443\u0441\u043a Flask \u0441\u0435\u0440\u0432\u0435\u0440\u0430 \u043d\u0430 \u043f\u043e\u0440\u0442\u0443 5000..."}
2026-06-14 11:43:18,999 - INFO - Запуск Flask сервера на порту 5000...
 * Serving Flask app 'api'
 * Debug mode: off
2026-06-14 11:43:19,017 - INFO - WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.0.104:5000
2026-06-14 11:43:19,017 - INFO - Press CTRL+C to quit
{"asctime": "2026-06-14 11:44:05,333", "name": "__main__", "levelname": "INFO", "message": "Health check"}
2026-06-14 11:44:05,333 - INFO - Health check
2026-06-14 11:44:05,333 - INFO - 127.0.0.1 - - [14/Jun/2026 11:44:05] "GET /health HTTP/1.1" 200 -
{"asctime": "2026-06-14 11:44:14,250", "name": "__main__", "levelname": "INFO", "message": "Health check"}
2026-06-14 11:44:14,250 - INFO - Health check
2026-06-14 11:44:14,251 - INFO - 127.0.0.1 - - [14/Jun/2026 11:44:14] "GET /health HTTP/1.1" 200 -
{"asctime": "2026-06-14 11:49:27,860", "name": "__main__", "levelname": "INFO", "message": "Predict \u0437\u0430\u043f\u0440\u043e\u0441", "request_id": "20260614114927860586", "remote_addr": "127.0.0.1"}
2026-06-14 11:49:27,860 - INFO - Predict запрос
2026-06-14 11:49:27,860 - INFO - Очистка аномалий...
2026-06-14 11:49:27,860 - INFO -   EDUCATION: 0 аномалий (0,5,6) заменены на 4 (others)
2026-06-14 11:49:27,869 - INFO -   MARRIAGE: 0 аномалий (0) заменены на 3 (others)
2026-06-14 11:49:27,869 - INFO - Создание дополнительных признаков...
2026-06-14 11:49:27,871 - INFO -   + PAY_DELAY_COUNT (корр. +0.398)
2026-06-14 11:49:27,871 - INFO -   + PAY_MAX (корр. +0.331)
2026-06-14 11:49:27,872 - INFO -   + PAY_TREND (корр. +0.129)
2026-06-14 11:49:27,874 - INFO -   + PAY_AMT_MEAN (корр. -0.102)
2026-06-14 11:49:27,874 - INFO -   + CREDIT_USAGE (корр. +0.086)
{"asctime": "2026-06-14 11:49:27,951", "name": "__main__", "levelname": "INFO", "message": "Predict \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442", "request_id": "20260614114927860586", "prediction": 0, "probability": 0.11821119704144045, "model_version": "v2"}

### Демонстрация работы
Скриншоты с примерами находятся в папке screenshots:
 - screenshots/health.jpg
 - screenshots/пример предсказания с явным выбором версий.jpg
 - screenshots/пример предсказания со случайным выбором версий.jpg
 - screenshots/Логи API_*

### Метрики модели 
V1:
F1-score: 0.54
Precision: 0.50
Recall: 0.59
ROC-AUC: 0.78

V2:
F1-score: 0.54
Precision: 0.51
Recall: 0.56
ROC-AUC: 0.77


#### Бизнес-метрики
Помимо технических метрик, для бизнеса важны:

- ожидаемые финансовые потери (Expected Loss):
Сумма потенциальных потерь от пропущенных дефолтов (по клиентам, ошибочно классифицированных как «не дефолт» (false negatives)).

- доля одобренных заявок при фиксированном уровне риска
Показывает, насколько модель позволяет увеличить объём кредитования без роста дефолтов.
Рассчитывается как доля клиентов с вероятностью дефолта ниже заданного порога при условии сохранения текущего уровня дефолтов в портфеле.