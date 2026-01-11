# Network Intrusion Detection System

Система обнаружения сетевых вторжений на основе машинного обучения, обученная на датасете CIC-IDS-2017. Этот проект реализует полный ML-пайплайн: от обработки сырых данных до обучения моделей, симуляции, **анализа реального трафика в реальном времени** и **интерактивного веб-интерфейса для мониторинга и офлайн-анализа**.

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20RF-orange)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Data Pipeline](#data-pipeline)
* [Model Training](#model-training)
* [Results](#results)
* [Testing & Simulation](#testing--simulation)
* [Real-Time Traffic Analysis](#real-time-traffic-analysis)
* [Web Interface](#web-interface)
* [Usage](#usage)
* [References](#references)

---

## Overview

Этот проект разрабатывает систему бинарной классификации для обнаружения вредоносного сетевого трафика. Система различает нормальный (benign) трафик и различные типы кибератак, включая DDoS, PortScan, Brute Force и другие.

### Key Features

* **Автоматизированный data pipeline**: загрузка сырых CSV, очистка, препроцессинг и разбиение на train/val/test
* **Несколько архитектур моделей**: Random Forest, XGBoost, LightGBM, нейронные сети
* **Обучение с ускорением на GPU**: поддержка CUDA для XGBoost, LightGBM и PyTorch
* **Ансамблевые методы**: взвешенное soft voting по лучшим моделям
* **Полноценный EDA**: автоматическая генерация визуализаций и audit-отчётов
* **Пайплайн симуляции**: real-time replay флоу с сбором метрик и визуализацией
* **Анализ трафика в реальном времени**: живой захват сетевых пакетов и классификация
* **Интерактивный веб-интерфейс**: дашборд для мониторинга в реальном времени и офлайн-анализа датасета
* **Сквозное тестирование**: unit-тесты и E2E-тесты для всех компонентов пайплайна

---

## Dataset

### CIC-IDS-2017

Датасет [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) создан Canadian Institute for Cybersecurity. Он содержит размеченные сетевые потоки (network flows), собранные за 5 дней, включая как нормальный трафик, так и различные типы атак.

| Property     | Value                       |
| ------------ | --------------------------- |
| Source       | University of New Brunswick |
| Duration     | 5 days (Monday-Friday)      |
| Total Flows  | 3,119,345                   |
| Features     | 79 (after preprocessing)    |
| Attack Types | 14                          |

### Attack Distribution

| Class                      | Count     | Percentage |
| -------------------------- | --------- | ---------- |
| BENIGN                     | 2,273,097 | 72.87%     |
| DoS Hulk                   | 231,073   | 7.41%      |
| PortScan                   | 158,930   | 5.09%      |
| DDoS                       | 128,027   | 4.10%      |
| DoS GoldenEye              | 10,293    | 0.33%      |
| FTP-Patator                | 7,938     | 0.25%      |
| SSH-Patator                | 5,897     | 0.19%      |
| DoS Slowloris              | 5,796     | 0.19%      |
| DoS Slowhttptest           | 5,499     | 0.18%      |
| Bot                        | 1,966     | 0.06%      |
| Web Attack - Brute Force   | 1,507     | 0.05%      |
| Web Attack - XSS           | 652       | 0.02%      |
| Infiltration               | 36        | <0.01%     |
| Web Attack - SQL Injection | 21        | <0.01%     |
| Heartbleed                 | 11        | <0.01%     |

### Class Distribution Visualization

![Class Distribution](reports/figures/01_class_distribution.png)

### Distribution by Day

![Day Distribution](reports/figures/02_day_distribution.png)

### Class Imbalance (Log Scale)

![Class Imbalance](reports/figures/03_class_imbalance_log.png)

---

## Project Structure

```
TraficAnalysis/
├── artifacts/                      # Обученные препроцессоры и схемы
│   ├── feature_schema.json         # Имена фичей и статистики
│   ├── label_mapping.json          # Кодировки меток классов
│   └── preprocessor.joblib         # Обученный scaler (RobustScaler)
│
├── configs/                        # Конфигурационные файлы
│   ├── data_pipeline.yaml          # Параметры обработки данных
│   ├── model_configs.yaml          # Гиперпараметры моделей
│   └── simulation.yaml             # Настройки симуляции
│
├── data/
│   ├── raw/                        # Оригинальные CSV файлы
│   │   └── CICIDS-2017/TrafficLabelling/
│   ├── interim/                    # Промежуточная обработка
│   │   ├── bronze_combined.parquet
│   │   └── manifest.json
│   └── processed/                  # Финальные обработанные данные
│       ├── processed_data.parquet
│       └── splits/
│           ├── train.parquet
│           ├── val.parquet
│           └── test.parquet
│
├── notebooks/
│   └── CIC_IDS_2017_model_training.ipynb
│
├── realtime/                       # Модуль анализа в реальном времени
│   ├── __init__.py
│   ├── analyzer.py                 # Инференс ML модели
│   ├── capture.py                  # Захват пакетов (scapy)
│   ├── config.py                   # Управление конфигурацией
│   ├── feature_extractor.py        # Извлечение фичей потоков (flow features)
│   ├── flow_aggregator.py          # Агрегация пакетов в потоки
│   ├── pipeline.py                 # Основной processing pipeline
│   ├── utils.py                    # Утилиты и helpers
│   └── web_interface.py            # Интеграция Flask/FastAPI
│
├── realtime_testing_artifacts/     # Результаты real-time сессий
│   └── YYYYMMDD_HHMMSS/
│       ├── flows.csv               # Все проанализированные потоки
│       ├── attacks.csv             # Только обнаруженные атаки
│       ├── stats.json              # Статистика сессии
│       ├── session.log             # Лог атак
│       ├── traffic_analysis.png    # Графики трафика
│       └── attack_analysis.png     # Разбор атак
│
├── reports/
│   ├── figures/                    # Визуализации EDA
│   └── simulation/                 # Результаты симуляции
│
├── scripts/
│   ├── run_data_pipeline.py        # Обработка данных
│   ├── run_simulation.py           # Офлайн симуляция
│   ├── run_realtime.py             # Real-time analyzer
│   ├── local_attack_test.py        # Симуляция атак для тестирования
│   ├── run_e2e_test.py             # End-to-end тесты
│   └── train_models.py             # Обучение моделей
│
├── src/                            # Исходный код
│   ├── data/                       # Обработка данных
│   ├── models/                     # Реализации моделей
│   ├── inference/                  # Пайплайн инференса
│   ├── simulation/                 # Компоненты симуляции
│   ├── database/                   # Хранилище базы данных
│   └── visualization/              # Отчёты и графики
│
├── tests/                          # Unit tests
│
├── training_artifacts/             # Обученные модели
│   ├── best_model_XGB_regularized.joblib
│   ├── confusion_matrix.png
│   ├── experiment_results.csv
│   └── feature_importance.csv
│
├── web/                            # Веб интерфейс
│   ├── app.py                      # Flask application factory
│   ├── routes/                     # Обработчики роутов
│   │   ├── main.py                 # Dashboard
│   │   ├── realtime.py             # Анализ в реальном времени
│   │   ├── offline.py              # Офлайн анализ датасета
│   │   └── api.py                  # REST API endpoints
│   ├── services/                   # Бизнес-логика
│   ├── templates/                  # HTML templates
│   ├── static/                     # CSS, JS, images
│   └── __init__.py
│
├── requirements.txt
└── LICENSE
```

### Key Components

| Directory/File                 | Description                                          |
| ------------------------------ | ---------------------------------------------------- |
| `realtime/`                    | Модуль анализа трафика в реальном времени            |
| `realtime_testing_artifacts/`  | Выходные артефакты real-time сессий                  |
| `scripts/run_realtime.py`      | Основная точка входа для анализа живого трафика      |
| `scripts/local_attack_test.py` | Симулирует атаки для тестирования IDS                |
| `src/inference/`               | Загрузка модели и пакетное предсказание              |
| `src/simulation/`              | Офлайн replay потоков и метрики                      |
| `web/`                         | Интерактивный веб-интерфейс на Flask, Chart.js и SSE |

---

## Installation

### Requirements

* Python 3.12+
* 8GB+ RAM (рекомендуется 16GB)
* GPU с поддержкой CUDA (опционально, для более быстрого обучения)
* **Права администратора/root** (для захвата пакетов)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/TraficAnalysis.git
cd TraficAnalysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Windows-Specific Setup for Real-Time Analysis

Захват пакетов в реальном времени на Windows требует **Npcap**:

1. **Скачать Npcap**: [https://npcap.com/#download](https://npcap.com/#download)

2. **Установить с совместимостью с WinPcap**:

   * Запустите установщик **от имени администратора**
   * Отметьте **"Install Npcap in WinPcap API-compatible Mode"**
   * Завершите установку

3. **Установить scapy**:

   ```powershell
   pip install scapy
   ```

4. **Проверить установку**:

   ```powershell
   python -c "from scapy.all import get_if_list; print(get_if_list())"
   ```

### Linux Setup for Real-Time Analysis

```bash
# Install libpcap
sudo apt-get install libpcap-dev

# Install scapy
pip install scapy

# Grant capture permissions (alternative to running as root)
sudo setcap cap_net_raw,cap_net_admin=eip $(which python)
```

---

## Data Pipeline

Пайплайн данных преобразует сырые CSV файлы в датасеты, готовые для модели, через следующие стадии:

### Pipeline Stages

```
Raw CSV → Bronze (merged) → Cleaned → Preprocessed → Train/Val/Test Splits
```

### Running the Pipeline

```bash
# Run complete pipeline
python scripts/run_data_pipeline.py

# Run specific steps
python scripts/run_data_pipeline.py --steps 1,2,3    # Manifest, ingest, EDA
python scripts/run_data_pipeline.py --steps 4,5      # Clean and split
```

### Processing Steps

| Step | Script                  | Description                                      |
| ---- | ----------------------- | ------------------------------------------------ |
| 1    | `01_manifest.py`        | Создать манифест данных с метаданными файлов     |
| 2    | `02_ingest_bronze.py`   | Объединить CSV в один Parquet файл               |
| 3    | `03_audit_eda.py`       | Аудит качества данных и EDA-визуализации         |
| 4    | `04_build_processed.py` | Очистить данные и применить препроцессинг        |
| 5    | `05_make_splits.py`     | Создать стратифицированные train/val/test сплиты |

### Data Cleaning

Очистка устраняет известные проблемы CIC-IDS-2017:

| Issue                | Solution                              |
| -------------------- | ------------------------------------- |
| Пропуски (NaN)       | Импутация медианой                    |
| Бесконечные значения | Заменяются на NaN, затем импутируются |
| Дубликаты строк      | Удаляются (~9.3% данных)              |
| Пустые строки меток  | Удаляются (~9.25% данных)             |
| Дубликаты колонок    | Удаляется `Fwd Header Length.1`       |
| Выбросы              | Клипаются по 0.1–99.9 перцентилю      |

### Feature Scaling

* **Scaler**: RobustScaler (устойчив к выбросам)
* **Features**: оставлены 79 числовых признаков
* **Dropped**: Flow ID, Source/Destination IP, Timestamp (чтобы предотвратить утечку данных)

### Data Splits

| Split | Rows      | Benign | Attack | Strategy   |
| ----- | --------- | ------ | ------ | ---------- |
| Train | 1,981,378 | 80.3%  | 19.7%  | Stratified |
| Val   | 424,581   | 80.3%  | 19.7%  | Stratified |
| Test  | 424,581   | 80.3%  | 19.7%  | Stratified |

---

## Model Training

### Models Evaluated

| Model          | Description                     | GPU Support |
| -------------- | ------------------------------- | ----------- |
| Random Forest  | Ансамбль деревьев решений       | Нет         |
| XGBoost        | Градиентный бустинг по деревьям | Да (CUDA)   |
| LightGBM       | Лёгкий градиентный бустинг      | Да (GPU)    |
| Neural Network | PyTorch MLP                     | Да (CUDA)   |

### Hyperparameter Configurations

#### XGBoost

```python
{
    "n_estimators": [100, 150, 200],
    "max_depth": [6, 8, 10],
    "learning_rate": [0.05, 0.1],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1.0],
    "scale_pos_weight": "auto"  # Обрабатывает дисбаланс классов
}
```

#### LightGBM

```python
{
    "n_estimators": [100, 200, 300],
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.05, 0.1, 0.15],
    "class_weight": "balanced"
}
```

#### Neural Network

```python
{
    "hidden_layers": [(64, 32), (128, 64, 32), (256, 128, 64)],
    "learning_rate": [0.0005, 0.001],
    "dropout": 0.3,
    "batch_size": 512,
    "max_epochs": 100
}
```

### Training Environment

Обучение выполнялось в Google Colab с:

* **GPU**: NVIDIA Tesla T4 (15GB VRAM)
* **RAM**: 12GB
* **Runtime**: ~15–25 минут суммарно

### Running Training

**Вариант 1: Google Colab (рекомендуется)**

1. Загрузите `splits.zip` и `artifacts.zip` в Google Drive
2. Откройте `notebooks/CIC_IDS_2017_model_training.ipynb` в Colab
3. Выберите GPU runtime: Runtime → Change runtime type → GPU
4. Запустите все ячейки

**Вариант 2: Локальное обучение**

```bash
python scripts/train_models.py
```

---

## Results

### Model Comparison

| Model               | Training Time | Val F1     | Val ROC-AUC | Test F1    | Test ROC-AUC |
| ------------------- | ------------- | ---------- | ----------- | ---------- | ------------ |
| **XGB_regularized** | 24.9s         | **0.9993** | 0.9999      | **0.9994** | 0.9999       |
| XGB_deep            | 28.1s         | 0.9993     | 0.9999      | 0.9993     | 0.9999       |
| LGBM_deep           | 60.3s         | 0.9993     | 0.9999      | 0.9993     | 0.9999       |
| LGBM_baseline       | 40.6s         | 0.9992     | 0.9999      | 0.9991     | 0.9999       |
| LGBM_fast           | 63.6s         | 0.9989     | 0.9998      | 0.9988     | 0.9998       |
| RF_baseline         | 522.7s        | 0.9989     | 0.9999      | 0.9988     | 0.9999       |
| RF_deep             | 320.8s        | 0.9981     | 0.9999      | 0.9982     | 0.9999       |
| XGB_baseline        | 21.9s         | 0.9974     | 0.9999      | 0.9973     | 0.9999       |
| RF_wide             | 563.1s        | 0.9901     | 0.9998      | 0.9900     | 0.9998       |

### Best Model: XGB_regularized

```python
{
    "n_estimators": 150,
    "max_depth": 8,
    "learning_rate": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0
}
```

Гиперпараметры:

* n_estimators: 150
* max_depth: 8
* learning_rate: 0.1
* reg_alpha: 0.1
* reg_lambda: 1.0

### Ensemble Performance

Топ-5 моделей объединены с помощью взвешенного soft voting:

| Metric    | Best Single Model | Ensemble | Difference |
| --------- | ----------------- | -------- | ---------- |
| F1        | 0.9994            | 0.9994   | +0.0000    |
| ROC-AUC   | 0.9999            | 0.9999   | +0.0000    |
| PR-AUC    | 0.9999            | 0.9999   | +0.0000    |
| Precision | 0.9991            | 0.9992   | +0.0001    |
| Recall    | 0.9997            | 0.9996   | -0.0001    |

### Confusion Matrix

![Confusion Matrix](training_artifacts/confusion_matrix.png)

### Classification Report

```
              precision    recall  f1-score   support

      Benign     1.0000    0.9998    0.9999    340985
      Attack     0.9992    0.9997    0.9994     83596

    accuracy                         0.9998    424581
   macro avg     0.9996    0.9997    0.9997    424581
weighted avg     0.9998    0.9998    0.9998    424581
```

### Top Feature Importance (XGBoost)

| Rank | Feature                | Importance |
| ---- | ---------------------- | ---------- |
| 1    | Init_Win_bytes_forward | 0.142      |
| 2    | Bwd Packet Length Std  | 0.089      |
| 3    | Flow IAT Std           | 0.076      |
| 4    | Fwd IAT Total          | 0.065      |
| 5    | Bwd Packet Length Mean | 0.058      |
| 6    | Flow Duration          | 0.054      |
| 7    | Fwd Packet Length Max  | 0.048      |
| 8    | Subflow Fwd Bytes      | 0.041      |
| 9    | Packet Length Std      | 0.038      |
| 10   | Average Packet Size    | 0.035      |

### Note on High Accuracy

Датасет CIC-IDS-2017 известен тем, что даёт очень высокие метрики точности (99%+) во многих исследованиях. Это связано с:

1. **Ярко выраженными сигнатурами атак**: сгенерированные атаки имеют однородные паттерны
2. **Контролируемой средой**: весь трафик из одной сети
3. **Синтетической природой**: атаки генерировались конкретными инструментами

Эти результаты согласуются с опубликованной литературой, но могут не отражать производительность в реальном мире. Для production-систем важно тестирование на разнообразном, реальном трафике.

---

## Testing & Simulation

Этот проект включает комплексный фреймворк тестирования и симуляции, чтобы проверить end-to-end пайплайн перед деплоем.

### Running Tests

#### Unit Tests

```bash
# Unit tests
pytest tests/ -v
```

**Test Coverage:**

# End-to-end tests

python scripts/run_e2e_test.py

**E2E Test Components:**

| Test                | Description                       | Validates                    |
| ------------------- | --------------------------------- | ---------------------------- |
| Model Loading       | Загрузка обученной XGBoost модели | Целостность файла модели     |
| Data Loading        | Загрузка test parquet данных      | Выход data pipeline          |
| Single Inference    | Предсказать один flow             | Функциональность предиктора  |
| Batch Inference     | Предсказать 1000 flows            | Пакетная обработка           |
| Inference Pipeline  | Обработать 5 batch’ей             | Интеграция полного пайплайна |
| Metrics Collector   | Агрегировать статистику           | Расчёт метрик                |
| Database Operations | CRUD операции                     | Хранилище SQLite             |
| No Data Loss        | Проверить количество flows        | Целостность пайплайна        |

**Expected Output:**

```
======================================================================
E2E TEST SUITE
======================================================================

Running: Model Loading... PASSED (1.80s)
Running: Data Loading... PASSED (0.58s)
Running: Single Inference... PASSED (0.42s)
Running: Batch Inference... PASSED (0.48s)
Running: Inference Pipeline... PASSED (0.42s)
Running: Metrics Collector... PASSED (1.01s)
Running: Database Operations... PASSED (0.15s)
Running: No Data Loss... PASSED (0.45s)

======================================================================
SUMMARY
======================================================================
Passed: 8/8
Failed: 0/8

All tests passed!
```

### Flow Simulation

Пайплайн симуляции проигрывает (replay) потоки из тестового датасета через обученную модель, собирая метрики в реальном времени.

#### Running Simulation

```bash
# Full test dataset simulation
python scripts/run_simulation.py

# Fast simulation with limits
python scripts/run_simulation.py --speed 5 --max-flows 50000

# Use validation set, skip database
python scripts/run_simulation.py --source val --no-db

# Quiet mode with custom output
python scripts/run_simulation.py --quiet --output results.json
```

#### Simulation Parameters

| Parameter        | Default | Description                       |
| ---------------- | ------- | --------------------------------- |
| `--source`       | test    | Источник данных: train, val, test |
| `--speed`        | 1.0     | Множитель скорости replay         |
| `--batch-size`   | 100     | Кол-во flows на batch             |
| `--max-flows`    | None    | Ограничить общее число flows      |
| `--max-duration` | None    | Ограничить время в секундах       |
| `--no-db`        | False   | Пропустить сохранение в базу      |
| `--no-viz`       | False   | Пропустить генерацию визуализаций |

### Simulation Results

| Metric              | Value             |
| ------------------- | ----------------- |
| Throughput          | ~50,000 flows/sec |
| Latency p99         | 0.02ms            |
| F1 Score            | 0.9994            |
| False Positive Rate | 0.026%            |

---

## Real-Time Traffic Analysis

### Overview

Модуль реального времени захватывает живой сетевой трафик, извлекает flow-фичи, совместимые с CICIDS2017, и классифицирует каждый поток с помощью обученной модели.

### Module Components

| File                            | Description                                                                 |
| ------------------------------- | --------------------------------------------------------------------------- |
| `realtime/capture.py`           | Захват пакетов с помощью scapy, поддержка Windows (Npcap) и Linux (libpcap) |
| `realtime/flow_aggregator.py`   | Агрегирует пакеты в двунаправленные потоки со статистиками                  |
| `realtime/feature_extractor.py` | Извлекает 79 CICIDS2017-совместимых фичей из потоков                        |
| `realtime/analyzer.py`          | Загружает модель и выполняет предсказания                                   |
| `realtime/pipeline.py`          | Оркестрирует полный processing pipeline                                     |
| `realtime/utils.py`             | Логирование, алерты, сбор метрик                                            |
| `realtime/config.py`            | Управление конфигурацией                                                    |
| `realtime/web_interface.py`     | REST API для интеграции с вебом                                             |

### Running Real-Time Analysis

**Prerequisites:**

* Права администратора/root
* Установлен Npcap (Windows) или libpcap (Linux)
* Обученная модель в `training_artifacts/`

**Basic Usage:**

```powershell
# Windows (Run PowerShell as Administrator)
python scripts/run_realtime.py -i "Беспроводная сеть" -v

# Linux (Run as root or with capabilities)
sudo python scripts/run_realtime.py -i eth0 -v
```

**Command Line Options:**

| Option              | Description                      | Default                                                |
| ------------------- | -------------------------------- | ------------------------------------------------------ |
| `-i, --interface`   | Имя сетевого интерфейса          | Интерактивный выбор                                    |
| `-m, --model`       | Путь к файлу модели              | `training_artifacts/best_model_XGB_regularized.joblib` |
| `-t, --threshold`   | Порог детекции                   | 0.5                                                    |
| `-f, --filter`      | BPF фильтр пакетов               | `ip`                                                   |
| `-d, --duration`    | Время работы в секундах          | 0 (бесконечно)                                         |
| `-v, --verbose`     | Показывать все flows             | False                                                  |
| `-q, --quiet`       | Показывать только атаки          | False                                                  |
| `-o, --output-dir`  | Директория вывода                | `realtime_testing_artifacts`                           |
| `--no-save`         | Отключить сохранение результатов | False                                                  |
| `--list-interfaces` | Показать доступные интерфейсы    | -                                                      |

**Examples:**

```powershell
# List available network interfaces
python scripts/run_realtime.py --list-interfaces

# Run for 5 minutes with verbose output
python scripts/run_realtime.py -i "Ethernet" -v -d 300

# Run with lower threshold (more sensitive)
python scripts/run_realtime.py -i "Wi-Fi" -t 0.3 -v

# Capture only HTTP/HTTPS traffic
python scripts/run_realtime.py -i "Ethernet" -f "tcp port 80 or tcp port 443"

# Run without saving results
python scripts/run_realtime.py -i "Ethernet" --no-save
```

### Real-Time Output

**Console Output (Verbose Mode):**

```
======================================================================
       REAL-TIME NETWORK TRAFFIC ANALYZER
======================================================================

----------------------------------------------------------------------
  Interface: Беспроводная сеть
  Model: best_model_XGB_regularized.joblib
  Threshold: 0.5
  Output: realtime_testing_artifacts\20260111_182417
----------------------------------------------------------------------

Starting... (Ctrl+C to stop)

[OK] Capture running

TIME     STATUS    SOURCE                  DESTINATION             PROTO  CONF   PACKETS
18:24:25   BENIGN   192.168.0.102:52304 ->   150.171.28.11:443   TCP  82.0%    38pkts
18:24:25   BENIGN   192.168.0.102:59781 -> 185.199.111.133:443   TCP  82.0%    26pkts
18:24:25   ATTACK   192.168.0.102:60498 ->    172.66.44.77:443   TCP  87.3%    14pkts

======================================================================
    ATTACK DETECTED!
======================================================================
  Time:       2026-01-11T18:24:25.123456
  Source:     192.168.0.102:60498
  Target:     172.66.44.77:443
  Protocol:   TCP
  Confidence: 87.3%
  Traffic:    14 packets, 2048 bytes
======================================================================
```

**Summary Mode (Default):**

```
[  45s] Pkts:    1,234 | Flows:   156 | Attacks: 🚨3 |   27.4 pps
```

### Session Artifacts

Каждая сессия создаёт папку с таймстампом в `realtime_testing_artifacts/`:

```
realtime_testing_artifacts/
└── 20260111_182417/
    ├── flows.csv              # Все проанализированные flows
    ├── attacks.csv            # Только обнаруженные атаки
    ├── stats.json             # Статистика сессии
    ├── session.log            # Файл лога атак
    ├── traffic_analysis.png   # Графики трафика
    └── attack_analysis.png    # Разбор атак (если атаки обнаружены)
```

**flows.csv Format:**

| Column     | Description                              |
| ---------- | ---------------------------------------- |
| timestamp  | Время детекции                           |
| src_ip     | IP-адрес источника                       |
| src_port   | Порт источника                           |
| dst_ip     | IP-адрес назначения                      |
| dst_port   | Порт назначения                          |
| protocol   | Номер протокола (6=TCP, 17=UDP)          |
| duration   | Длительность потока в секундах           |
| packets    | Общее число пакетов                      |
| bytes      | Общее число байтов                       |
| prediction | Предсказание модели (0=benign, 1=attack) |
| class_name | BENIGN или ATTACK                        |
| confidence | Уверенность модели                       |
| is_attack  | Булев флаг                               |

**stats.json Example:**

```json
{
  "session_id": "20260111_182417",
  "start_time": "2026-01-11T18:24:18.575942",
  "end_time": "2026-01-11T19:19:07.341739",
  "runtime_seconds": 3288.77,
  "total_packets": 13289,
  "total_flows": 13842,
  "total_attacks": 150,
  "attack_rate": 0.0108,
  "packets_per_second": 4.04,
  "pipeline_stats": {
    "packets_processed": 13288,
    "flows_analyzed": 13842,
    "attacks_detected": 150,
    "analysis_errors": 0
  },
  "analyzer_stats": {
    "total_predictions": 13842,
    "benign_count": 13692,
    "attack_count": 150,
    "model_loaded": true,
    "n_features": 79,
    "latency_p50_ms": 0.52,
    "latency_p95_ms": 0.99,
    "latency_p99_ms": 3.07
  }
}
```

### Traffic Analysis Charts

Аналайзер автоматически генерирует визуализации:

![Traffic Analysis](realtime_testing_artifacts/20260111_174606/traffic_analysis.png)

**Charts include:**

* **Скорость сетевого трафика**: packets/sec по времени
* **Скорость анализа flows**: flows/sec
* **Обнаруженные атаки**: накопительное число атак
* **Attack Rate**: процент вредоносных flows

### Testing with Simulated Attacks

Чтобы тестировать IDS без реальных атак, используйте скрипт симуляции атак:

```powershell
# Terminal 1: Start the analyzer
python scripts/run_realtime.py -i "Ethernet" -v

# Terminal 2: Run simulated attacks
python scripts/local_attack_test.py -a all
```

**Available Attack Types:**

| Attack    | Description                  | Command              |
| --------- | ---------------------------- | -------------------- |
| portscan  | TCP сканирование портов      | `-a portscan -c 500` |
| syn       | SYN flood (много соединений) | `-a syn -c 1000`     |
| udp       | UDP flood                    | `-a udp -c 1000`     |
| http      | Flood HTTP запросами         | `-a http -c 200`     |
| slowloris | Медленная HTTP атака         | `-a slowloris -d 30` |
| brute     | Симуляция brute force        | `-a brute -c 50`     |
| all       | Все атаки последовательно    | `-a all`             |

**Example:**

```powershell
# Run all attack types
python scripts/local_attack_test.py -a all

# Port scan only
python scripts/local_attack_test.py -a portscan -c 200

# SYN flood to specific port
python scripts/local_attack_test.py -a syn -p 80 -c 500

# Attack a specific target
python scripts/local_attack_test.py -t 192.168.1.100 -a portscan
```

### Troubleshooting

| Issue                   | Solution                                                            |
| ----------------------- | ------------------------------------------------------------------- |
| "No interfaces found"   | Установите Npcap с WinPcap API compatibility mode                   |
| "Permission denied"     | Запустите от имени администратора (Windows) или root (Linux)        |
| "Model not found"       | Проверьте путь в `-m`, по умолчанию ожидается `training_artifacts/` |
| "scapy import error"    | Выполните `pip install scapy`                                       |
| No packets captured     | Проверьте имя интерфейса, попробуйте `--list-interfaces`            |
| All flows marked BENIGN | Модель может требовать дообучения под ваши паттерны трафика         |

### Performance Notes

| Metric                 | Typical Value              |
| ---------------------- | -------------------------- |
| Обработка пакетов      | 10,000+ пакетов/сек        |
| Задержка анализа flows | <1ms (p50), <3ms (p99)     |
| Использование памяти   | ~200MB базово + flow cache |
| Использование CPU      | 5–15% одного ядра          |

---

## Web Interface

Система включает интерактивный веб-интерфейс на **Flask**, **HTML/CSS**, **JavaScript** и **Chart.js** для:

* мониторинга в реальном времени (живой захват + стриминг обновлений),
* офлайн-анализа датасета (batch inference + метрики),
* базового REST API доступа для интеграций.

### Launching the Web Interface

#### Windows (PowerShell)

Запускайте из корня проекта **в PowerShell от имени администратора**, с **активированным виртуальным окружением**:

```powershell
# 1) Activate venv
.\.venv\Scripts\Activate.ps1

# 2) Run the web server (recommended entry point)
python scripts/run_web.py --debug
```

Затем откройте:

* `http://127.0.0.1:5000`

#### Linux/Mac

```bash
source .venv/bin/activate
python do.py
```

### Dashboard (`/`)

Главная страница даёт быстрый обзор:

* **Информация о модели**: имя/тип модели, число фичей, эталонные метрики (например, test F1).
* **Последние сессии**: прошлые real-time запуски с runtime, количеством flows и количеством атак.
* Навигация на страницы **Real-Time** и **Offline**.

![Dashboard](web/screenshots/dash.png)

### Real-Time Analysis (`/realtime`)

Мониторинг живого сетевого трафика:

* **Выбор интерфейса**: выбрать сетевой интерфейс из обнаруженного списка.
* **Управление порогом**: настройка чувствительности (по умолчанию: `0.5`).
* **Панель live-статистики**:

  * пакеты, flows, атаки,
  * packets/sec,
  * attack rate (%).
* **Графики**:

  * скорость трафика (packets/sec),
  * attack rate (%) по времени.
* **Таблица последних flows**:

  * до 50 последних flows,
  * опциональный фильтр “только атаки”,
  * включает source/destination, protocol, confidence, число пакетов.
* **Server-Sent Events (SSE)**: стриминг обновлений без перезагрузки страницы.

#### Online Example Result

![Online analysis result](web/screenshots/online.png)

### Offline Analysis (`/offline`)

Загрузка и анализ исторических датасетов:

* **Загрузка файла**: поддерживаются `.csv` и `.parquet`.
* **Автоматический анализ**:

  * определяет ground-truth метки, если они присутствуют,
  * применяет тот же препроцессинг + модель, что и при обучении,
  * считает полный набор метрик классификации (precision/recall/F1/confusion matrix) при наличии меток.
* **Визуализация результатов**:

  * summary карточки (общее число flows, attack rate, benign vs attack),
  * гистограмма распределения вероятностей,
  * метрики классификации (если есть метки).
* **Индикатор прогресса**: показывает процент выполнения во время анализа.

#### Offline Example Result

![Offline analysis result](web/screenshots/ofline.png)

### API Endpoints

Веб-интерфейс предоставляет REST API для интеграции:

| Endpoint                | Description                                               |
| ----------------------- | --------------------------------------------------------- |
| `GET /api/model-info`   | Имя и тип модели, число фичей и эталонные test-метрики(а) |
| `GET /api/sessions`     | Последние real-time сессии анализа                        |
| `GET /api/health`       | Health check                                              |
| `POST /offline/analyze` | Запустить офлайн-анализ датасета                          |
| `GET /offline/progress` | Опрос прогресса анализа и результатов                     |

### Architecture

* **Backend**: Flask + blueprints (`main`, `realtime`, `offline`, `api`)
* **Frontend**: vanilla JS + Chart.js + SSE
* **State**: in-process global state objects для фоновых задач
* **Execution**: background threads для длительных офлайн-задач

---

## Usage

### Loading a Trained Model

```python
import joblib
import pandas as pd

# Load model
model = joblib.load("training_artifacts/best_model_XGB_regularized.joblib")

# Load preprocessor
preprocessor = joblib.load("artifacts/preprocessor.joblib")

# Load feature schema
import json
with open("artifacts/feature_schema.json") as f:
    schema = json.load(f)

feature_cols = schema["feature_columns"]

# Predict on new data
def predict(df):
    X = df[feature_cols].values
    X_scaled = preprocessor.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    return predictions, probabilities
```

### Loading the Ensemble

```python
import json
import joblib
import numpy as np

# Load ensemble config
with open("training_artifacts/ensemble_config.json") as f:
    config = json.load(f)

# Load models
models = {}
for name in config["models"]:
    models[name] = joblib.load(f"training_artifacts/{name}.joblib")

weights = config["weights"]

# Ensemble prediction
def ensemble_predict(X):
    probas = []
    for model in models.values():
        probas.append(model.predict_proba(X))
    
    weighted_proba = np.zeros_like(probas[0])
    total_weight = sum(weights)
    
    for proba, weight in zip(probas, weights):
        weighted_proba += proba * (weight / total_weight)
    
    return np.argmax(weighted_proba, axis=1), weighted_proba[:, 1]
```

### Using the Inference Pipeline

```python
from src.inference import Predictor, InferencePipeline

# Initialize predictor
predictor = Predictor(
    model_path="training_artifacts/best_model_XGB_regularized.joblib",
    preprocessor_path="artifacts/preprocessor.joblib",
    feature_schema_path="artifacts/feature_schema.json",
    threshold=0.5
)
predictor.load()

# Create pipeline
pipeline = InferencePipeline(predictor)

# Process flows
alerts = pipeline.process_batch(
    features=X,
    flow_indices=list(range(len(X))),
    true_labels=y  # Optional
)

# Get statistics
stats = pipeline.get_stats()
print(f"F1: {stats['f1']:.4f}")
print(f"Alerts: {stats['total_alerts']}")
```

### Using Real-Time Pipeline Programmatically

```python
from realtime import RealtimePipeline

# Define callbacks
def on_attack(result):
    print(f"Attack from {result.src_ip}: {result.confidence:.1%}")

def on_flow(result):
    print(f"Flow: {result.src_ip} -> {result.dst_ip}")

# Create pipeline
pipeline = RealtimePipeline(
    interface="Ethernet",
    model_path="training_artifacts/best_model_XGB_regularized.joblib",
    preprocessor_path="artifacts/preprocessor.joblib",
    feature_schema_path="artifacts/feature_schema.json",
    threshold=0.5,
    on_attack_detected=on_attack_detected,
    on_flow_analyzed=on_flow_analyzed
)

# Start capture
pipeline.start()

# Run for 60 seconds
import time
time.sleep(60)

# Get results
stats = pipeline.get_stats()
attacks = pipeline.get_recent_attacks(10)

# Stop
pipeline.stop()
```

---

## References

### Dataset

```
Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018).
"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"
International Conference on Information Systems Security and Privacy (ICISSP)
```

### Related Work

```
Panigrahi, R., & Borah, S. (2018).
"A detailed analysis of CICIDS2017 dataset for designing Intrusion Detection Systems"
International Journal of Engineering & Technology

Aksu, D., Ustebay, S., Aydin, M. A., & Atmaca, T. (2018).
"Intrusion Detection with Comparative Analysis of Supervised Learning Techniques"
International Symposium on Computer and Information Sciences (ISCIS)
```

### Dataset Download

* Official: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
* Kaggle: [https://www.kaggle.com/datasets/ciaboroghigiovanni/cicids2017](https://www.kaggle.com/datasets/ciaboroghigiovanni/cicids2017)

---

## License

Этот проект распространяется по лицензии MIT — см. файл [LICENSE](LICENSE) для деталей.

---

## Acknowledgments

* Canadian Institute for Cybersecurity (CIC) за датасет CIC-IDS-2017
* University of New Brunswick за хостинг датасета и документацию
