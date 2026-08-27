"""ML-модель для прогнозирования результатов и классификации уровня подготовки."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import joblib
import os

def prepare_data_for_prediction(df):
    """Подготавливает данные для обучения модели прогнозирования."""
    # Для каждого ученика берём первые 4 теста как признаки, 5-й как целевую переменную
    students = df['student_id'].unique()
    X_list = []
    y_list = []
    
    for student_id in students:
        student_data = df[df['student_id'] == student_id].sort_values('test_number')
        
        if len(student_data) >= 5:
            # Берём результаты первых 4 тестов
            features = student_data.iloc[:4][['run_100m', 'pull_ups', 'long_jump', 
                                               'shuttle_run', 'abs_exercises']].values.flatten()
            # Целевая переменная — результат 5-го теста
            target = student_data.iloc[4][['run_100m', 'pull_ups', 'long_jump', 
                                            'shuttle_run', 'abs_exercises']].values
            
            X_list.append(features)
            y_list.append(target)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y

def train_prediction_model(df):
    """Обучает модель прогнозирования результатов."""
    X, y = prepare_data_for_prediction(df)
    
    # Прогнозируем каждый показатель отдельно
    models = {}
    metrics = {}
    
    target_names = ['run_100m', 'pull_ups', 'long_jump', 'shuttle_run', 'abs_exercises']
    
    for i, target_name in enumerate(target_names):
        y_target = y[:, i]
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        models[target_name] = pipeline
        metrics[target_name] = mae
    
    return models, metrics

def classify_fitness_level(df):
    """Классифицирует уровень подготовки учеников."""
    latest_test = df[df['test_number'] == df['test_number'].max()].copy()
    
    # Нормализуем и инвертируем показатели (где меньше = лучше)
    latest_test['run_100m_score'] = -latest_test['run_100m']
    latest_test['shuttle_run_score'] = -latest_test['shuttle_run']
    
    # Считаем общий рейтинг
    score_cols = ['run_100m_score', 'pull_ups', 'long_jump', 'shuttle_run_score', 'abs_exercises']
    latest_test['total_score'] = latest_test[score_cols].sum(axis=1)
    
    # Определяем уровни подготовки
    q25 = latest_test['total_score'].quantile(0.25)
    q75 = latest_test['total_score'].quantile(0.75)
    
    latest_test['fitness_level'] = pd.cut(
        latest_test['total_score'],
        bins=[-np.inf, q25, q75, np.inf],
        labels=['Низкий', 'Средний', 'Высокий']
    )
    
    return latest_test[['student_id', 'total_score', 'fitness_level']]

def train_classifier(df):
    """Обучает классификатор уровня подготовки."""
    classified = classify_fitness_level(df)
    
    # Берём данные до предпоследнего теста для прогнозирования
    X_list = []
    y_list = []
    
    for student_id in classified['student_id']:
        student_data = df[df['student_id'] == student_id].sort_values('test_number')
        
        if len(student_data) >= 4:
            # Берём первые 3 теста как признаки
            features = student_data.iloc[:3][['run_100m', 'pull_ups', 'long_jump', 
                                               'shuttle_run', 'abs_exercises']].values.flatten()
            # Целевая переменная — уровень подготовки по последнему тесту
            level = classified[classified['student_id'] == student_id]['fitness_level'].values[0]
            
            X_list.append(features)
            y_list.append(level)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    
    print(f"✅ Точность классификатора: {accuracy:.4f}")
    print("\nОтчёт классификации:")
    print(classification_report(y_test, pipeline.predict(X_test)))
    
    return pipeline

def save_models(prediction_models, classifier, path='models/'):
    """Сохраняет модели."""
    os.makedirs(path, exist_ok=True)
    
    for name, model in prediction_models.items():
        joblib.dump(model, f'{path}/predict_{name}.pkl')
    
    joblib.dump(classifier, f'{path}/classifier.pkl')
    print(f"💾 Модели сохранены в {path}")

def load_models(path='models/'):
    """Загружает модели."""
    prediction_models = {}
    target_names = ['run_100m', 'pull_ups', 'long_jump', 'shuttle_run', 'abs_exercises']
    
    for name in target_names:
        prediction_models[name] = joblib.load(f'{path}/predict_{name}.pkl')
    
    classifier = joblib.load(f'{path}/classifier.pkl')
    
    return prediction_models, classifier

def predict_next_results(prediction_models, student_history):
    """Прогнозирует результаты следующего теста."""
    # Берём последние 4 теста
    features = student_history.tail(4)[['run_100m', 'pull_ups', 'long_jump', 
                                         'shuttle_run', 'abs_exercises']].values.flatten().reshape(1, -1)
    
    predictions = {}
    for name, model in prediction_models.items():
        predictions[name] = model.predict(features)[0]
    
    return predictions

def predict_fitness_level(classifier, student_history):
    """Прогнозирует уровень подготовки."""
    # Берём первые 3 теста
    features = student_history.head(3)[['run_100m', 'pull_ups', 'long_jump', 
                                         'shuttle_run', 'abs_exercises']].values.flatten().reshape(1, -1)
    
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]
    
    return prediction, probabilities
