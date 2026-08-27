"""Статистический анализ и визуализация данных."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def calculate_class_statistics(df):
    """Рассчитывает статистику по классу."""
    latest_test = df[df['test_number'] == df['test_number'].max()]
    
    stats_dict = {
        'Среднее 100м (сек)': latest_test['run_100m'].mean(),
        'Медиана 100м (сек)': latest_test['run_100m'].median(),
        'Ст.откл. 100м': latest_test['run_100m'].std(),
        'Среднее подтягивания': latest_test['pull_ups'].mean(),
        'Медиана подтягивания': latest_test['pull_ups'].median(),
        'Среднее прыжок (см)': latest_test['long_jump'].mean(),
        'Среднее челночный бег': latest_test['shuttle_run'].mean(),
        'Среднее пресс': latest_test['abs_exercises'].mean(),
        'Всего учеников': latest_test['student_id'].nunique()
    }
    return stats_dict

def analyze_progress(df):
    """Анализирует прогресс класса от теста к тесту."""
    progress = df.groupby('test_number').agg({
        'run_100m': 'mean',
        'pull_ups': 'mean',
        'long_jump': 'mean',
        'shuttle_run': 'mean',
        'abs_exercises': 'mean'
    }).reset_index()
    
    # Рассчитываем процент улучшения
    for col in ['run_100m', 'shuttle_run']:  # меньше = лучше
        baseline = progress[col].iloc[0]
        progress[f'{col}_improvement'] = ((baseline - progress[col]) / baseline) * 100
    
    for col in ['pull_ups', 'long_jump', 'abs_exercises']:  # больше = лучше
        baseline = progress[col].iloc[0]
        progress[f'{col}_improvement'] = ((progress[col] - baseline) / baseline) * 100
    
    return progress

def identify_at_risk_students(df, threshold_percentile=25):
    """Выявляет отстающих учеников."""
    latest_test = df[df['test_number'] == df['test_number'].max()]
    
    # Нормализуем результаты (приводим к единой шкале)
    # Для бега: меньше = лучше, инвертируем
    latest_test['run_100m_score'] = -latest_test['run_100m']
    latest_test['shuttle_run_score'] = -latest_test['shuttle_run']
    
    # Считаем общий рейтинг
    score_cols = ['run_100m_score', 'pull_ups', 'long_jump', 'shuttle_run_score', 'abs_exercises']
    latest_test['total_score'] = latest_test[score_cols].sum(axis=1)
    
    # Определяем порог
    threshold = latest_test['total_score'].quantile(threshold_percentile / 100)
    at_risk = latest_test[latest_test['total_score'] <= threshold].copy()
    
    return at_risk[['student_id', 'run_100m', 'pull_ups', 'long_jump', 
                    'shuttle_run', 'abs_exercises', 'total_score']]

def calculate_correlations(df):
    """Рассчитывает корреляции между показателями."""
    latest_test = df[df['test_number'] == df['test_number'].max()]
    metrics = ['run_100m', 'pull_ups', 'long_jump', 'shuttle_run', 'abs_exercises']
    corr_matrix = latest_test[metrics].corr()
    return corr_matrix

def plot_progress_charts(progress_df):
    """Создаёт графики прогресса класса."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Динамика результатов класса', fontsize=16, fontweight='bold')
    
    metrics = [
        ('run_100m', '100м бег (сек)', 'red'),
        ('pull_ups', 'Подтягивания (раз)', 'blue'),
        ('long_jump', 'Прыжок в длину (см)', 'green'),
        ('shuttle_run', 'Челночный бег (сек)', 'orange'),
        ('abs_exercises', 'Пресс за 1 мин (раз)', 'purple')
    ]
    
    for idx, (col, title, color) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        ax.plot(progress_df['test_number'], progress_df[col], 
                marker='o', linewidth=2, color=color, markersize=8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Номер теста')
        ax.set_ylabel('Значение')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(progress_df['test_number'])
    
    plt.tight_layout()
    return fig

def plot_distribution(df, metric='pull_ups'):
    """Создаёт график распределения результатов."""
    latest_test = df[df['test_number'] == df['test_number'].max()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(latest_test[metric], bins=15, kde=True, ax=ax, color='steelblue')
    ax.set_title(f'Распределение результатов: {metric}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Количество учеников')
    ax.axvline(latest_test[metric].mean(), color='red', linestyle='--', 
               label=f'Среднее: {latest_test[metric].mean():.2f}')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(corr_matrix):
    """Создаёт тепловую карту корреляций."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, ax=ax, 
                xticklabels=['100м', 'Подтяг.', 'Прыжок', 'Челн.бег', 'Пресс'],
                yticklabels=['100м', 'Подтяг.', 'Прыжок', 'Челн.бег', 'Пресс'])
    ax.set_title('Корреляция между показателями', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def generate_analytics_report(df):
    """Генерирует текстовый аналитический отчёт."""
    stats = calculate_class_statistics(df)
    progress = analyze_progress(df)
    at_risk = identify_at_risk_students(df)
    
    report = f"""
# 📊 АНАЛИТИЧЕСКИЙ ОТЧЁТ ПО ФИЗИЧЕСКОЙ ПОДГОТОВКЕ

## Общая статистика класса
- **Количество учеников:** {stats['Всего учеников']}
- **Средний результат 100м:** {stats['Среднее 100м (сек)']:.2f} сек
- **Среднее количество подтягиваний:** {stats['Среднее подтягивания']:.1f} раз
- **Средний прыжок в длину:** {stats['Среднее прыжок (см)']:.1f} см

## Прогресс класса
- **Улучшение 100м:** {progress['run_100m_improvement'].iloc[-1]:.1f}%
- **Улучшение подтягиваний:** {progress['pull_ups_improvement'].iloc[-1]:.1f}%
- **Улучшение прыжка:** {progress['long_jump_improvement'].iloc[-1]:.1f}%

## Отстающие ученики
Выявлено **{len(at_risk)}** учеников с результатами ниже 25-го перцентиля.

### Рекомендации:
1. Уделить дополнительное внимание {len(at_risk)} отстающим ученикам
2. Сфокусироваться на упражнениях, показывающих наименьший прогресс
3. Рассмотреть индивидуальные программы тренировок
"""
    return report
