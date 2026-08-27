"""Генерация синтетических данных о физической подготовке учащихся."""
import numpy as np
import pandas as pd

def generate_students_data(n_students=30, n_tests=5, seed=42):
    """
    Генерирует данные о результатах тестов учащихся.
    
    Тесты:
    - 100м бег (сек) — меньше лучше
    - Подтягивания (раз)
    - Прыжок в длину (см)
    - Челночный бег 3x10 (сек) — меньше лучше
    - Пресс за 1 минуту (раз)
    """
    np.random.seed(seed)
    
    students = []
    student_id = 1
    
    for test_num in range(1, n_tests + 1):
        for _ in range(n_students):
            # Базовые результаты с индивидуальной вариацией
            base_fitness = np.random.normal(0, 1)  # общая физподготовка
            
            # 100м бег (8-15 сек)
            run_100m = 11.5 - base_fitness * 0.8 + np.random.normal(0, 0.5)
            run_100m = np.clip(run_100m, 8, 15)
            
            # Подтягивания (0-20 раз)
            pull_ups = 8 + base_fitness * 4 + np.random.normal(0, 2)
            pull_ups = np.clip(pull_ups, 0, 20)
            
            # Прыжок в длину (150-250 см)
            long_jump = 200 + base_fitness * 20 + np.random.normal(0, 10)
            long_jump = np.clip(long_jump, 150, 250)
            
            # Челночный бег 3x10 (20-30 сек)
            shuttle_run = 25 - base_fitness * 2 + np.random.normal(0, 1)
            shuttle_run = np.clip(shuttle_run, 20, 30)
            
            # Пресс за 1 минуту (20-60 раз)
            abs_exercises = 40 + base_fitness * 10 + np.random.normal(0, 5)
            abs_exercises = np.clip(abs_exercises, 20, 60)
            
            # Улучшение от теста к тесту (прогресс)
            progress_factor = 1 + (test_num - 1) * 0.02 * np.random.uniform(0.5, 1.5)
            
            students.append({
                'student_id': student_id,
                'test_number': test_num,
                'run_100m': run_100m / progress_factor,
                'pull_ups': pull_ups * progress_factor,
                'long_jump': long_jump * progress_factor,
                'shuttle_run': shuttle_run / progress_factor,
                'abs_exercises': abs_exercises * progress_factor
            })
            student_id += 1
    
    df = pd.DataFrame(students)
    
    # Нормативы (для сравнения)
    df['run_100m_grade'] = pd.cut(df['run_100m'], 
                                   bins=[0, 12, 13.5, 15], 
                                   labels=['Отлично', 'Хорошо', 'Удовл.'])
    df['pull_ups_grade'] = pd.cut(df['pull_ups'], 
                                   bins=[-1, 8, 12, 20], 
                                   labels=['Удовл.', 'Хорошо', 'Отлично'])
    
    return df

def generate_single_student_history(student_id=1, n_tests=5, seed=42):
    """Генерирует историю одного ученика."""
    np.random.seed(seed + student_id)
    
    history = []
    base_fitness = np.random.normal(0, 1)
    
    for test_num in range(1, n_tests + 1):
        progress = 1 + (test_num - 1) * 0.03
        
        history.append({
            'test_number': test_num,
            'run_100m': 11.5 - base_fitness * 0.8 + np.random.normal(0, 0.3) / progress,
            'pull_ups': (8 + base_fitness * 4 + np.random.normal(0, 1)) * progress,
            'long_jump': (200 + base_fitness * 20 + np.random.normal(0, 5)) * progress,
            'shuttle_run': (25 - base_fitness * 2 + np.random.normal(0, 0.5)) / progress,
            'abs_exercises': (40 + base_fitness * 10 + np.random.normal(0, 2)) * progress
        })
    
    return pd.DataFrame(history)pip show streamlit
