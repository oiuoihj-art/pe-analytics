"""Streamlit-дашборд для учителя физкультуры."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from data_generator import generate_students_data, generate_single_student_history
from analytics import (calculate_class_statistics, analyze_progress, 
                       identify_at_risk_students, calculate_correlations,
                       generate_analytics_report)
from model import (train_prediction_model, train_classifier, 
                   predict_next_results, predict_fitness_level,
                   save_models, load_models)

st.set_page_config(page_title="Физкультура — Аналитика", page_icon="🏃", layout="wide")

# === Заголовок ===
st.title("🏃 Система мониторинга физической подготовки учащихся")
st.markdown("""
**Аналитический дашборд для учителя физкультуры**  
Прогнозирование результатов • Выявление отстающих • Рекомендации по тренировкам
""")

# === Загрузка данных ===
@st.cache_data
def load_data():
    return generate_students_data(n_students=30, n_tests=5)

df = load_data()

# === Боковая панель ===
st.sidebar.header("📊 Навигация")
page = st.sidebar.radio("Выберите раздел", [
    "📈 Общая статистика",
    "👤 Анализ ученика",
    "🤖 ML-прогнозирование",
    "⚠️ Отстающие ученики",
    "📝 Аналитический отчёт"
])

# === Загрузка/обучение моделей ===
@st.cache_resource
def load_or_train_models(data):
    models_path = 'models/'
    if os.path.exists(f'{models_path}/classifier.pkl'):
        return load_models(models_path)
    else:
        st.info("⏳ Обучение ML-моделей...")
        pred_models = train_prediction_model(data)
        classifier = train_classifier(data)
        save_models(pred_models, classifier, models_path)
        return pred_models, classifier

# === Страница 1: Общая статистика ===
if page == "📈 Общая статистика":
    st.header("📈 Общая статистика класса")
    
    stats = calculate_class_statistics(df)
    progress = analyze_progress(df)
    
    # Метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Учеников", f"{stats['Всего учеников']}")
    with col2:
        st.metric("Среднее 100м", f"{stats['Среднее 100м (сек)']:.2f} сек")
    with col3:
        st.metric("Подтягивания", f"{stats['Среднее подтягивания']:.1f} раз")
    with col4:
        st.metric("Прыжок в длину", f"{stats['Среднее прыжок (см)']:.1f} см")
    
    st.markdown("---")
    
    # Графики прогресса
    st.subheader("📊 Динамика результатов класса")
    
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=('100м бег', 'Подтягивания', 'Прыжок в длину',
                                       'Челночный бег', 'Пресс'))
    
    metrics = [
        ('run_100m', 1, 1, 'red'),
        ('pull_ups', 1, 2, 'blue'),
        ('long_jump', 1, 3, 'green'),
        ('shuttle_run', 2, 1, 'orange'),
        ('abs_exercises', 2, 2, 'purple')
    ]
    
    for col, row, col_num, color in metrics:
        fig.add_trace(
            go.Scatter(x=progress['test_number'], y=progress[col],
                      mode='lines+markers', name=col, line=dict(color=color)),
            row=row, col=col_num
        )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Корреляции
    st.subheader("🔗 Корреляция между показателями")
    corr = calculate_correlations(df)
    
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r',
                         labels=dict(x="Показатель", y="Показатель", color="Корреляция"),
                         x=['100м', 'Подтяг.', 'Прыжок', 'Челн.бег', 'Пресс'],
                         y=['100м', 'Подтяг.', 'Прыжок', 'Челн.бег', 'Пресс'])
    st.plotly_chart(fig_corr, use_container_width=True)

# === Страница 2: Анализ ученика ===
elif page == "👤 Анализ ученика":
    st.header("👤 Анализ конкретного ученика")
    
    student_id = st.selectbox("Выберите ученика", 
                              sorted(df['student_id'].unique()),
                              format_func=lambda x: f"Ученик #{x}")
    
    student_history = df[df['student_id'] == student_id].sort_values('test_number')
    
    # График истории
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=('100м бег', 'Подтягивания', 'Прыжок в длину',
                                       'Челночный бег', 'Пресс'))
    
    metrics = [
        ('run_100m', 1, 1, 'red'),
        ('pull_ups', 1, 2, 'blue'),
        ('long_jump', 1, 3, 'green'),
        ('shuttle_run', 2, 1, 'orange'),
        ('abs_exercises', 2, 2, 'purple')
    ]
    
    for col, row, col_num, color in metrics:
        fig.add_trace(
            go.Scatter(x=student_history['test_number'], y=student_history[col],
                      mode='lines+markers', name=col, line=dict(color=color)),
            row=row, col=col_num
        )
    
    fig.update_layout(height=500, showlegend=False, title_text=f"Динамика ученика #{student_id}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Таблица результатов
    st.subheader("📋 Результаты тестов")
    display_cols = ['test_number', 'run_100m', 'pull_ups', 'long_jump', 'shuttle_run', 'abs_exercises']
    st.dataframe(student_history[display_cols].round(2), use_container_width=True)

# === Страница 3: ML-прогнозирование ===
elif page == "🤖 ML-прогнозирование":
    st.header("🤖 ML-прогнозирование результатов")
    
    pred_models, classifier = load_or_train_models(df)
    
    student_id = st.selectbox("Выберите ученика для прогноза", 
                              sorted(df['student_id'].unique()),
                              format_func=lambda x: f"Ученик #{x}",
                              key="ml_student")
    
    student_history = df[df['student_id'] == student_id].sort_values('test_number')
    
    if st.button("🔮 Сделать прогноз", use_container_width=True):
        # Прогноз результатов
        predictions = predict_next_results(pred_models, student_history)
        
        st.subheader("📊 Прогноз результатов следующего теста")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("100м бег", f"{predictions['run_100m']:.2f} сек")
        with col2:
            st.metric("Подтягивания", f"{predictions['pull_ups']:.1f} раз")
        with col3:
            st.metric("Прыжок в длину", f"{predictions['long_jump']:.1f} см")
        with col4:
            st.metric("Челночный бег", f"{predictions['shuttle_run']:.2f} сек")
        with col5:
            st.metric("Пресс", f"{predictions['abs_exercises']:.1f} раз")
        
        # Прогноз уровня подготовки
        level, probs = predict_fitness_level(classifier, student_history)
        
        st.subheader("🎯 Прогноз уровня подготовки")
        st.success(f"**Уровень подготовки:** {level}")
        
        # Вероятности
        prob_df = pd.DataFrame({
            'Уровень': classifier.classes_,
            'Вероятность': probs
        })
        
        fig_probs = px.bar(prob_df, x='Уровень', y='Вероятность',
                           title='Вероятности уровней подготовки',
                           color='Уровень', text_auto='.1%')
        st.plotly_chart(fig_probs, use_container_width=True)

# === Страница 4: Отстающие ученики ===
elif page == "⚠️ Отстающие ученики":
    st.header("⚠️ Отстающие ученики")
    
    at_risk = identify_at_risk_students(df, threshold_percentile=25)
    
    st.warning(f"Выявлено **{len(at_risk)}** учеников с результатами ниже 25-го перцентиля")
    
    if len(at_risk) > 0:
        st.dataframe(at_risk.round(2), use_container_width=True)
        
        # Визуализация
        st.subheader("📊 Сравнение с классом")
        
        latest_test = df[df['test_number'] == df['test_number'].max()]
        
        fig = go.Figure()
        
        for metric in ['pull_ups', 'long_jump', 'abs_exercises']:
            fig.add_trace(go.Box(y=latest_test[metric], name=metric))
        
        fig.update_layout(title='Распределение результатов класса',
                         yaxis_title='Значение')
        st.plotly_chart(fig, use_container_width=True)
        
        # Рекомендации
        st.subheader("💡 Рекомендации")
        st.markdown("""
        1. **Индивидуальный подход** — разработать персональные программы для отстающих
        2. **Дополнительные занятия** — организовать факультативные тренировки
        3. **Мотивация** — поставить конкретные достижимые цели
        4. **Мониторинг** — отслеживать прогресс еженедельно
        """)
    else:
        st.success("Все ученики показывают удовлетворительные результаты!")

# === Страница 5: Аналитический отчёт ===
elif page == "📝 Аналитический отчёт":
    st.header("📝 Аналитический отчёт")
    
    report = generate_analytics_report(df)
    st.markdown(report)
    
    # Кнопка скачивания
    st.download_button(
        label="📥 Скачать отчёт (Markdown)",
        data=report,
        file_name="pe_analytics_report.md",
        mime="text/markdown",
        use_container_width=True
    )
    
    # Экспорт данных
    st.subheader("📊 Экспорт данных")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Скачать все данные (CSV)",
            data=csv,
            file_name="pe_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        latest_test = df[df['test_number'] == df['test_number'].max()]
        csv_latest = latest_test.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Скачать последний тест (CSV)",
            data=csv_latest,
            file_name="pe_latest_test.csv",
            mime="text/csv",
            use_container_width=True
        )
