import streamlit as st

st.title("🏃 ТЕСТ")
st.write("Если видишь этот текст — всё работает!")

st.header("Статистика")
st.write("Учеников: 30")
st.write("Средний результат 100м: 12.5 сек")

if st.button("Нажми меня"):
    st.success("Кнопка работает! 🎉")
