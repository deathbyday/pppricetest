# app.py — Streamlit application สำหรับทำนายความเสี่ยงโรคเบาหวาน

import streamlit as st
import numpy as np
import joblib
import json

# ===== การตั้งค่าหน้าเว็บ =====
# st.set_page_config ต้องเป็น Streamlit command แรกเสมอ
# ถ้าเรียกทีหลังจะ error
st.set_page_config(
    page_title="ระบบทำนายโอกาศเข้าบริการ",
    page_icon="🚗",          # icon ที่แสดงบน browser tab
    layout="centered",        # centered หรือ wide
    initial_sidebar_state="expanded"
)

# ===== โหลดโมเดลและข้อมูล =====
# ใช้ @st.cache_resource เพื่อโหลดโมเดลครั้งเดียว
# โดยไม่โหลดซ้ำทุกครั้งที่ผู้ใช้ interact กับ app
# นี่คือ performance optimization ที่สำคัญมาก
@st.cache_resource
def load_model():
    """โหลด pipeline และ metadata — ทำครั้งเดียวตอนเริ่ม app"""
    pipeline = joblib.load("model_artifacts/linear_model.pkl")
    with open("model_artifacts/model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return pipeline, metadata

# โหลดโมเดล — Streamlit จะแสดง spinner ระหว่างรอ
with st.spinner("กำลังโหลดโมเดล..."):
    pipeline, metadata = load_model()

# ===== Sidebar: ข้อมูลเกี่ยวกับโมเดล =====
# Sidebar เหมาะสำหรับข้อมูลเสริมที่ไม่ใช่ส่วนหลักของ app
with st.sidebar:
    st.header("ℹ️ เกี่ยวกับโมเดลนี้")
    st.write(f"**ประเภทโมเดล:** {metadata['model_type']}")
    st.write(f"**ความแม่นยำ:** {metadata['accuracy']*100:.1f}%")
    st.write(f"**ข้อมูล train:** {metadata['training_samples']:,} ราย")
    st.markdown("![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)")

    st.divider()  # เส้นคั่น
    st.subheader("⚠️ ข้อควรระวัง")
    st.warning(
        "เป็นแค่ตัวทดสอบ ไม่มีข้อมีผลจริง"
    )

# ===== ส่วนหลัก: Header =====
st.title("คำนวณโอกาศที่ลูกค้าจะเข้าบริการ")
st.markdown("""

""")

st.divider()

# ===== ส่วนรับ Input =====
st.subheader("ข้อมูล")

# แบ่งหน้าจอเป็นสองคอลัมน์เพื่อให้ดูกระชับขึ้น
col1, col2 = st.columns(2)

with col1:
    # st.number_input สำหรับตัวเลขที่ต้องการความแม่นยำ
    temperature = st.number_input(
        "อุณหภูมิช่วงเวลานี้ ในองศาเซลเซียส",
        min_value=0, max_value=99,
        value=25, step=1,
    )

    apparentTemperature = st.number_input(
        "รู้สึกอุณหภูมิช่วงเวลานี้เท่าไหร่ ในองศาเซลเซียส",
        min_value=0, max_value=300,
        value=30, step=1,
    )

    precipProbability = st.number_input(
        "โอกาศฝนจะตกหรือไม่",
        min_value=0, max_value=150,
        value=60, step=1,
    )

    humidity = st.number_input(
        "ความชื้น",
        min_value=0, max_value=100,
        value=20, step=1,
    )

with col2:
    windGust = st.number_input(
        "ความเร็วลม",
        min_value=0, max_value=900,
        value=8, step=1,
    )

    pressure = st.number_input(
        "แรงดันอากาศ",
        min_value=0.0, max_value=70.0,
        value=10.0, step=0.1,
        format="%.1f",
    )

    uvIndex = st.number_input(
        "โอกาศฝนจะตกหรือไม่",
        min_value=0.0, max_value=99.0,
        value=0.35, step=0.001,
        format="%.3f",
    )

    precipIntensityMax = st.number_input(
        "โอกาศที่ฝนจะตกในรอบวัน",
        min_value=1, max_value=120,
        value=80, step=1
    )

st.divider()

# ===== ปุ่มทำนายและแสดงผล =====
# การใช้ column เพื่อจัดปุ่มให้อยู่กลาง
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_button = st.button(
        "ประมวลผลค่า",
        use_container_width=True,
        type="primary"  # ทำให้ปุ่มสีเด่น
    )

# เมื่อผู้ใช้กดปุ่ม จึงค่อยทำการทำนาย
# การตรวจสอบ if predict_button ทำให้ app ไม่ทำนายโดยอัตโนมัติตอนโหลด
if predict_button:

    # รวบรวม input เป็น array ในลำดับเดียวกับที่ train
    # ลำดับนี้ต้องตรงกับ feature_names ที่บันทึกไว้จาก Colab
    input_data = np.array([[
        temperature,
    apparentTemperature,   
    precipProbability,     
    humidity,              
    windGust,             
    pressure,             
    uvIndex,             
    precipIntensityMax
    ]])

    # ทำนายด้วย pipeline — การ scale เกิดขึ้นอัตโนมัติภายใน pipeline
    with st.spinner("กำลังประเมิน..."):
        prediction = pipeline.predict(input_data)


    st.subheader("ผลการคำนวณ")


    st.success(f"""
        ### โอกาศที่ผู้คนเริ่มอยากใช้บริการ: {abs(prediction[0]/25):.2f}%
        """)
    st.markdown("![Alt Text](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExcTlidnMwdjZpNnVnYjVxa3FmbG9pZ2o2Y2wwbWcwOGwxc3Fia29yMiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7aCYDNm1kXgSUgXm/giphy.gif)")

