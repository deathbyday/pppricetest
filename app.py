# app.py — Streamlit application สำหรับทำนายความเสี่ยงโรคเบาหวาน

import streamlit as st
import numpy as np
import joblib
import json

# ===== การตั้งค่าหน้าเว็บ =====
# st.set_page_config ต้องเป็น Streamlit command แรกเสมอ
# ถ้าเรียกทีหลังจะ error
st.set_page_config(
    page_title="ระบบทำนายความเสี่ยงโรคเบาหวาน",
    page_icon="🩺",          # icon ที่แสดงบน browser tab
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

    st.divider()  # เส้นคั่น

    st.subheader("⚠️ ข้อควรระวัง")
    st.warning(
        "ผลลัพธ์นี้เป็นการประเมินเบื้องต้นจาก AI เท่านั้น "
        "ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้ "
        "กรุณาปรึกษาแพทย์หากมีข้อสงสัย"
    )

# ===== ส่วนหลัก: Header =====
st.title("🩺 ระบบประเมินความเสี่ยงโรคเบาหวาน")
st.markdown("""
กรอกข้อมูลสุขภาพของคุณด้านล่าง ระบบจะประเมินความเสี่ยงการเป็นโรคเบาหวาน
โดยใช้โมเดล Machine Learning ที่ train จากข้อมูลผู้ป่วย 768 ราย
""")

st.divider()

# ===== ส่วนรับ Input =====
st.subheader("📋 กรอกข้อมูลสุขภาพ")

# แบ่งหน้าจอเป็นสองคอลัมน์เพื่อให้ดูกระชับขึ้น
col1, col2 = st.columns(2)

with col1:
    # st.number_input สำหรับตัวเลขที่ต้องการความแม่นยำ
    temperature = st.number_input(
        "อุณหภูมิช่วงเวลานี้ ในองศาเซลเซียส",
        min_value=0, max_value=99,
        value=1, step=1,
        help="กรอก 0 หากไม่เคยตั้งครรภ์"
    )

    apparentTemperature = st.number_input(
        "รู้สึกอุณหภูมิช่วงเวลานี้เท่าไหร่ ในองศาเซลเซียส",
        min_value=0, max_value=300,
        value=120, step=1,
        help="ค่าปกติ: 70–100 mg/dL (อดอาหาร)"
    )

    precipProbability = st.number_input(
        "โอกาศฝนจะตกหรือไม่",
        min_value=0, max_value=150,
        value=72, step=1,
        help="ค่าตัวล่างของความดันโลหิต"
    )

    humidity = st.number_input(
        "ความชื้น",
        min_value=0, max_value=100,
        value=20, step=1,
        help="วัดที่ต้นแขนด้านหลัง"
    )

with col2:
    windGust = st.number_input(
        "ทิศทางลม",
        min_value=0, max_value=900,
        value=80, step=1,
        help="ระดับ insulin ในเลือด 2 ชั่วโมงหลังทดสอบ"
    )

    pressure = st.number_input(
        "แรงดันอากาศ",
        min_value=0.0, max_value=70.0,
        value=25.0, step=0.1,
        format="%.1f",
        help="น้ำหนัก(kg) ÷ ส่วนสูง(m)²"
    )

    uvIndex = st.number_input(
        "ค่า UV",
        min_value=0.0, max_value=99.0,
        value=0.35, step=0.001,
        format="%.3f",
        help="ค่าที่สะท้อนความเสี่ยงทางพันธุกรรม (0–2.5)"
    )

    precipIntensityMax = st.number_input(
        "โอกาศที่ฝนจะตกในรอบวัน",
        min_value=1, max_value=120,
        value=30, step=1
    )

st.divider()

# ===== ปุ่มทำนายและแสดงผล =====
# การใช้ column เพื่อจัดปุ่มให้อยู่กลาง
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_button = st.button(
        "🔍 ประเมินความเสี่ยง",
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


    st.subheader("📊 ผลการประเมิน")


    st.success(f"""
        ### โอกาศที่ผู้คนเริ่มอยากใช้บริการ: {abs(prediction):.2f}**
        """)

    # แสดงข้อมูลที่กรอกสรุปกลับ เพื่อให้ผู้ใช้ตรวจสอบว่าถูกต้อง
    with st.expander("📋 ดูข้อมูลที่กรอก"):
        summary = {
            "จำนวนครั้งตั้งครรภ์": temperature,
            "ระดับน้ำตาล (mg/dL)": apparentTemperature,
            "ความดันโลหิต (mmHg)": precipProbability,
            "ความหนาผิวหนัง (mm)": humidity,
            "Insulin (mu U/ml)": windGust,
            "BMI": pressure,
            "Diabetes Pedigree Function": uvIndex,
            "อายุ (ปี)": precipIntensityMax
        }
        import pandas as pd
        st.dataframe(
            pd.DataFrame.from_dict(summary, orient="index", columns=["ค่า"]),
            use_container_width=True
        )