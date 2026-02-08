import streamlit as st
import engine
import PIL.Image
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="IntelliQueue AI", page_icon="👁️", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    div.stButton > button {
        width: 100%; background-color: #FF4B4B; color: white;
        font-size: 18px; font-weight: bold; border-radius: 10px;
        border: none; padding: 0.5em;
    }
    div.stButton > button:hover { background-color: #FF2B2B; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("👁️ IntelliQueue AI")
st.markdown("### *Autonomous Retail Reasoning Engine*")

# --- Real Time vs Simulation Logic ---
now = datetime.now()
current_hour_real = now.hour
current_day_real = now.weekday()

# --- Sidebar ---
st.sidebar.header("📍 Control Panel")

# Slider Logic
hour = st.sidebar.slider("Analysis Time (24h)", 8, 22, value=14)

# 🔥 Simulation Indicator
# Agar Slider ka time Real Time se alag hai, toh "Simulation Mode" dikhao
if hour != current_hour_real:
    st.sidebar.warning(f"⚠️ Simulation Mode Active\nAnalyzing scenario for **{hour}:00** (Not Live)")
else:
    st.sidebar.success(f"🔴 System Live\nReal-time monitoring: **{hour}:00**")

# Inputs
crowd = st.sidebar.slider("Crowd Density", 1, 100, 35)
staff = st.sidebar.slider("Active Staff", 1, 10, 3)
context = st.sidebar.selectbox("Context", ["Normal Day", "Holiday Rush", "Staff Shortage", "Technical Issue"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📷 CCTV Feed")
uploaded_file = st.sidebar.file_uploader("Upload Frame (Optional)", type=["jpg", "png", "jpeg"])

image_input = None
if uploaded_file is not None:
    image_input = PIL.Image.open(uploaded_file)
    st.sidebar.image(image_input, caption=f"snapshot_{hour}00.jpg", use_container_width=True) # Caption mein time match kar diya
    st.sidebar.success("✅ Frame Synced with Timeline")

# --- Main Action ---
if st.button("🚀 Run Temporal Analysis"):
    with st.spinner(f'Processing spatial-temporal data for {hour}:00...'):
        try:
            qe = engine.IntelliQueueEngine()
            
            # Prediction
            wait_time, reasoning = qe.get_prediction(hour, current_day_real, staff, crowd, context, image_file=image_input)
            
            # Display
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric(label="Est. Wait", value=f"{wait_time} min")
            with col2:
                if wait_time > 20:
                    st.error(f"⚠️ Critical Load at {hour}:00")
                elif wait_time > 10:
                    st.warning(f"⚠️ Moderate Load at {hour}:00")
                else:
                    st.success(f"✅ Normal Flow at {hour}:00")

            st.markdown("---")
            st.markdown(reasoning)
            
        except Exception as e:
            st.error(f"System Error: {str(e)}")

st.divider()
st.caption("Gemini 3 Hackathon | Autonomous Action Agent")
