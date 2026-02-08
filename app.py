import streamlit as st
import engine
import PIL.Image
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="IntelliQueue AI", page_icon="👁️", layout="centered")

# --- Custom CSS for Professional UI ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    div.stButton > button {
        width: 100%; background-color: #FF4B4B; color: white;
        font-size: 18px; font-weight: bold; border-radius: 10px;
        border: none; padding: 0.5em;
    }
    div.stButton > button:hover { background-color: #FF2B2B; color: white; }
    .stAlert { font-weight: 500; }
    </style>
    """, unsafe_allow_html=True)

# --- App Header ---
st.title("👁️ IntelliQueue AI")
st.markdown("### *Autonomous Retail Reasoning Engine*")
st.caption("Real-world spatial reasoning powered by **Gemini 1.5 Flash**.")

# --- Real-Time Logic & Default Settings ---
# Capture current real-world time for default initialization
now = datetime.now()
current_hour_real = now.hour
current_day_real = now.weekday() # 0 = Monday, 6 = Sunday

# Default to 2 PM (14:00) if current time is outside business hours (for demo safety)
default_time = current_hour_real
if current_hour_real < 8 or current_hour_real > 22:
    default_time = 14

# --- Sidebar: Control Panel ---
st.sidebar.header("📍 Control Panel")

# 1. Temporal Input (Time Slider)
hour = st.sidebar.slider("Analysis Time (24h)", 
                        min_value=8, max_value=22, 
                        value=default_time,
                        help="Select time to simulate past/future scenarios.")

# 2. Simulation Indicator (UI Feedback)
# If the selected time differs from real time, indicate 'Simulation Mode'
if hour != current_hour_real:
    st.sidebar.warning(f"⚠️ Simulation Mode Active\nSimulating scenario for **{hour}:00** (Not Live)")
else:
    st.sidebar.success(f"🔴 System Live\nReal-time monitoring: **{hour}:00**")

# 3. Spatial Inputs (Crowd & Staff)
crowd = st.sidebar.slider("Crowd Density", 1, 100, 35)
staff = st.sidebar.slider("Active Staff", 1, 10, 3)

# 4. Contextual Input
# "Rainy Weather" is included here for environmental reasoning
context = st.sidebar.selectbox("Environmental Context", 
    ["Normal Day", "Holiday Rush", "Staff Shortage", "Technical Issue", "Rainy Weather"])

st.sidebar.markdown("---")

# 5. Multimodal Input (CCTV Snapshot)
st.sidebar.markdown("### 📷 Visual Intelligence")
uploaded_file = st.sidebar.file_uploader("Upload CCTV Frame (Optional)", type=["jpg", "png", "jpeg"])

image_input = None
if uploaded_file is not None:
    # Process image for display and API usage
    image_input = PIL.Image.open(uploaded_file)
    st.sidebar.image(image_input, caption=f"snapshot_{hour}00.jpg", use_container_width=True)
    st.sidebar.success("✅ Visual Data Synced")

# --- Main Action Trigger ---
if st.button("🚀 Analyze Queue & Generate Action Plan"):
    
    # Spinner gives user feedback while API processes data
    with st.spinner(f'Orchestrating autonomous reasoning for {hour}:00...'):
        try:
            # Initialize the Engine
            qe = engine.IntelliQueueEngine()
            
            # Get Prediction (Pass inputs to the engine)
            wait_time, reasoning = qe.get_prediction(hour, current_day_real, staff, crowd, context, image_file=image_input)
            
            # --- Results Display ---
            col1, col2 = st.columns([1, 3])
            
            # Metric Display (The "What")
            with col1:
                st.metric(label="Predicted Wait", value=f"{wait_time} min")
            
            # Visual Status Indicator
            with col2:
                if wait_time > 20:
                    st.error(f"⚠️ Critical Load at {hour}:00")
                elif wait_time > 10:
                    st.warning(f"⚠️ Moderate Delays at {hour}:00")
                else:
                    st.success(f"✅ Optimal Flow at {hour}:00")

            # Reasoning Output (The "Why" & "How")
            st.markdown("---")
            st.markdown("### 🧠 Autonomous Action Plan")
            st.markdown(reasoning)
            
        except Exception as e:
            st.error(f"System Error: {str(e)}")
            st.info("Tip: Please check your API Key configuration.")

# --- Footer ---
st.divider()
st.caption("Built for Google DeepMind Gemini 3 Hackathon | Action Era Track")
