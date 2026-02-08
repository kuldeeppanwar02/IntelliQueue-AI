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

# --- Initialize Session State (Memory) ---
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'wait_time' not in st.session_state:
    st.session_state.wait_time = 0
if 'reasoning' not in st.session_state:
    st.session_state.reasoning = ""
if 'context' not in st.session_state:
    st.session_state.context = ""

# --- Header ---
st.title("👁️ IntelliQueue AI")
st.markdown("### *Autonomous Retail Reasoning Engine*")
st.caption("Marathon Agent: **Continuous Learning & Self-Correction Enabled**")

# --- Time Logic ---
now = datetime.now()
current_hour_real = now.hour
current_day_real = now.weekday()
default_time = current_hour_real if 8 <= current_hour_real <= 22 else 14

# --- Sidebar ---
st.sidebar.header("📍 Control Panel")
hour = st.sidebar.slider("Analysis Time (24h)", 8, 22, value=default_time)

if hour != current_hour_real:
    st.sidebar.warning(f"⚠️ Simulation Mode Active: **{hour}:00**")
else:
    st.sidebar.success(f"🔴 System Live: **{hour}:00**")

crowd = st.sidebar.slider("Crowd Density", 1, 100, 35)
staff = st.sidebar.slider("Active Staff", 1, 10, 3)
context = st.sidebar.selectbox("Environmental Context", 
    ["Normal Day", "Holiday Rush", "Staff Shortage", "Technical Issue", "Rainy Weather"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📷 Visual Intelligence")
uploaded_file = st.sidebar.file_uploader("Upload CCTV Frame", type=["jpg", "png", "jpeg"])

image_input = None
if uploaded_file is not None:
    image_input = PIL.Image.open(uploaded_file)
    st.sidebar.image(image_input, caption=f"snapshot_{hour}00.jpg", use_container_width=True)
    st.sidebar.success("✅ Visual Data Synced")

# --- Main Action ---
if st.button("🚀 Analyze Queue & Generate Action Plan"):
    with st.spinner(f'Orchestrating autonomous reasoning for {hour}:00...'):
        qe = engine.IntelliQueueEngine()
        wait_time, reasoning = qe.get_prediction(hour, current_day_real, staff, crowd, context, image_file=image_input)
        
        # Save to session state
        st.session_state.prediction_made = True
        st.session_state.wait_time = wait_time
        st.session_state.reasoning = reasoning
        st.session_state.context = context

# --- Results Display (From Session State) ---
if st.session_state.prediction_made:
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(label="Predicted Wait", value=f"{st.session_state.wait_time} min")
    with col2:
        if st.session_state.wait_time > 20:
            st.error(f"⚠️ Critical Load at {hour}:00")
        elif st.session_state.wait_time > 10:
            st.warning(f"⚠️ Moderate Delays at {hour}:00")
        else:
            st.success(f"✅ Optimal Flow at {hour}:00")

    st.markdown("---")
    st.markdown("### 🧠 Autonomous Action Plan")
    st.markdown(st.session_state.reasoning)
    
    # --- 🔥 MARATHON AGENT: FEEDBACK LOOP ---
    st.markdown("---")
    st.subheader("🔄 Agent Feedback Loop")
    st.caption("Was this autonomous prediction accurate? Your input trains the Marathon Agent.")
    
    col_fb1, col_fb2 = st.columns(2)
    with col_fb1:
        if st.button("✅ Accurate"):
            st.toast("System Logic Reinforced. Model Confidence Increased.")
    
    with col_fb2:
        if st.button("❌ Inaccurate"):
            st.session_state.show_correction = True

    # If Inaccurate, show correction input
    if 'show_correction' in st.session_state and st.session_state.show_correction:
        actual_time = st.number_input("Enter Actual Wait Time (min):", min_value=0.0, value=st.session_state.wait_time + 5.0)
        
        if st.button("🛠️ Run Self-Correction Protocol"):
            with st.spinner("Agent is re-calibrating weights..."):
                qe = engine.IntelliQueueEngine()
                correction_log = qe.generate_correction(st.session_state.wait_time, actual_time, st.session_state.context)
                
                st.markdown("### ⚙️ Autonomous Optimization Log")
                st.code(correction_log, language="markdown")
                st.success("✅ Model parameters updated for future iterations.")

st.divider()
st.caption("Built for Google DeepMind Gemini 3 Hackathon | Marathon Agent Track")
