import streamlit as st
import engine

st.set_page_config(page_title="IntelliQueue AI", page_icon="⏳", layout="centered")

# Custom Styling for "Wow Factor"
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⏳ IntelliQueue AI")
st.markdown("### *Autonomous Context-Aware Reasoning Engine*")
st.caption("Moving beyond simple chat into the **Action Era** of Retail.")

# Sidebar for Inputs (Spatiotemporal Factors)
st.sidebar.header("📍 Real-time Environment")
crowd = st.sidebar.slider("Current Crowd Density (People)", 1, 100, 20)
staff = st.sidebar.slider("Active Staff Units", 1, 10, 3)
context = st.sidebar.selectbox("Environmental Context", 
    ["Normal Day", "Holiday Rush", "Staff Shortage", "Technical Issue", "Rainy Weather"])

st.sidebar.divider()
st.sidebar.info("Built for **Google DeepMind Gemini 3 Hackathon**")

# Main Action
if st.button("🚀 Analyze Queue & Generate Action Plan"):
    with st.spinner('Orchestrating AI Reasoning...'):
        # Mocking hour=14 (2PM), day=2 (Wednesday)
        # result ab tuple return karega: (wait_time, reasoning)
        wait_time, reasoning = engine.get_prediction(14, 2, staff, crowd, context)
        
        # 1. Output Header
        st.success(f"## Predicted Wait Time: {wait_time} Minutes")
        
        # 2. Display Gemini's Multimodal Reasoning
        st.markdown("---")
        st.markdown(reasoning) # Isme Thought Signature aur Action Plan automatically formatted aayenge

st.divider()
st.caption("Powered by **Gemini 1.5 Flash** | Spatial-Temporal Reasoning Enabled")
