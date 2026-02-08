import streamlit as st
import engine  # Import the engine module

# Page Configuration
st.set_page_config(page_title="IntelliQueue AI", page_icon="⏳", layout="centered")

# Custom Styling for Professional Look
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
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Header & Branding
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

# Main Action Button
if st.button("🚀 Analyze Queue & Generate Action Plan"):
    with st.spinner('Orchestrating AI Reasoning...'):
        try:
            # 1. Create an instance of the engine class (Ye line fix hai!)
            qe = engine.IntelliQueueEngine()
            
            # 2. Call the prediction function using the object 'qe'
            # Mocking hour=14 (2PM), day=2 (Wednesday) for demo
            wait_time, reasoning = qe.get_prediction(14, 2, staff, crowd, context)
            
            # 3. Output Header
            st.success(f"## Predicted Wait Time: {wait_time} Minutes")
            
            # 4. Display Gemini's Multimodal Reasoning
            st.markdown("---")
            st.markdown("### 🧠 Autonomous Reasoning & Action Plan")
            st.markdown(reasoning) 
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.divider()
st.caption("Powered by **Gemini Flash Lite Latest** | Spatial-Temporal Reasoning Enabled")

