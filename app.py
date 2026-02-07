import streamlit as st
from engine import IntelliQueueEngine

st.set_page_config(page_title="IntelliQueue AI", page_icon="🚀")

st.title("🤖 IntelliQueue AI: Smart Queue Manager")
st.markdown("Eliminating physical queues with Gemini 3 Reasoning & ML")

# Sidebar for Inputs
st.sidebar.header("Real-time Stats")
crowd = st.sidebar.slider("Current Crowd (People)", 1, 100, 20)
staff = st.sidebar.slider("Active Staff Counters", 1, 10, 3)
context = st.sidebar.selectbox("Context", ["Normal Day", "Holiday Rush", "Staff Shortage", "Technical Issue", "Rainy Weather"])

if st.button("Analyze Queue"):
    with st.spinner('AI is reasoning...'):
        engine = IntelliQueueEngine()
        # Mocking hour=14 (2PM), day=2 (Wednesday)
        result = engine.get_prediction(14, 2, staff, crowd, context)
        
        # UI Display
        st.success(f"### Estimated Wait Time: {result['wait_time']} Minutes")
        st.info(f"#### 🧠 AI Reasoning & Logic:\n{result['reasoning']}")

st.divider()
st.caption("Built for Gemini 3 Hackathon | Powered by Google AI Studio")