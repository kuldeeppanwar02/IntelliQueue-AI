import streamlit as st
import engine
import PIL.Image

st.set_page_config(page_title="IntelliQueue AI", page_icon="👁️", layout="centered")

# Styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #FF4B4B; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("👁️ IntelliQueue AI: Vision & Reasoning")
st.markdown("### *Multimodal Autonomous Agent for the Action Era*")
st.caption("Powered by **Gemini 1.5 Flash** | Sees, Reasons, & Acts")

# --- Sidebar ---
st.sidebar.header("📍 Environment Data")
crowd = st.sidebar.slider("Crowd Density", 1, 100, 20)
staff = st.sidebar.slider("Active Staff", 1, 10, 3)
context = st.sidebar.selectbox("Context", ["Normal Day", "Technical Issue", "Staff Shortage", "Holiday Rush"])

st.sidebar.divider()
st.sidebar.markdown("### 📷 Visual Input (Multimodal)")
uploaded_file = st.sidebar.file_uploader("Upload CCTV Snapshot", type=["jpg", "png", "jpeg"])

image_input = None
if uploaded_file is not None:
    image_input = PIL.Image.open(uploaded_file)
    st.sidebar.image(image_input, caption="Uploaded Snapshot", use_column_width=True)
    st.sidebar.success("✅ Image Loaded for Analysis")

# --- Main Action ---
if st.button("🚀 Analyze Queue (Visual + Data)"):
    with st.spinner('Gemini is analyzing visual spatial patterns...'):
        try:
            qe = engine.IntelliQueueEngine()
            
            # Image pass kar rahe hain function mein
            wait_time, reasoning = qe.get_prediction(14, 2, staff, crowd, context, image_file=image_input)
            
            st.success(f"## Predicted Wait Time: {wait_time} Minutes")
            
            st.markdown("---")
            st.markdown(reasoning)
            
        except Exception as e:
            st.error(f"System Error: {str(e)}")

st.divider()
st.caption("Built for Google DeepMind Gemini 3 Hackathon")


