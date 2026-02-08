import joblib
import pandas as pd
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class IntelliQueueEngine:
    def __init__(self):
        # ML model load karna
        # Ensure 'models/queue_model.pkl' exists in your repo
        try:
            self.model = joblib.load('models/queue_model.pkl')
        except:
            self.model = None # Fallback if model missing during deploy
            
        # API Key load karna (Streamlit secrets or .env)
        self.api_key = os.environ.get("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    def get_prediction(self, hour, day, staff, crowd, context="Normal Day"):
        # 1. ML Prediction (Base Calculation)
        if self.model:
            input_data = pd.DataFrame([[hour, day, staff, crowd]], 
                                     columns=['hour_of_day', 'day_of_week', 'current_staff', 'current_crowd'])
            base_wait = self.model.predict(input_data)[0]
        else:
            # Simple fallback logic if ML model file is not found
            base_wait = (crowd / (staff + 1)) * 5 

        # --- ⚡ SMART LOGIC LAYER (Updated for Realism) ---
        if context == "Normal Day":
            base_wait = base_wait * 0.8  
        elif context == "Holiday Rush":
            base_wait = base_wait * 1.5  
        elif context == "Staff Shortage":
            base_wait = base_wait * 1.3  
        elif context == "Rainy Weather":
            base_wait = base_wait * 1.1  
        elif context == "Technical Issue":
            base_wait = (base_wait * 2.0) + 10.0 

        # --- SANITY CHECK (Low Crowd Logic) ---
        if crowd < 5 and context != "Technical Issue":
            max_logical_wait = crowd * 4  
            if base_wait > max_logical_wait:
                base_wait = max_logical_wait
        
        final_wait_time = round(base_wait, 1)

        # 2. Gemini API Call (Using gemini-1.5-flash-lite for high quota)
        model_name = "gemini-1.5-flash-lite"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
        
        prompt = f"""
        Act as an Autonomous Retail Reasoning Engine for the Gemini 3 Action Era.
        
        Situation:
        - Current Time: {hour}:00
        - Staff Units: {staff}
        - Crowd Density: {crowd} people
        - Environmental Context: {context}
        - ML Predicted Wait Time: {final_wait_time} minutes.

        Please provide your analysis in the following structured format:

        ### 🧠 Thought Signature (Spatial-Temporal Reasoning)
        Briefly explain your "thinking process" on how the {context} and crowd density specifically impact the flow of customers. Mention cause and effect.

        ### 📋 Manager's Action Plan (Thinking Levels)
        - **Level 1 (Immediate - 5 mins):** One tactical action.
        - **Level 2 (Short-term - 1 hour):** One operational adjustment.
        - **Level 3 (Strategic - Long-term):** One suggestion to prevent future {context} issues.
        
        Keep it professional and concise.
        """

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        headers = {'Content-Type': 'application/json'}

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response_json = response.json()
            
            # Extracting the AI text
            if "candidates" in response_json:
                reasoning = response_json['candidates'][0]['content']['parts'][0]['text']
            else:
                reasoning = f"⚠️ AI Reasoning Engine Offline (Error: {response_json.get('error', {}).get('message', 'Unknown')})"
        except Exception as e:
            reasoning = f"⚠️ Connection Error: {str(e)}. Base ML prediction remains valid."

        return final_wait_time, reasoning





