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
        self.model = joblib.load('models/queue_model.pkl')
        # API Key load karna
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    def get_prediction(self, hour, day, staff, crowd, context="Normal Day"):
        # 1. ML Prediction (Base Calculation)
        input_data = pd.DataFrame([[hour, day, staff, crowd]], 
                                 columns=['hour_of_day', 'day_of_week', 'current_staff', 'current_crowd'])
        base_wait = self.model.predict(input_data)[0]

        # --- ⚡ SMART LOGIC LAYER (Updated for Realism) ---
        
        if context == "Normal Day":
            base_wait = base_wait * 0.8   # Efficiency badh gayi
            
        elif context == "Holiday Rush":
            base_wait = base_wait * 1.5   # Bheed aur complexity zyada
            
        elif context == "Staff Shortage":
            base_wait = base_wait * 1.3   # Staff kam, kaam slow
            
        elif context == "Rainy Weather":
            base_wait = base_wait * 1.1   # Thoda slow (Log slow move karte hain)
            
        elif context == "Technical Issue":
            # Sabse bada problem: Time Double + 10 min extra repair time
            base_wait = (base_wait * 2.0) + 10.0 

        # --- SANITY CHECK (Low Crowd Logic) ---
        # Agar bheed kam hai (5 se kam), lekin 'Technical Issue' NAHI hai, toh time kam karo.
        if crowd < 5 and context != "Technical Issue":
            # Logical max wait per person (e.g., 4 mins)
            max_logical_wait = crowd * 4  
            if base_wait > max_logical_wait:
                base_wait = max_logical_wait
        
        # -------------------------------------

        # 2. Gemini 3 API Call (Using Requests for Stability)
        model_name = "gemini-flash-lite-latest"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
        
        prompt = f"""
        System: Queue Manager AI using Gemini 3.
        Stats: Wait Time {base_wait:.1f} mins, Staff: {staff}, Crowd: {crowd}, Context: {context}.
        Task: Explain the wait time calculation logic based on the context and suggest 1 fix.
        """
        
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                response_json = response.json()
                if "candidates" in response_json and response_json["candidates"]:
                    reasoning = response_json["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    reasoning = "AI response format unexpected. Showing raw stats."
            else:
                reasoning = f"Server Error ({response.status_code}): {response.text}"
                
        except Exception as e:
            reasoning = f"Connection failed: {str(e)}"

        return {
            "wait_time": round(base_wait, 1),
            "reasoning": reasoning

        }





