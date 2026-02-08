import joblib
import pandas as pd
import os
import google.generativeai as genai
import PIL.Image
from dotenv import load_dotenv

load_dotenv()

class IntelliQueueEngine:
    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        try:
            self.model = joblib.load('models/queue_model.pkl')
        except:
            self.model = None 

    def get_prediction(self, hour, day, staff, crowd, context="Normal Day", image_file=None):
        
        # --- ML Prediction ---
        base_wait = 5.0 
        if self.model:
            try:
                input_data = pd.DataFrame([[hour, day, staff, crowd]], 
                                         columns=['hour_of_day', 'day_of_week', 'current_staff', 'current_crowd'])
                base_wait = self.model.predict(input_data)[0]
            except:
                base_wait = (crowd / (staff + 1)) * 4

        if context == "Technical Issue":
            base_wait = (base_wait * 2.0) + 10.0
        elif context == "Staff Shortage":
            base_wait = base_wait * 1.3
        elif context == "Rainy Weather":
            base_wait = base_wait * 1.1
            
        final_wait_time = round(base_wait, 1)

        # --- Gemini Prompt Logic (Fixed for Time Mismatch) ---
        reasoning = "AI analysis unavailable."
        
        try:
            model = genai.GenerativeModel('gemini-flash-lite-latest')
            
            # 🔥 CRITICAL CHANGE: 
            # Hum AI ko bol rahe hain: "Ye ek RECORDED SNAPSHOT hai jo {hour}:00 baje liya gaya tha."
            # Ab Gemini current time se confuse nahi hoga.
            
            prompt_text = f"""
            Act as an Autonomous Retail Reasoning Engine.
            
            Situation Context:
            You are analyzing a specific timeframe to assist store managers.
            - Target Time of Analysis: {hour}:00 Hours (24h format)
            - Staff Active: {staff}
            - Predicted Wait: {final_wait_time} mins
            - Environmental Context: {context}
            """

            content_input = [prompt_text]

            if image_file:
                prompt_text += """
                \n[VISUAL EVIDENCE PROVIDED]
                A CCTV Snapshot corresponding to the Target Time ({hour}:00) is attached.
                1. Analyze the crowd density and mood in the image.
                2. Verify if the visual crowd matches the data input ({crowd} people).
                3. Treat this image as the ground truth for that specific time.
                """
                content_input.append(image_file)
            else:
                prompt_text += "\nNo visual input. Base analysis on data parameters only."

            prompt_text += """
            \nOutput Format:
            ### 👁️ Visual & Temporal Analysis
            (Confirm if the scene looks consistent with a busy/quiet period at {hour}:00).

            ### 🧠 Root Cause Diagnosis
            Why is the wait time {final_wait_time} mins at this specific hour?

            ### 📋 Action Plan (Time-Specific)
            - **Immediate:** Action for {hour}:05.
            - **Strategic:** Adjustment for this shift.
            """
            
            content_input[0] = prompt_text
            response = model.generate_content(content_input)
            reasoning = response.text
            
        except Exception as e:
            reasoning = f"⚠️ AI Analysis Failed: {str(e)}. ML Prediction valid."

        return final_wait_time, reasoning






