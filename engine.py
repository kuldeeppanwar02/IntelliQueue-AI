import joblib
import pandas as pd
import os
import google.generativeai as genai
import PIL.Image
from dotenv import load_dotenv

load_dotenv()

class IntelliQueueEngine:
    def __init__(self):
        # 1. API Configuration
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        # 2. ML Model Load (Fallback logic ke saath)
        try:
            self.model = joblib.load('models/queue_model.pkl')
        except:
            self.model = None 

    def get_prediction(self, hour, day, staff, crowd, context="Normal Day", image_file=None):
        
        # --- Step 1: Base ML Calculation ---
        base_wait = 5.0 
        if self.model:
            try:
                input_data = pd.DataFrame([[hour, day, staff, crowd]], 
                                         columns=['hour_of_day', 'day_of_week', 'current_staff', 'current_crowd'])
                base_wait = self.model.predict(input_data)[0]
            except:
                base_wait = (crowd / (staff + 1)) * 4

        # Context Logic
        if context == "Technical Issue":
            base_wait = (base_wait * 2.0) + 10.0
        elif context == "Staff Shortage":
            base_wait = base_wait * 1.3
            
        final_wait_time = round(base_wait, 1)

        # --- Step 2: Gemini Multimodal Reasoning ---
        
        try:
            # Hum 'gemini flash' use kar rahe hain jo Images dekh sakta hai
            model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')
            
            # Base Prompt
            prompt_text = f"""
            Act as an Autonomous Retail Reasoning Engine (Gemini 3 Action Era).
            
            Data Inputs:
            - Time: {hour}:00
            - Context: {context}
            - Predicted Wait: {final_wait_time} mins
            - Staff Count: {staff}
            """

            # Agar Image hai, toh Prompt change hoga
            if image_file:
                prompt_text += """
                \n[VISUAL ANALYSIS REQUIRED]
                An image of the current queue (CCTV Snapshot) is provided. 
                1. Analyze the crowd's visible sentiment (frustrated, calm, chaotic).
                2. Identify any spatial bottlenecks visible in the scene.
                3. Combine this visual insight with the data above.
                """
                content_input = [prompt_text, image_file] # List mein text + image dono jayenge
            else:
                prompt_text += "\nNo visual input provided. Base analysis on data only."
                content_input = [prompt_text]

            prompt_text += """
            \nProvide output in this format:
            ### 👁️ Visual & Spatial Analysis
            (If image provided: Describe what you see in the queue regarding mood and density. If no image: State "Data-only analysis".)

            ### 🧠 Thought Signature
            Explain the cause-and-effect of the wait time based on inputs.

            ### 📋 Manager's Action Plan
            - **Level 1 (Immediate):** One quick fix.
            - **Level 2 (Short-term):** One operational change.
            - **Level 3 (Strategic):** One long-term prevention tip.
            """

            # API Call
            response = model.generate_content(content_input)
            reasoning = response.text
            
        except Exception as e:
            reasoning = f"⚠️ AI Analysis Failed: {str(e)}. ML Prediction is still valid."

        return final_wait_time, reasoning






