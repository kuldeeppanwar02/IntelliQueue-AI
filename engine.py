import joblib
import pandas as pd
import os
import google.generativeai as genai
import PIL.Image
from dotenv import load_dotenv

load_dotenv()

class IntelliQueueEngine:
    """
    Core reasoning engine for IntelliQueue AI.
    Features: Hybrid Prediction, Multimodal Analysis, and Autonomous Self-Correction.
    """

    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        try:
            self.model = joblib.load('models/queue_model.pkl')
        except:
            self.model = None 

    def get_prediction(self, hour, day, staff, crowd, context="Normal Day", image_file=None):
        """Generates initial prediction and reasoning."""
        
        # --- Base Calculation ---
        base_wait = 5.0 
        if self.model:
            try:
                input_data = pd.DataFrame([[hour, day, staff, crowd]], 
                                         columns=['hour_of_day', 'day_of_week', 'current_staff', 'current_crowd'])
                base_wait = self.model.predict(input_data)[0]
            except:
                base_wait = (crowd / (staff + 1)) * 4

        # Context Multipliers
        if context == "Technical Issue":
            base_wait = (base_wait * 2.0) + 10.0
        elif context == "Staff Shortage":
            base_wait = base_wait * 1.3
        elif context == "Rainy Weather":
            base_wait = base_wait * 1.1
        elif context == "Holiday Rush":
            base_wait = base_wait * 1.5
            
        final_wait_time = round(base_wait, 1)

        # --- Gemini Reasoning ---
        reasoning = "AI analysis unavailable."
        try:
            model = genai.GenerativeModel('gemini-flash-lite-latest')
            
            prompt_text = f"""
            Act as an Autonomous Retail Reasoning Engine.
            Target Time: {hour}:00 | Staff: {staff} | Context: {context} | Predicted Wait: {final_wait_time} min.
            """

            content_input = [prompt_text]

            if image_file:
                prompt_text += "\n[VISUAL EVIDENCE PROVIDED]: Analyze the attached CCTV snapshot."
                content_input.append(image_file)
            else:
                prompt_text += "\nNo visual input provided."

            prompt_text += """
            \nOutput Format:
            ### 👁️ Visual & Temporal Analysis
            (Analyze consistency between data and visual).

            ### 🧠 Root Cause Diagnosis
            Why is the wait time {final_wait_time} min?

            ### 📋 Autonomous Action Plan
            - **Immediate Directive:** Specific action.
            - **Strategic Adjustment:** Long term fix.
            """
            
            content_input[0] = prompt_text
            response = model.generate_content(content_input)
            reasoning = response.text
            
        except Exception as e:
            reasoning = f"⚠️ Analysis Failed: {str(e)}"

        return final_wait_time, reasoning

    # 🔥 NEW FUNCTION: SELF-CORRECTION LOOP
    def generate_correction(self, predicted, actual, context):
        """
        Marathon Agent Capability: 
        Analyzes the gap between prediction and reality to 'learn' and adjust.
        """
        try:
            model = genai.GenerativeModel('gemini-flash-lite-latest')
            
            error_gap = actual - predicted
            
            prompt = f"""
            SYSTEM ALERT: PREDICTION MISMATCH DETECTED.
            Initialize Autonomous Self-Correction Protocol.
            
            Data:
            - Context: {context}
            - AI Predicted Wait: {predicted} min
            - Actual Reported Wait: {actual} min
            - Error Gap: {error_gap} min
            
            Task:
            1. Analyze why the prediction failed (e.g., did we underestimate the 'Technical Issue' impact?).
            2. Generate a 'Weight Adjustment' strategy for future predictions.
            3. Provide a brief log entry for the system database.
            
            Output Format:
            **🔴 Error Analysis:** [Why the gap occurred]
            **⚙️ Parameter Tuning:** [How the agent is adjusting its logic]
            **✅ System Update:** "Model weights updated for {context} scenarios."
            """
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Self-Correction Failed: {str(e)}"





