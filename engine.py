import joblib
import pandas as pd
import os
import google.generativeai as genai
import PIL.Image
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

class IntelliQueueEngine:
    """
    Core reasoning engine for IntelliQueue AI.
    Handles Hybrid Logic: Traditional ML (Random Forest) + Generative AI (Gemini 1.5 Flash).
    """

    def __init__(self):
        # 1. Configure Google Gemini API
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        # 2. Load the Pre-trained ML Model (with fallback safety)
        try:
            self.model = joblib.load('models/queue_model.pkl')
        except Exception as e:
            print(f"Warning: ML Model not found. Using mathematical fallback. Error: {e}")
            self.model = None 

    def get_prediction(self, hour, day, staff, crowd, context="Normal Day", image_file=None):
        """
        Generates wait time prediction and autonomous reasoning.
        
        Args:
            hour (int): Hour of the day (0-23).
            day (int): Day of the week (0=Mon, 6=Sun).
            staff (int): Number of active staff members.
            crowd (int): Estimated number of people in queue.
            context (str): Environmental factor (e.g., "Rainy Weather").
            image_file (PIL.Image): Optional CCTV snapshot for visual analysis.

        Returns:
            tuple: (final_wait_time, reasoning_text)
        """
        
        # --- Step 1: Base Calculation (Quantitative) ---
        base_wait = 5.0  # Default baseline
        
        # Try using the Random Forest Regressor
        if self.model:
            try:
                input_data = pd.DataFrame([[hour, day, staff, crowd]], 
                                         columns=['hour_of_day', 'day_of_week', 'current_staff', 'current_crowd'])
                base_wait = self.model.predict(input_data)[0]
            except:
                # Fallback: Simple heuristic formula if model fails
                base_wait = (crowd / (staff + 1)) * 4

        # --- Step 2: Contextual Multipliers (Rule-based Logic) ---
        # Adjusting math based on environmental stress factors
        if context == "Technical Issue":
            base_wait = (base_wait * 2.0) + 10.0 # Critical failure doubles wait time
        elif context == "Staff Shortage":
            base_wait = base_wait * 1.3 # Inefficiency factor
        elif context == "Rainy Weather":
            base_wait = base_wait * 1.1 # Slower movement factor
        elif context == "Holiday Rush":
            base_wait = base_wait * 1.5 # High volume factor
            
        final_wait_time = round(base_wait, 1)

        # --- Step 3: Gemini Multimodal Reasoning (Qualitative) ---
        reasoning = "AI analysis unavailable."
        
        try:
            # Using Gemini Flash Lite Latest for speed and multimodal capabilities
            model = genai.GenerativeModel('gemini-flash-lite-latest')
            
            # Constructing the System Prompt
            # We explicitly frame this as a "Recorded Snapshot" analysis to handle time simulation
            prompt_text = f"""
            Act as an Autonomous Retail Reasoning Engine (Gemini 3 Action Era).
            
            Situation Context:
            You are analyzing a specific timeframe to assist store managers.
            - Target Time of Analysis: {hour}:00 Hours (24h format)
            - Staff Active: {staff}
            - Predicted Wait: {final_wait_time} mins
            - Environmental Context: {context}
            """

            content_input = [prompt_text]

            # Logic: If an image is provided, analyze it as evidence
            if image_file:
                prompt_text += """
                \n[VISUAL EVIDENCE PROVIDED]
                A CCTV Snapshot corresponding to the Target Time ({hour}:00) is attached.
                1. Analyze the crowd density and mood in the image.
                2. Verify if the visual crowd matches the data input ({crowd} people).
                3. Treat this image as the ground truth for that specific time.
                """
                content_input.append(image_file) # Append image object to request
            else:
                prompt_text += "\nNo visual input provided. Base analysis on data parameters only."

            prompt_text += """
            \nOutput Format:
            ### 👁️ Visual & Temporal Analysis
            (Confirm if the scene looks consistent with the reported context at {hour}:00).

            ### 🧠 Root Cause Diagnosis
            Explain the causal link between the {context} and the {final_wait_time} min wait.

            ### 📋 Autonomous Action Plan
            - **Immediate Directive (P1):** Action for {hour}:05.
            - **Strategic Adjustment (P2):** Staffing change for the next hour.
            """
            
            # Update the text prompt in the list
            content_input[0] = prompt_text

            # Execute API Call
            response = model.generate_content(content_input)
            reasoning = response.text
            
        except Exception as e:
            reasoning = f"⚠️ AI Analysis Failed: {str(e)}. ML Prediction remains valid."

        return final_wait_time, reasoning







