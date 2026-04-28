import gradio as gr
import joblib
import numpy as np
from pathlib import Path
import sys

cd = Path(sys.argv[0]).resolve().parent

rf_model = joblib.load(cd/"rf_model.pkl")
logreg_model = joblib.load(cd/"logreg_model.pkl")

def predict_stroke(model_choice, gender, age, hypertension, heart_disease, 
                   residence_type, work_type, avg_glucose_level, bmi):
    
    ## Data Validation ----------------------
    if any(v is None or v == "" for v in [gender, age, hypertension, heart_disease, 
                                residence_type, work_type, avg_glucose_level, bmi]):
        return "⚠️ Please fill in all fields to get a prediction."
    try:
        age = float(age)
        avg_glucose_level = float(avg_glucose_level)
        bmi = float(bmi)
    except ValueError:
        return "⚠️ Please enter valid numbers for Age, Glucose Level, and BMI."
    

    ## WARNINGS -------------------------------
    warnings = []
    
    if age < 2:
        warnings.append("⚠️ **Limited training data for infants under age 2.** Predictions may be less reliable.")
        
        # BMI for infants is very different from adults
        if bmi > 19 or bmi < 12.5:
            warnings.append("⚠️ **Unusual BMI for infant.** Typical infant BMI: ~18. Please verify input.")
        
        # Glucose levels for infants
        if avg_glucose_level > 150:
            warnings.append("⚠️ **High glucose level for infant.** Please consult a pediatrician immediately.")
    
    # General outlier detection
    if age >= 2 and bmi > 50 or bmi < 15:
        warnings.append("⚠️ **Extreme BMI value.** Model trained primarily on BMI 15-40.")
    
    if avg_glucose_level > 160:
        warnings.append("⚠️ **Critically high glucose level.** Please consider seeking medical attention immediately.")
    
    if age < 18:
        warnings.append("⚠️ **Pediatric case**: Model primarily trained on adult data. Use with caution.")
    if age < 18 and bmi > 35:
        warnings.append("⚠️ **Limited pediatric obesity data in training set. (high BMI & age <18)**")
                        

    #-------------------

    # Encoding to match training
    gender_encoded = {'Male': 0, 'Female': 1, 'Other': -1}[gender]
    residence_encoded = {'Rural': 0, 'Urban': 1}[residence_type]
    work_encoded = {'Private': 0, 'Self-employed': 1, 'Government job': 2, 
                    'Child': -1, 'Never worked': -2}[work_type]
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    

    input_data = np.array([[gender_encoded, age, hypertension_encoded, 
                           heart_disease_encoded, work_encoded, 
                           avg_glucose_level, bmi]])
    
    if model_choice == "Model 1 (Random Forest)":
        model = rf_model
    else:
        model = logreg_model
    
    # prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]+0.25  # Probability of stroke (class 1)
    

    result = f"""
    Based off your data, we predict you have a **{proba:.1%} chance** of getting a stroke in the future!
    """

     # Add warnings if any
    if warnings:
        result += "\n\n---\n\n**Important Warnings:**\n\n"
        result += "\n\n".join(warnings)
    
    # General disclaimer for extreme cases
    if bmi > 35 or avg_glucose_level > 200 or age < 18:
        result += "\n\n**Clinical Review Recommended**: Your statistics suggest that the prediction should be reviewed by a healthcare professional, as the model may be inaccurate. (You might be getting this warning because you are below 18, your average glucose level is >200, or your BMI is above 35.)"
    
    return result



interface = gr.Interface(
    fn=predict_stroke,
    inputs=[
        gr.Radio(["Model 1 (Random Forest)", "Model 2 (Logistic Regression)"], 
                 label="Select Model", 
                 value="Model 1 (Random Forest)"),
        gr.Dropdown(["Male", "Female", "Other"], 
                    value = "",
                    label="Gender"),
        gr.Number(label="Age", 
                #   placeholder="e.g., 50",
                  minimum=0, 
                  value="",
                  maximum=120,
                  info="Whole number for ages ≥2, decimal for ages 0-2 (eg. 1.78 or 36)"),
        gr.Radio(["Yes", "No"], 
                 label="Hypertension",
                 info="Does the patient have hypertension?"),
        gr.Radio(["Yes", "No"], 
                 label="Heart Disease",
                 info="Does the patient have any heart diseases?"),
        gr.Dropdown(["Rural", "Urban"], 
                    label="Residence Type",
                    value = ""),
        gr.Dropdown(["Private", "Self-employed", "Government job", "Never worked", "Child"], 
                    label="Work Type",
                    value = ""),
        gr.Number(label="Average Glucose Level (obtain information from health provider.)", 
                #   placeholder="e.g., 100",
                  minimum=0,
                  value = "",
                  info="Decimal number up to 2 decimal points (eg. 100.25)"),
        gr.Number(label="BMI (Body Mass Index)", 
                #   placeholder="e.g., 25",
                  minimum=0,
                  value = "",
                  info="Decimal number up to 1 decimal point (eg. 25.1)")
    ],
    outputs=gr.Markdown(label="Output"),
    title="Stroke Risk Prediction Interface",
    description="Enter patient information to predict stroke risk using machine learning. The AI can make mistakes, so please check information with medical professionals. Use this as a ***predictor***, not professional advice.",
    # examples=[
    #     ["Model 1 (Random Forest)", "Male", 67, "Yes", "Yes", "Urban", "Private", 228.69, 36.6],
    #     ["Model 2 (Logistic Regression)", "Female", 45, "No", "No", "Rural", "Self-employed", 95.12, 28.5],
    # ],
    # css=custom_css,
    # theme=mod_theme,
    theme=gr.themes.Glass(),
)

interface.launch(inbrowser=True)