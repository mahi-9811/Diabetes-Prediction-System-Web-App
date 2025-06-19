import os
import streamlit as st
import pandas as pd
import joblib
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from fpdf import FPDF

# Create reports folder if doesn't exist
os.makedirs("reports", exist_ok=True)

# Load model, scaler, and explainer (update paths as needed)
model = joblib.load("model/random_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")
explainer = joblib.load("shap/explainer.pkl")

st.title("🩺 AI-Based Diabetes Prediction System")
st.markdown("Enter patient details in the sidebar to predict diabetes risk.")

# Sidebar input form
def get_user_input():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 140, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 900, 79)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0, format="%.1f")
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, format="%.3f")
    age = st.sidebar.slider("Age", 1, 120, 33)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()
input_scaled = scaler.transform(input_df)

# Prediction
pred = model.predict(input_scaled)[0]
prob = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")
status = "Diabetic" if pred else "Non-Diabetic"
color = "red" if pred else "green"
st.markdown(f"<h3 style='color:{color}'>{status}</h3>", unsafe_allow_html=True)
st.write(f"Probability of Diabetes: `{prob:.2f}`")

# Get SHAP values for input
shap_values = explainer.shap_values(input_scaled)

# Debug prints for expected_value and shap_values


# Handling multi-class / binary classification
if isinstance(shap_values, list):
    base_value = explainer.expected_value[1]
    shap_val = shap_values[1][0]  # SHAP values for sample 0, positive class
else:
    base_value = explainer.expected_value
    shap_val = shap_values[0]

# Generate SHAP force plot
# Choose class index = 1 (usually the "diabetic" class)
class_idx = 1
base_value = explainer.expected_value[class_idx]
shap_val = shap_values[0, :, class_idx]  # shape (8,)

force_plot_html = shap.plots.force(
    base_value,
    shap_val,
    features=input_df.iloc[0],
    feature_names=input_df.columns.tolist()
)


st.subheader("SHAP Explanation (Force Plot)")
components.html(f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>", height=400)

# Function to save static SHAP force plot image (for PDF report)
def save_shap_force_plot_image(base_value, shap_val, input_df):
    plt.figure(figsize=(8, 3))
    shap.force_plot(base_value, shap_val, input_df.iloc[0], matplotlib=True, show=False)
    image_path = "reports/force_plot.png"
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()
    return image_path

# PDF report generation
def generate_pdf(patient_data, pred, prob):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Diabetes Prediction Report", ln=True, align="C")

    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Patient Health Parameters:", ln=True)
    for k, v in patient_data.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    pdf.ln(5)
    result_text = "Diabetic" if pred else "Non-Diabetic"
    pdf.set_text_color(255, 0, 0) if pred else pdf.set_text_color(0, 128, 0)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Prediction: {result_text}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Probability: {prob:.2f}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "SHAP Feature Contribution (see plot):", ln=True)

    # Add static SHAP image
    image_path = save_shap_force_plot_image(base_value, shap_val, input_df)
    pdf.image(image_path, x=10, y=pdf.get_y() + 5, w=180)
   
    file_path = "reports/diabetes_report.pdf"
    pdf.output(file_path)
    return file_path

if st.button("Generate PDF Report"):
    file_path = generate_pdf(input_df.iloc[0].to_dict(), pred, prob)
    with open(file_path, "rb") as f:
        st.download_button("📥 Download Report", f, file_name="diabetes_report.pdf")

