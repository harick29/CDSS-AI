import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os

print("Starting AI Assistant...")

# Make sure the model exists
if not os.path.exists('skin_model.h5'):
    print("Model not found! Please run train_model.py first.")
    exit()

print("Loading trained AI model...")
model = tf.keras.models.load_model('skin_model.h5')

def analyze_image(img):
    if img is None:
        return "Error", "No scanner image provided."
        
    # Preprocess image to match training
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    
    # AI Prediction
    prediction = model.predict(img)[0][0]
    
    if prediction > 0.5:
        # Malignant classification rules
        diagnosis = f"🔴 MALIGNANT (AI Confidence: {prediction*100:.1f}%)"
        rules = """🚨 URGENT ACTION REQUIRED: MALIGNANT DIAGNOSIS 🚨

Doctor Instructions:
1. Schedule an immediate biopsy to confirm pathology.
2. Order complete blood work (CBC, Comprehensive Metabolic Panel).
3. Patient must avoid direct sunlight and UV exposure.

Drug Prescription & Dosage Rules:
- Primary Drug (Chemotherapeutic Agent-X): Administer 50mg IV weekly.
- Pre-Medication (Anti-nausea Z): Prescribe 10mg taken orally 1 hour before Agent-X.
- Pain Management (Drug-P): 400mg every 6 hours as needed for pain.

Follow-up Protocol:
- Patient needs to be checked by the oncology department every 7 days for the next month."""
    else:
        # Benign classification rules
        diagnosis = f"🟢 BENIGN (AI Confidence: {(1-prediction)*100:.1f}%)"
        rules = """✅ STATUS: BENIGN DIAGNOSIS

Doctor Instructions:
1. No immediate surgical action or biopsy required.
2. Photograph and measure the affected area for future comparison.

Drug Prescription & Dosage Rules:
- No aggressive drugs required.
- Inflammation Care: Prescribe Topical Hydrocortisone 1% cream, apply a thin layer twice daily for 7 days if inflamed.
- Allergy Relief: 10mg Cetirizine daily if itching occurs.

Follow-up Protocol:
- Routine dermatological check-up in 6 months. Instruct patient to return immediately if the spot changes color or size."""
        
    return diagnosis, rules

# Build the Web Interface
print("Building Doctor Interface...")
interface = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(),
    outputs=[
        gr.Textbox(label="🔍 AI Diagnosis Result"), 
        gr.Textbox(label="📋 Doctor Rules, Instructions & Drug Prescriptions", lines=15)
    ],
    title="⚕️ AI Dermatologist & Prescription Assistant",
    description="Upload a patient's skin image. The AI will analyze it to provide a benign/malignant diagnosis and immediately generate the specific rules and drug dosages you (the Doctor) must administer."
)

print("Interface ready! Launching...")
interface.launch()
