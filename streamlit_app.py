import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import json
import shap
import streamlit.components.v1 as components
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="CKD Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Gemini API Configuration ---
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"
    if API_KEY else ""
)

# --- Gemini API Call Function (with Retries) ---
def call_gemini_api(system_prompt, user_prompt):
    """
    Calls the Gemini API with a system prompt and a user prompt.
    Includes exponential backoff retry logic.
    """
    # Check if API key is provided
    if not API_KEY:
        return """
        **⚠️ AI Features Disabled**
        
        The AI-powered features (Diet Planner and Chatbot) require a Gemini API key.
        
        To enable these features:
        1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Add it to the `API_KEY` variable in the code
        3. Restart the app
        
        **Alternative:** You can still use the ML prediction features without AI assistance.
        """
    
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 2048,
        }
    }
    
    max_retries = 5
    delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(GEMINI_API_URL, json=payload, headers={'Content-Type': 'application/json'})
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates'][0]['content']['parts'][0]['text']:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "Error: The API response was not in the expected format. Check safety filters."
            elif response.status_code == 429:
                # Throttling
                st.toast(f"API Throttled. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2 # Exponential backoff
            else:
                return f"API Error: {response.status_code}\n{response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {e}"
        except json.JSONDecodeError as e:
            return f"Error decoding JSON response: {e}\nResponse received: {response.text}"
        except Exception as e:
            return f"Unexpected error: {e}"
            
    return "Error: Maximum API retry attempts exceeded."


# --- Load Models and Columns ---
@st.cache_resource
def load_all_models():
    """
    Loads all necessary models and artefacts into the cache.
    """
    models = {}
    try:
        models['tabular_model'] = joblib.load('tabular_model.joblib')
        models['cnn_model'] = load_model('cnn_model.h5')
        with open('model_columns.pkl', 'rb') as f:
            models['model_columns'] = pickle.load(f)
        
        # Create the SHAP explainer for the tabular model
        # Check if the model is a pipeline or a direct classifier
        model_pipeline = models['tabular_model']
        
        if hasattr(model_pipeline, 'named_steps'):
            # It's a pipeline - extract classifier and preprocessor
            classifier = model_pipeline.named_steps['classifier']
            preprocessor = model_pipeline.named_steps['preprocessor']
        else:
            # It's a direct classifier (like RandomForestClassifier)
            classifier = model_pipeline
            preprocessor = None
        
        # Create SHAP explainer for tree-based models
        if hasattr(classifier, 'predict_proba') and hasattr(classifier, 'estimators_'):
            # Use TreeExplainer for tree-based models (Random Forest, etc.)
            models['shap_explainer'] = shap.TreeExplainer(classifier)
            models['shap_preprocessor'] = preprocessor
        else:
            models['shap_explainer'] = None
            models['shap_preprocessor'] = None
            
    except FileNotFoundError as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.sidebar.info("Please ensure 'tabular_model.joblib', 'model_columns.pkl', and 'cnn_model.h5' are in the same directory as the app.")
        return None
    except Exception as e:
        st.sidebar.error(f"Unexpected loading error: {e}")
        return None
    return models

models = load_all_models()

# --- CNN Image Preprocessing Function ---
def preprocess_image(image):
    """
    Preprocesses the uploaded image to match the CNN model's input.
    """
    img = image.resize((150, 150)) # Adjust to model's input size (150x150)
    img = img.convert('RGB') # Ensure 3 channels
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch of 1
    img_array /= 255.0 # Rescale (as done in training)
    return img_array

# --- Function to display SHAP plot ---
def st_shap(plot, height=None):
    """
    Helper function to display SHAP plots in Streamlit.
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=True)

# --- Navigation Sidebar ---
st.sidebar.title("🩺 CKD Prediction Dashboard")
st.sidebar.markdown("Navigate to the different sections of the app.")

page = st.sidebar.radio(
    "Select a page:",
    [
        "🏠 Home",
        "📊 Prediction (Tabular Data)",
        "🖼️ Prediction (CT Scan)",
        "🥗 AI Diet Planner",
        "🤖 AI Explanatory Chatbot"
    ]
)

# --- Page: Home ---
if page == "🏠 Home":
    st.title("Welcome to the CKD Prediction Dashboard")
    st.markdown("""
    This application uses Machine Learning and Generative AI to help you:
    
    1.  **Predict the risk of Chronic Kidney Disease (CKD)** using tabular (clinical) data or CT scan images.
    2.  **Understand the predictions** through an explanatory chatbot and SHAP visualisations.
    3.  **Get a sample diet plan** generated by AI based on your status.

    **Important:** This tool is for informational and educational purposes only. **It does not substitute professional medical advice, diagnosis, or treatment.** Always consult your doctor with any questions regarding your health.

    ### How to Start
    
    * Use the menu on the left to navigate.
    * **`Prediction (Tabular Data)`**: Enter your clinical values manually.
    * **`Prediction (CT Scan)`**: Upload an image of a renal CT scan.
    * **`AI Diet Planner`**: Get an AI-generated meal plan.
    * **`AI Explanatory Chatbot`**: Ask questions about CKD and the results.
    """)
    st.image("https://res.cloudinary.com/dgna7ajrl/image/upload/v1761511750/img-kidney-care_mvurfn.jpg", use_column_width=True)

# --- Page: Prediction (Tabular Data) ---
elif page == "📊 Prediction (Tabular Data)":
    st.title("📊 CKD Prediction with Tabular Data")
    st.markdown("Enter the clinical (manual) values in the form below. This form is based on the 'kidney_disease.csv' dataset.")

    if not models:
        st.error("Models are not loaded. Prediction is unavailable.")
    else:
        model_columns = models['model_columns']
        
        # Create a form for data entry
        with st.form("tabular_input_form"):
            st.subheader("Patient Data Form")
            
            # Create columns for a cleaner layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=1, max_value=100, value=50)
                bp = st.number_input("Blood Pressure (bp)", min_value=40, max_value=200, value=80, help="Diastolic, in mm/Hg")
                sg = st.selectbox("Specific Gravity (sg)", [1.005, 1.010, 1.015, 1.020, 1.025], index=2)
                al = st.selectbox("Albumin (al)", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=0)
                su = st.selectbox("Sugar (su)", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=0)
                rbc = st.selectbox("Red Blood Cells (rbc)", ["normal", "abnormal"], index=0)
                pc = st.selectbox("Pus Cell (pc)", ["normal", "abnormal"], index=0)
                pcc = st.selectbox("Pus Cell Clumps (pcc)", ["notpresent", "present"], index=0)
                
            with col2:
                ba = st.selectbox("Bacteria (ba)", ["notpresent", "present"], index=0)
                bgr = st.number_input("Blood Glucose Random (bgr)", min_value=50, max_value=500, value=120, help="in mgs/dl")
                bu = st.number_input("Blood Urea (bu)", min_value=1.0, max_value=400.0, value=36.0, help="in mgs/dl")
                sc = st.number_input("Serum Creatinine (sc)", min_value=0.1, max_value=80.0, value=1.2, help="in mgs/dl")
                sod = st.number_input("Sodium (sod)", min_value=100.0, max_value=180.0, value=138.0, help="in mEq/L")
                pot = st.number_input("Potassium (pot)", min_value=2.0, max_value=10.0, value=4.5, help="in mEq/L")
                hemo = st.number_input("Haemoglobin (hemo)", min_value=3.0, max_value=20.0, value=15.0, help="in gms")
                pcv = st.number_input("Packed Cell Volume (pcv)", min_value=10, max_value=60, value=44)

            with col3:
                wc = st.number_input("White Blood Cell Count (wc)", min_value=2000, max_value=30000, value=7800, help="in cells/cumm")
                rc = st.number_input("Red Blood Cell Count (rc)", min_value=2.0, max_value=9.0, value=5.2, help="in millions/cmm")
                htn = st.selectbox("Hypertension (htn)", ["no", "yes"], index=0)
                dm = st.selectbox("Diabetes Mellitus (dm)", ["no", "yes"], index=0)
                cad = st.selectbox("Coronary Artery Disease (cad)", ["no", "yes"], index=0)
                appet = st.selectbox("Appetite (appet)", ["good", "poor"], index=0)
                pe = st.selectbox("Paedal Edema (pe)", ["no", "yes"], index=0)
                ane = st.selectbox("Anaemia (ane)", ["no", "yes"], index=0)

            submitted = st.form_submit_button("Analyse and Predict", type="primary", use_container_width=True)

        if submitted:
            with st.spinner("Processing and predicting..."):
                try:
                    # Create the input dictionary from the form values
                    input_data = {
                        'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
                        'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba,
                        'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod,
                        'pot': pot, 'hemo': hemo, 'pcv': pcv,
                        'wc': wc, 'rc': rc, 'htn': htn,
                        'dm': dm, 'cad': cad, 'appet': appet,
                        'pe': pe, 'ane': ane
                    }
                    
                    # Convert to DataFrame
                    input_df = pd.DataFrame([input_data])
                    
                    # Create one-hot encoded features to match model expectations
                    # Initialize all expected columns with 0
                    model_data = {}
                    for col in model_columns:
                        model_data[col] = 0
                    
                    # Set numerical values
                    model_data['age'] = age
                    model_data['bp'] = bp
                    model_data['sg'] = sg
                    model_data['al'] = al
                    model_data['su'] = su
                    model_data['bgr'] = bgr
                    model_data['bu'] = bu
                    model_data['sc'] = sc
                    model_data['sod'] = sod
                    model_data['pot'] = pot
                    model_data['hemo'] = hemo
                    model_data['pcv'] = pcv
                    model_data['wc'] = wc
                    model_data['rc'] = rc
                    
                    # Set one-hot encoded categorical values
                    if rbc == 'normal':
                        model_data['rbc_normal'] = 1
                    if pc == 'normal':
                        model_data['pc_normal'] = 1
                    if pcc == 'present':
                        model_data['pcc_present'] = 1
                    if ba == 'present':
                        model_data['ba_present'] = 1
                    if htn == 'yes':
                        model_data['htn_yes'] = 1
                    if dm == 'yes':
                        model_data['dm_yes'] = 1
                    if cad == 'yes':
                        model_data['cad_yes'] = 1
                    if appet == 'poor':
                        model_data['appet_poor'] = 1
                    if pe == 'yes':
                        model_data['pe_yes'] = 1
                    if ane == 'yes':
                        model_data['ane_yes'] = 1
                    
                    # Convert to DataFrame with correct column order
                    input_df = pd.DataFrame([model_data])
                    input_df = input_df[model_columns] 
                    
                    # Perform the prediction
                    model_pipeline = models['tabular_model']
                    prediction = model_pipeline.predict(input_df)
                    prediction_proba = model_pipeline.predict_proba(input_df)
                    
                    # Display results
                    st.subheader("Prediction Result")
                    result_class = prediction[0]
                    # Find the index of the 'ckd' class
                    ckd_class_index = list(model_pipeline.classes_).index('ckd')
                    proba_ckd = prediction_proba[0][ckd_class_index]
                    
                    if result_class == 'ckd':
                        st.error(f"**Prediction: High Risk of CKD** (Probability: {proba_ckd:.2%})", icon="⚠️")
                    else:
                        st.success(f"**Prediction: Low Risk of CKD** (Probability: {1-proba_ckd:.2%})", icon="✅")

                    st.markdown("---")
                    
                    # --- SHAP Explainability ---
                    st.subheader("Prediction Explanation (SHAP)")
                    st.markdown("""
                    This chart shows which factors (features) had the biggest impact on this specific prediction.
                    - **Red arrows** (positive forces) push the prediction towards "High Risk (CKD)".
                    - **Blue arrows** (negative forces) push the prediction towards "Low Risk (notckd)".
                    """)
                    
                    if models['shap_explainer']:
                        with st.spinner("Generating SHAP explanation..."):
                            try:
                                # 1. Prepare the data for SHAP
                                if models['shap_preprocessor']:
                                    # If there's a preprocessor, use it
                                    input_processed = models['shap_preprocessor'].transform(input_df)
                                    feature_names = models['shap_preprocessor'].get_feature_names_out()
                                else:
                                    # If no preprocessor, use the data directly
                                    input_processed = input_df.values
                                    feature_names = input_df.columns.tolist()
                                
                                # Get SHAP values first to determine the correct feature names
                                shap_values = models['shap_explainer'].shap_values(input_processed)
                                
                                # Get SHAP values first to determine the correct number of features
                                if isinstance(shap_values, list):
                                    # Binary classification returns a list [shap_val_class_0, shap_val_class_1]
                                    if ckd_class_index < len(shap_values):
                                        shap_vals_temp = shap_values[ckd_class_index]
                                    else:
                                        shap_vals_temp = shap_values[0]
                                else:
                                    shap_vals_temp = shap_values
                                
                                # Flatten to get the actual number of features
                                if shap_vals_temp.ndim > 1:
                                    shap_vals_temp = shap_vals_temp[0, :]
                                shap_vals_temp = shap_vals_temp.flatten()
                                
                                # Create feature names based on the actual SHAP values length
                                num_shap_features = len(shap_vals_temp)
                                shap_feature_names = [f"feature_{i}" for i in range(num_shap_features)]
                                st.info(f"Using generated feature names: {len(shap_feature_names)} features to match {num_shap_features} SHAP values")
                                
                                # 3. Handle different SHAP value formats and debug info
                                st.info(f"Debug: ckd_class_index = {ckd_class_index}, model classes = {model_pipeline.classes_}")
                                
                                # Properly extract and flatten SHAP values
                                if isinstance(shap_values, list):
                                    # Binary classification returns a list [shap_val_class_0, shap_val_class_1]
                                    if ckd_class_index < len(shap_values):
                                        shap_vals = shap_values[ckd_class_index]
                                    else:
                                        # Fallback to first class if index is out of bounds
                                        shap_vals = shap_values[0]
                                        st.info("Using first class SHAP values as fallback")
                                else:
                                    # Single array format
                                    shap_vals = shap_values
                                
                                # Ensure shap_vals is 1D array
                                if shap_vals.ndim > 1:
                                    shap_vals = shap_vals[0, :]  # Take first row if 2D
                                
                                # Flatten to ensure 1D
                                shap_vals = shap_vals.flatten()
                                
                                st.info(f"Debug: SHAP values shape = {shap_vals.shape}, SHAP feature names length = {len(shap_feature_names)}")
                                
                                # Use the correct feature names for SHAP
                                final_feature_names = shap_feature_names
                                
                                # 4. Ensure dimensions match
                                if len(shap_vals) != len(final_feature_names):
                                    st.warning(f"SHAP dimension mismatch: {len(shap_vals)} values vs {len(final_feature_names)} features. Skipping SHAP plot.")
                                else:
                                    # Create the force plot with simplified expected value handling
                                    try:
                                        expected_val = models['shap_explainer'].expected_value
                                        
                                        # Handle different expected value formats
                                        if isinstance(expected_val, (list, np.ndarray)):
                                            if len(expected_val) > ckd_class_index:
                                                exp_val = expected_val[ckd_class_index]
                                            else:
                                                exp_val = expected_val[0]  # Use first value as fallback
                                        else:
                                            exp_val = expected_val
                                        
                                        # Create a simple bar plot instead of force plot to avoid complexity
                                        st.subheader("Feature Importance (SHAP Values)")
                                        
                                        # Create a DataFrame for easier visualization
                                        # Ensure both arrays are the same length and 1D
                                        min_len = min(len(final_feature_names), len(shap_vals))
                                        feature_importance_df = pd.DataFrame({
                                            'Feature': final_feature_names[:min_len],
                                            'SHAP_Value': shap_vals[:min_len]
                                        }).sort_values('SHAP_Value', key=abs, ascending=False)
                                        
                                        # Display as a bar chart
                                        st.bar_chart(feature_importance_df.set_index('Feature')['SHAP_Value'])
                                        
                                        # Show the top features in a table
                                        st.subheader("Top Contributing Features")
                                        st.dataframe(feature_importance_df.head(10), use_container_width=True)
                                        
                                    except Exception as plot_error:
                                        st.warning(f"SHAP visualization failed: {plot_error}. Using alternative display.")
                                        
                                        # Fallback: Show SHAP values as a simple table
                                        st.subheader("Feature Contributions (SHAP Values)")
                                        min_len = min(len(final_feature_names), len(shap_vals))
                                        feature_importance_df = pd.DataFrame({
                                            'Feature': final_feature_names[:min_len],
                                            'SHAP_Value': shap_vals[:min_len]
                                        }).sort_values('SHAP_Value', key=abs, ascending=False)
                                        st.dataframe(feature_importance_df, use_container_width=True)
                            except Exception as e:
                                st.warning(f"SHAP explanation failed: {e}. The prediction is still valid.")
                    else:
                        st.warning("SHAP explainer is not available for this model.")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.exception(e)

# --- Page: Prediction (CT Scan) ---
elif page == "🖼️ Prediction (CT Scan)":
    st.title("🖼️ Prediction with CT Scan")
    st.markdown("Upload an image of a renal CT scan to analyse it for 'Cyst', 'Normal', 'Stone', or 'Tumor'.")

    # IMPORTANT: Define the class names in the *exact same order* as your model was trained.
    # Check your 'train_generator.class_indices' output from Colab.
    # Example: {'cyst': 0, 'normal': 1, 'stone': 2, 'tumor': 3}
    CLASS_NAMES_CNN = ['cyst', 'normal', 'stone', 'tumor']
    st.info(f"This model predicts the following classes: **{', '.join(CLASS_NAMES_CNN)}**")


    if not models:
        st.error("The CNN model is not loaded. Prediction is unavailable.")
    else:
        uploaded_file = st.file_uploader("Choose a CT scan image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded CT Scan", use_column_width=False, width=300)
            
            if st.button("Analyse Image", type="primary", use_container_width=True):
                with st.spinner("Processing image and predicting..."):
                    try:
                        # Preprocess the image
                        processed_image = preprocess_image(image)
                        
                        # Perform the prediction
                        cnn_model = models['cnn_model']
                        prediction = cnn_model.predict(processed_image)
                        
                        # Get the index of the highest probability class
                        predicted_class_index = np.argmax(prediction[0])
                        class_name = CLASS_NAMES_CNN[predicted_class_index]
                        probability = prediction[0][predicted_class_index]
                        
                        st.subheader("Prediction Result")
                        if class_name == 'normal':
                            st.success(f"**Prediction: Normal** (Confidence: {probability:.2%})", icon="✅")
                        else:
                            st.error(f"**Prediction: {class_name.title()} Detected** (Confidence: {probability:.2%})", icon="⚠️")
                            
                        st.markdown(f"*(Raw model output: {prediction[0]})*")
                        
                        st.info("""
                        **Image Explainability (Grad-CAM):**
                        Implementing live Grad-CAM is complex. An alternative is:
                        1.  Use the **AI Explanatory Chatbot** to ask what these findings mean.
                        2.  Implement a library like `tf-explain` in the backend if deployed on a more powerful server.
                        """)
                        
                    except Exception as e:
                        st.error(f"Error during image prediction: {e}")
                        st.exception(e)

# --- Page: AI Diet Planner ---
elif page == "🥗 AI Diet Planner":
    st.title("🥗 AI-Assisted Diet Planner")
    st.markdown("Generate a sample diet plan based on CKD status. Always consult a nutritionist.")
    
    # Input options
    ckd_stage = st.selectbox(
        "Select the CKD status or stage:",
        ("Low Risk / Normal", "High Risk / Early Stage (1-2)", "Moderate Stage (3)", "Advanced Stage (4-5, no dialysis)", "Patient on Dialysis")
    )
    
    preferences = st.text_input("Preferences or allergies (optional):", placeholder="e.g., vegetarian, peanut allergy")
    
    if st.button("Generate 3-Day Diet Plan", type="primary", use_container_width=True):
        with st.spinner("🤖 The AI is thinking of a delicious and healthy plan..."):
            
            # --- System Prompt for the Dietitian ---
            system_prompt = """
            You are an expert renal dietitian. Your task is to create a sample 3-day meal plan for a patient with Chronic Kidney Disease (CKD).
            You must follow these rules STRICTLY:
            1.  **Focus on the CKD stage:** Adjust sodium, potassium, phosphorus, and protein based on the user-provided stage.
            2.  **Low Sodium:** Always low in sodium (<2000mg/day).
            3.  **Potassium/Phosphorus Control:** Restrict for advanced stages and dialysis. Be more flexible for early stages.
            4.  **Protein:** Adjust protein (lower for pre-dialysis stages 4-5, higher for dialysis, moderate for stages 1-3).
            5.  **Format:** Use Markdown. Be clear and concise (Breakfast, Lunch, Dinner, 1 Snack per day).
            6.  **Disclaimer:** You MUST include a disclaimer at the end: "This is a sample plan and does not replace advice from a doctor or registered dietitian."
            """
            
            # --- User Prompt ---
            user_prompt = f"""
            Generate a 3-day meal plan for a patient in the following state: "{ckd_stage}".
            Take these preferences into account: "{preferences if preferences else 'none'}".
            """
            
            # --- API Call ---
            response_text = call_gemini_api(system_prompt, user_prompt)
            
            st.markdown(response_text)

# --- Page: AI Explanatory Chatbot ---
elif page == "🤖 AI Explanatory Chatbot":
    st.title("🤖 AI Explanatory Chatbot")
    st.markdown("Ask questions about CKD, your test results, or risk factors.")
    st.info("**Disclaimer:** This chatbot is an AI assistant and not a medical professional. Its advice does not constitute a diagnosis. Please consult a doctor.")

    # --- System Prompt for the Chatbot ---
    SYSTEM_PROMPT_CHATBOT = """
    You are a friendly and empathetic AI assistant in a renal health app. Your name is "RenalBot".
    Your purpose is to explain complex medical concepts about Chronic Kidney Disease (CKD) in VERY SIMPLE terms.
    
    STRICT RULES:
    1.  **DO NOT DIAGNOSE:** Never say "You have CKD" or "Your result means you are ill". Instead, speak in general terms.
        - BAD: "Your creatinine of 2.5 is high, you have CKD."
        - GOOD: "A serum creatinine level of 2.5 mg/dL is generally considered elevated. High levels can indicate that the kidneys are not filtering blood as efficiently as they should. It is very important to discuss this specific result with your doctor."
    2.  **ALWAYS ADVISE A DOCTOR:** End ANY question about test results or symptoms by advising the user to speak with their doctor.
    3.  **BE SIMPLE:** Avoid medical jargon. Explain things like "creatinine", "glomerular filtration rate (GFR)", "proteinuria".
    4.  **USE MARKDOWN:** Format your answers for easy reading (lists, bold text).
    """

    # --- Initialise chat history ---
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I'm RenalBot. I'm here to help you understand kidney health better. What would you like to ask?"}
        ]

    # --- Display chat history messages ---
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- User input field ---
    if prompt := st.chat_input("Write your question here... (e.g., 'What is creatinine?')"):
        # Add user message to history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤖 RenalBot is thinking..."):
                # Create the content history for the API
                # The Gemini v1beta API doesn't have a direct 'role' in the history,
                # so we will simulate a single user prompt with context.
                # For a true chatbot, you would send the 'contents' history.
                
                # For this model, a simple approach is to send only the latest prompt
                # with the system prompt.
                
                response_text = call_gemini_api(SYSTEM_PROMPT_CHATBOT, prompt)
                
                st.markdown(response_text)
                
                # Add assistant response to history
                st.session_state.chat_messages.append({"role": "assistant", "content": response_text})

