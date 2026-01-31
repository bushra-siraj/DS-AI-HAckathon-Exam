import streamlit as st
import pandas as pd
import joblib
import numpy as np

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from PIL import Image

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Talent Scout Pro", layout="wide", page_icon="🧠")

# --- CUSTOM ANIMATED CSS ---
st.markdown("""
    <style>
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .main { animation: fadeIn 1.5s; background-color: #f0f2f6; }
    .stButton>button { 
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%); 
        color: white; border: none; border-radius: 20px; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.05); box-shadow: 0px 4px 15px rgba(0,0,0,0.2); }
    .model-card { 
        background: white; padding: 20px; border-radius: 15px; 
        box-shadow: 5px 5px 15px rgba(0,0,0,0.05); text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    models = {
        "XGBoost": joblib.load('xgboost_model.pkl'),
        "Random Forest": joblib.load('rf_model.pkl'),
        "Logistic Regression": joblib.load('lr_model.pkl'),
        "SVM": joblib.load('svm_model.pkl'),
        "KNN": joblib.load('knn_model.pkl')
    }
    preprocessor = joblib.load('preprocessor.pkl')
    return models, preprocessor

# Load your actual dataset for overview
df = pd.read_csv('AI_Resume_Screening.csv')

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("💎 Talent Analytics")
menu = st.sidebar.selectbox("Navigation", ["Overview", "Insights Gallery", "Model Performance", "Live Multi-Model Predictor"])

# --- PAGE 1: OVERVIEW ---
if menu == "Overview":
    st.title("🌟 Dataset Intelligence Overview")
    st.write("Real-time summary of the candidate training pool.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Candidates", len(df), "+5%")
    with col2: st.metric("Avg AI Score", f"{df['AI Score (0-100)'].mean():.1f}")
    with col3: st.metric("Hire Rate", f"{(df['Recruiter Decision'] == 'Hire').mean()*100:.1f}%")
    with col4: st.metric("Target Roles", len(df['Job Role'].unique()))
    
    st.subheader("Data Explorer")
    st.dataframe(df.style.background_gradient(cmap='Blues'), use_container_width=True)

# --- PAGE 2: INSIGHTS GALLERY (The 17 Graphs) ---
elif menu == "Insights Gallery":
    st.title("🎨 Visual Insights Gallery")
    
    # USE THIS: Path handling that works on all computers
    import os
    base_path = os.path.dirname(__file__) # Gets the folder where app.py is
    graph_folder = os.path.join(base_path, "images")

    tab1, tab2 = st.tabs(["📊 Univariate Analysis", "🔗 Bivariate Analysis"])

    with tab1:
        st.subheader("Distribution of Individual Features")
        img_path = os.path.join(graph_folder, "numerical_distributions.png")
        
        # Add a check to see if the file exists before showing it
        if os.path.exists(img_path):
            st.image(img_path, caption="Numerical Feature Distributions", use_column_width=True)
        else:
            st.error(f"Could not find: {img_path}")

    with tab2:
        st.subheader("Relationships between Features")
        with st.expander("Click to view all 14 Bivariate Relationships"):
            cols = st.columns(2)
            bivariate_images = [
                "feature_correlation.png", "education_vs_decision.png", 
                "ai_score_vs_decision.png", 'skills_vs_decision.png', 'projects_vs_decision.png', 
                'salary_vs_decision.png', 'ai_score_by_role.png', 'experience_ai_correlation.png', 
                'certifications_hired.png', 'edu_vs_aiscore.png', 'exp_by_role.png', 'projects_violin.png',
                'feature_correlation.png', 'overall_patterns.png'
            ]
            
            for i, img_name in enumerate(bivariate_images):
                col_idx = i % 2 
                with cols[col_idx]:
                    full_img_path = os.path.join(graph_folder, img_name)
                    if os.path.exists(full_img_path):
                        st.image(full_img_path, use_column_width=True)
                    else:
                        st.warning(f"Missing: {img_name}")

# --- PAGE 3: PERFORMANCE ---
elif menu == "Model Performance":
    st.title("🏆 Model Comparison")
    perf_data = {
        "Model": ["XGBoost", "Logistic Regression", "SVM", "KNN", "Random Forest"],
        "Accuracy": [1.000, 0.989, 0.964, 0.929, 0.782],
        "F1-Score": [1.000, 0.993, 0.976, 0.954, 0.876]
    }
    performance_df = pd.DataFrame(perf_data)
    st.table(performance_df)
    st.bar_chart(performance_df.set_index("Model")["Accuracy"])

# --- PAGE 4: LIVE PREDICTION (Multi-Model) ---
elif menu == "Live Multi-Model Predictor":
    st.title("🔮 Multi-Model Candidate Assessment")
    st.write("Input features below. All 5 models will evaluate the candidate simultaneously.")
    
    models, preprocessor = load_models()
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            exp = st.number_input("Experience (Years)", 0, 40, 5)
            ai_score = st.slider("AI Score (0-100)", 0, 100, 75)
        with c2:
            edu = st.selectbox("Education Level", df['Education'].unique())
            salary = st.number_input("Salary Expectation ($)", 30000, 200000, 80000)
        with c3:
            role = st.selectbox("Target Job Role", df['Job Role'].unique())
            proj = st.number_input("Projects Count", 0, 20, 3)
            
    if st.button("🔥 Run Unified Assessment"):
        input_df = pd.DataFrame([{
            'Experience (Years)': exp, 'Education': edu, 'Projects Count': proj,
            'AI Score (0-100)': ai_score, 'Salary Expectation ($)': salary,
            'Job Role': role, 'Skills': 'Python', 'Certifications': 'None'
        }])
        
        processed_input = preprocessor.transform(input_df)
        
        progress = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress.progress(percent_complete + 1)
        
        results = {}
        for name, m in models.items():
            pred = m.predict(processed_input)[0]
            results[name] = "HIRE" if pred == 1 else "REJECT"
            
        st.write("### Model Decisions")
        res_cols = st.columns(5)
        for i, (name, decision) in enumerate(results.items()):
            with res_cols[i]:
                color = "#dcfce7" if decision == "HIRE" else "#fee2e2"
                text_color = "#166534" if decision == "HIRE" else "#991b1b"
                st.markdown(f"""
                    <div class="model-card" style="background-color: {color};">
                        <h4 style="color: #333;">{name}</h4>
                        <h2 style="color: {text_color};">{decision}</h2>
                    </div>
                """, unsafe_allow_html=True)
        
        if list(results.values()).count("HIRE") >= 3:
            st.balloons()
            st.success("✨ **Final Consensus: Recommended for Hire**")
        else:
            st.error("⚠️ **Final Consensus: Not Recommended**")