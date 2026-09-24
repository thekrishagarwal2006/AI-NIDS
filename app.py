import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI-Based Explainable & Adversarial-Resistant NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Academic CSS Styling
st.markdown("""
    <style>
    /* Dark Academic Palette */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Header Banner */
    .header-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        padding: 24px 30px;
        border-radius: 12px;
        color: #FFFFFF;
        text-align: left;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        font-size: 30px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 6px;
    }
    .header-subtitle {
        font-size: 15px;
        color: #94A3B8;
        font-weight: 400;
    }
    
    /* Cards */
    .academic-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        color: #E2E8F0;
        margin-bottom: 20px;
    }
    .academic-card h4 {
        color: #6366F1;
        margin-top: 0;
        font-size: 17px;
        font-weight: 600;
    }

    /* Badges */
    .badge-category {
        background-color: #334155;
        color: #38BDF8;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Top Academic Header Banner
st.markdown("""
    <div class="header-container">
        <div class="header-title">🛡️ AI-Based Explainable & Adversarial-Resistant NIDS</div>
        <div class="header-subtitle">Academic Research Prototype • NSL-KDD Benchmark Evaluation</div>
    </div>
""", unsafe_allow_html=True)

# Helper function to load models & datasets
@st.cache_resource
def load_all_artifacts():
    prep = joblib.load("models/preprocessor.pkl") if os.path.exists("models/preprocessor.pkl") else None
    best = joblib.load("models/best_model.pkl") if os.path.exists("models/best_model.pkl") else None
    return prep, best

@st.cache_data
def load_datasets():
    from src.data_preprocessing import COLUMNS, map_attack_category
    train_df = pd.read_csv("data/KDDTrain+.txt", names=COLUMNS) if os.path.exists("data/KDDTrain+.txt") else None
    test_df = pd.read_csv("data/KDDTest+.txt", names=COLUMNS) if os.path.exists("data/KDDTest+.txt") else None
    
    if train_df is not None:
        if 'difficulty_level' in train_df.columns:
            train_df = train_df.drop(columns=['difficulty_level'])
        train_df['attack_cat'] = train_df['attack'].apply(map_attack_category)
        
    if test_df is not None:
        if 'difficulty_level' in test_df.columns:
            test_df = test_df.drop(columns=['difficulty_level'])
        test_df['attack_cat'] = test_df['attack'].apply(map_attack_category)
        
    return train_df, test_df

preprocessor, best_model_info = load_all_artifacts()
df_train, df_test = load_datasets()

# Sidebar Setup
st.sidebar.markdown("### 📌 System Navigation")
page = st.sidebar.radio("Select View", [
    "📋 System Overview & Architecture",
    "📊 Exploratory Data Analysis",
    "🤖 Model Evaluation Leaderboard",
    "🔍 Live Flow Inference & XAI (SHAP)",
    "🛡️ Adversarial Robustness & Defense"
])

st.sidebar.markdown("---")
st.sidebar.caption("Evaluated on untouched NSL-KDD benchmark test set.")

# PAGE 1: OVERVIEW
if page == "📋 System Overview & Architecture":
    st.header("📋 System Overview & Technical Architecture")
    
    st.markdown("""
    <div class="academic-card">
        <h4>System Abstract</h4>
        This project implements a multi-model Network Intrusion Detection System (NIDS) incorporating <b>Explainable AI (XAI via SHAP)</b>, 
        <b>Adversarial Evasion Vulnerability Testing (FGSM & PGD)</b>, and <b>Adversarial Defense Training</b>.
        Evaluated on the standard <b>NSL-KDD benchmark dataset</b> using rigorous train-only SMOTE resampling and untouched test evaluation.
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Dataset", "125,973 Flows", "KDDTrain+.txt")
    c2.metric("Untouched Test Set", "22,544 Flows", "KDDTest+.txt")
    c3.metric("Evaluated Models", "3 Algorithms", "RF, XGBoost, DNN")
    c4.metric("Top Model Accuracy", "79.17%", "Keras DNN")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Technical Execution Pipeline")
        st.markdown("""
        1. **Data Ingestion**: Load raw train and test flows from NSL-KDD benchmark files.
        2. **Categorical Alignment & Scaling**: One-Hot encode symbolic attributes (`protocol_type`, `service`, `flag`) and scale numerical attributes with `StandardScaler`.
        3. **Class Imbalance Remediation**: Apply SMOTE strictly to the training partition.
        4. **Model Training**: Fit Random Forest, XGBoost Classifier, and Keras Deep Neural Network (DNN).
        5. **Untouched Test Evaluation**: Benchmark Accuracy, Macro/Weighted F1, and False Positive Rate (FPR).
        6. **SHAP Explainability**: Compute global feature importance and local instance attributions.
        7. **Adversarial Vulnerability Analysis**: Generate FGSM and PGD gradient evasion samples.
        8. **Robustness Defense Retraining**: Retrain Keras DNN on clean + adversarial training sets.
        """)
        
    with col2:
        st.subheader("🎯 Attack Taxonomy Mapping")
        st.markdown("""
        - **Normal**: Safe, legitimate traffic connections.
        - **DoS (Denial of Service)**: Resource exhaustion flooding (e.g. `smurf`, `neptune`).
        - **Probe**: Reconnaissance port scanning & host sweeps (e.g. `satan`, `ipsweep`).
        - **R2L (Remote to Local)**: Unauthorized remote access attempts (e.g. `guess_passwd`, `warezmaster`).
        - **U2R (User to Root)**: Privilege escalation to administrative superuser (e.g. `buffer_overflow`).
        """)

# PAGE 2: EXPLORATORY DATA ANALYSIS (EDA)
elif page == "📊 Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Quantitative distribution and correlation analysis across the NSL-KDD benchmark dataset.")
    
    if df_train is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Attack Category Distribution")
            cat_counts = df_train['attack_cat'].value_counts().reset_index()
            cat_counts.columns = ['Category', 'Count']
            
            fig1 = px.pie(
                cat_counts, names='Category', values='Count', hole=0.3,
                title="Traffic Category Proportion (Training Set)",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig1.update_traces(textinfo='percent+label')
            fig1.update_layout(height=420, template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader("3. Feature Correlation Matrix")
            num_cols = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
                        'serror_rate', 'same_srv_rate', 'diff_srv_rate', 'dst_host_count', 'dst_host_srv_count']
            avail_cols = [c for c in num_cols if c in df_train.columns]
            corr = df_train[avail_cols].corr()
            
            fig3 = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="Viridis",
                title="Correlation Heatmap (Key Network Attributes)"
            )
            fig3.update_layout(height=450, template="plotly_dark")
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            st.subheader("2. Top 10 Specific Traffic Subtypes")
            top_attacks = df_train['attack'].value_counts().head(10).reset_index()
            top_attacks.columns = ['Traffic Type', 'Count']
            
            fig2 = px.bar(
                top_attacks, x='Count', y='Traffic Type', orientation='h', color='Traffic Type',
                text='Count', color_discrete_sequence=px.colors.qualitative.D3,
                title="Top 10 Attack Subtypes in NSL-KDD"
            )
            fig2.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig2.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'}, height=420, template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("4. Source Bytes vs. Connection Count Distribution")
            df_sample_plot = df_train.sample(n=min(3000, len(df_train)), random_state=42).copy()
            df_sample_plot['Log Source Bytes'] = np.log1p(df_sample_plot['src_bytes'])
            df_sample_plot['Log Connection Count'] = np.log1p(df_sample_plot['count'])
            
            fig4 = px.scatter(
                df_sample_plot, x='Log Source Bytes', y='Log Connection Count',
                color='attack_cat', opacity=0.7,
                title="Log(Source Bytes) vs Log(Count) Distribution",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig4.update_layout(height=450, template="plotly_dark")
            st.plotly_chart(fig4, use_container_width=True)

# PAGE 3: MODEL EVALUATION
elif page == "🤖 Model Evaluation Leaderboard":
    st.header("🤖 Model Evaluation & Metrics Leaderboard")
    st.markdown("Benchmarking model generalization on the untouched test partition (`KDDTest+.txt`).")
    
    if os.path.exists("results/model_comparison.csv"):
        df_comp = pd.read_csv("results/model_comparison.csv")
        
        st.subheader("📈 Quantitative Metric Comparison")
        st.dataframe(df_comp.style.format({
            'Accuracy': '{:.2%}',
            'Macro Precision': '{:.2%}',
            'Macro Recall': '{:.2%}',
            'Macro F1-Score': '{:.2%}',
            'Weighted F1-Score': '{:.2%}',
            'Normal FPR': '{:.2%}',
            'Macro FPR': '{:.2%}'
        }), use_container_width=True)
        
        st.subheader("📊 Model Metric Visual Leaderboard")
        df_melt = pd.melt(df_comp, id_vars=['Model'], value_vars=['Accuracy', 'Macro F1-Score', 'Weighted F1-Score'], var_name='Metric', value_name='Score')
        
        fig_comp = px.bar(
            df_melt, x='Model', y='Score', color='Metric', barmode='group',
            text=df_melt['Score'].apply(lambda x: f"{x*100:.1f}%"),
            title="Evaluation Metrics Across Classification Models",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_comp.update_traces(textposition='outside')
        fig_comp.update_layout(yaxis=dict(range=[0, 1.1]), height=450, template="plotly_dark")
        st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("📌 Confusion Matrix Inspection")
    selected_model_name = st.selectbox("Select Model Architecture", ["DNN (Best Model)", "XGBoost", "Random Forest"])
    
    if preprocessor is not None and df_test is not None:
        from src.data_preprocessing import CATEGORICAL_COLS
        scaler = preprocessor['scaler']
        label_encoder = preprocessor['label_encoder']
        
        X_test_raw = df_test.drop(columns=['attack', 'attack_cat'])
        y_test_true = label_encoder.transform(df_test['attack_cat'])
        
        X_test_encoded = pd.get_dummies(X_test_raw, columns=CATEGORICAL_COLS)
        train_encoded_cols = preprocessor['train_encoded_columns']
        X_test_aligned = pd.DataFrame(0, index=X_test_encoded.index, columns=train_encoded_cols)
        for col in X_test_encoded.columns:
            if col in train_encoded_cols:
                X_test_aligned[col] = X_test_encoded[col]
                
        X_test_scaled = scaler.transform(X_test_aligned)
        
        model_file_map = {
            "DNN (Best Model)": "models/dnn_model.keras",
            "XGBoost": "models/xgboost.pkl",
            "Random Forest": "models/random_forest.pkl"
        }
        
        target_file = model_file_map[selected_model_name]
        if os.path.exists(target_file):
            if target_file.endswith(".keras"):
                import tensorflow as tf
                mod = tf.keras.models.load_model(target_file)
                preds = np.argmax(mod.predict(X_test_scaled, verbose=0), axis=1)
            else:
                mod = joblib.load(target_file)
                preds = mod.predict(X_test_scaled)
                
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test_true, preds)
            class_names = list(label_encoder.classes_)
            
            fig_cm = px.imshow(
                cm, x=class_names, y=class_names, text_auto=True,
                color_continuous_scale="Purples",
                title=f"Confusion Matrix: True vs Predicted Classes ({selected_model_name})",
                labels=dict(x="Predicted Class", y="Actual True Class")
            )
            fig_cm.update_layout(height=450, template="plotly_dark")
            st.plotly_chart(fig_cm, use_container_width=True)

# PAGE 4: LIVE FLOW CLASSIFIER & SHAP XAI
elif page == "🔍 Live Flow Inference & XAI (SHAP)":
    st.header("🔍 Real-Time Flow Inference & SHAP Explainability")
    st.markdown("Perform instance-level flow classification and analyze feature contribution vectors using SHAP (SHapley Additive exPlanations).")
    
    if preprocessor is None or best_model_info is None or df_test is None:
        st.error("Preprocessed artifacts missing. Run 'python main.py' first.")
    else:
        scaler = preprocessor['scaler']
        label_encoder = preprocessor['label_encoder']
        feature_names = preprocessor['feature_names']
        best_model_name = best_model_info['best_model_name']
        model = best_model_info['model']
        
        st.markdown(f"**Deployed Model**: `<span class='badge-category'>{best_model_name}</span>`", unsafe_allow_html=True)
        st.write("")
        
        sample_idx = st.slider("Select Network Traffic Flow Index", min_value=0, max_value=len(df_test)-1, value=0, step=1)
        
        ground_truth = df_test['attack_cat'].iloc[sample_idx]
        raw_attack = df_test['attack'].iloc[sample_idx]
        
        from src.data_preprocessing import CATEGORICAL_COLS
        X_test_raw = df_test.drop(columns=['attack', 'attack_cat'])
        X_test_encoded = pd.get_dummies(X_test_raw, columns=CATEGORICAL_COLS)
        
        train_encoded_cols = preprocessor['train_encoded_columns']
        X_test_aligned = pd.DataFrame(0, index=X_test_encoded.index, columns=train_encoded_cols)
        for col in X_test_encoded.columns:
            if col in train_encoded_cols:
                X_test_aligned[col] = X_test_encoded[col]
                
        X_test_scaled = scaler.transform(X_test_aligned)
        sample_vector = X_test_scaled[sample_idx]
        
        # Predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(pd.DataFrame([sample_vector], columns=feature_names))[0]
        else:
            p = model.predict(np.array([sample_vector]), verbose=0)
            probs = p[0] if p.ndim > 1 else np.array([1 - p[0], p[0]])
            
        pred_class_idx = np.argmax(probs)
        pred_class_name = label_encoder.inverse_transform([pred_class_idx])[0]
        confidence = probs[pred_class_idx]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Class", pred_class_name.upper())
        m2.metric("Prediction Confidence", f"{confidence*100:.2f}%")
        m3.metric("Ground Truth Label", ground_truth.upper())
        m4.metric("Classification Result", "MATCHED ✅" if pred_class_name == ground_truth else "MISCLASSIFIED ❌")
        
        st.markdown("---")
        st.subheader("🔎 Instance-Level Feature Contribution Vector (SHAP Values)")
        
        from src.shap_explainability import explain_single_prediction
        explanation_text, feat_contribs = explain_single_prediction(
            model, sample_vector, feature_names, label_encoder, X_background=X_test_scaled[:50], sample_idx=sample_idx, results_dir="results/shap"
        )
        
        top_15 = feat_contribs[:15]
        df_local_shap = pd.DataFrame(top_15, columns=['Feature', 'Feature Value', 'SHAP Value'])
        df_local_shap['Contribution'] = df_local_shap['SHAP Value'].apply(lambda x: 'Positive Contribution' if x > 0 else 'Negative Contribution')
        
        fig_local = px.bar(
            df_local_shap.sort_values(by='SHAP Value', ascending=True),
            x='SHAP Value', y='Feature', orientation='h', color='Contribution',
            text=df_local_shap['SHAP Value'].apply(lambda x: f"{x:+.4f}"),
            color_discrete_map={'Positive Contribution': '#F87171', 'Negative Contribution': '#60A5FA'},
            title=f"Top Feature Contributions for Prediction: {pred_class_name.upper()} (Sample #{sample_idx})"
        )
        fig_local.update_traces(textposition='outside')
        fig_local.update_layout(height=500, template="plotly_dark")
        st.plotly_chart(fig_local, use_container_width=True)
        
        st.subheader("💬 Qualitative XAI Text Summary")
        st.info(explanation_text)

# PAGE 5: ADVERSARIAL ATTACKS & DEFENSE
elif page == "🛡️ Adversarial Robustness & Defense":
    st.header("🛡️ Adversarial Vulnerability & Defense Analysis")
    st.markdown("Evaluating Deep Neural Network performance degradation under **Fast Gradient Sign Method (FGSM)** and **Projected Gradient Descent (PGD)** evasion attacks.")
    
    if os.path.exists("results/adversarial/adversarial_evaluation.csv"):
        df_adv_eval = pd.read_csv("results/adversarial/adversarial_evaluation.csv")
        
        st.subheader("💥 Baseline DNN Evasion Performance Degradation")
        fig_adv_eval = px.bar(
            df_adv_eval, x='Condition', y='Accuracy', color='Condition',
            text=df_adv_eval['Accuracy'].apply(lambda x: f"{x*100:.1f}%"),
            title="Clean Accuracy vs. Accuracy Drop Under FGSM & PGD Attacks",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_adv_eval.update_traces(textposition='outside')
        fig_adv_eval.update_layout(yaxis=dict(range=[0, 1.1]), showlegend=False, height=450, template="plotly_dark")
        st.plotly_chart(fig_adv_eval, use_container_width=True)
        
        st.subheader("📋 Empirical Evasion Attack Results")
        st.dataframe(df_adv_eval.style.format({
            'Accuracy': '{:.2%}',
            'Accuracy Drop': '{:.2%}'
        }), use_container_width=True)
        
    st.markdown("---")
    st.subheader("🛡️ Defense Evaluation: Baseline DNN vs. Adversarially Retrained DNN")
    
    comp_df = pd.DataFrame([
        {'Model': 'Baseline DNN', 'Condition': 'Clean', 'Accuracy': 0.7892},
        {'Model': 'Baseline DNN', 'Condition': 'FGSM (eps=0.1)', 'Accuracy': 0.6604},
        {'Model': 'Baseline DNN', 'Condition': 'PGD (eps=0.1)', 'Accuracy': 0.6516},
        {'Model': 'Adversarially Retrained DNN', 'Condition': 'Clean', 'Accuracy': 0.7748},
        {'Model': 'Adversarially Retrained DNN', 'Condition': 'FGSM (eps=0.1)', 'Accuracy': 0.4228},
        {'Model': 'Adversarially Retrained DNN', 'Condition': 'PGD (eps=0.1)', 'Accuracy': 0.3240},
    ])
    
    fig_def = px.bar(
        comp_df, x='Condition', y='Accuracy', color='Model', barmode='group',
        text=comp_df['Accuracy'].apply(lambda x: f"{x*100:.1f}%"),
        title="Robustness Comparison Across Attack Conditions",
        color_discrete_map={'Baseline DNN': '#EF4444', 'Adversarially Retrained DNN': '#3B82F6'}
    )
    fig_def.update_traces(textposition='outside')
    fig_def.update_layout(yaxis=dict(range=[0, 1.1]), height=450, template="plotly_dark")
    st.plotly_chart(fig_def, use_container_width=True)
