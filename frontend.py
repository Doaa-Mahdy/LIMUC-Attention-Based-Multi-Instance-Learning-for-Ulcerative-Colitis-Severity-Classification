import streamlit as st
import requests
import pandas as pd
import time
import base64

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LIMUC AI Medical System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"
ALLOWED_TYPES = ["jpg", "png", "jpeg", "bmp", "webp", "tiff"]

# --- SIDEBAR DASHBOARD ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    st.title("LIMUC AI System")
    st.caption("v1.2 | Ulcerative Colitis Scoring")
    st.markdown("---")
    
    st.subheader("🔌 System Status")
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        if response.status_code == 200:
            st.success("🟢 Backend Online")
            data = response.json()
            
            # Show Model Status
            with st.expander("Model Health Check", expanded=True):
                if data['models_loaded'].get('regressor'):
                    st.markdown("✅ **Regressor:** Active")
                else:
                    st.markdown("❌ **Regressor:** Offline")
                
                if data['models_loaded'].get('mil'):
                    st.markdown("✅ **MIL Model:** Active")
                else:
                    st.markdown("❌ **MIL Model:** Offline")
        else:
            st.error("⚠️ Backend Error")
    except:
        st.error("🔴 Backend Offline")
        st.info("Run `python main.py` in terminal.")

    st.markdown("---")
    st.info("This tool assists endoscopists in grading Ulcerative Colitis severity. Always verify with clinical judgment.")

# --- MAIN LAYOUT ---
st.title("🩺 Medical Diagnosis Dashboard")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔬 Single Frame Analysis", "📁 Patient Diagnosis (MIL)", "📏 Severity Regression"])

# ==========================================
# TASK 1: SINGLE FRAME CLASSIFICATION
# ==========================================
with tab1:
    st.markdown("### Task 1: Frame Classification")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.write("#### 1. Upload Endoscopy Frame")
        uploaded_file = st.file_uploader("Upload Image", type=ALLOWED_TYPES, key="t1")
        
        if uploaded_file:
            st.image(uploaded_file, caption="Preview", use_container_width=True)
            
            # ANALYZE BUTTON
            if st.button("Analyze Frame", key="btn1"):
                with st.spinner("Processing image..."):
                    try:
                        files = {"file": uploaded_file.getvalue()}
                        res = requests.post(f"{API_URL}/predict/classification", files=files)
                        if res.status_code == 200:
                            st.session_state['t1_result'] = res.json()
                            # Clear old heatmap if new image analyzed
                            if 't1_heatmap' in st.session_state:
                                del st.session_state['t1_heatmap']
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

            # EXPLAIN BUTTON (Only shows if analysis is done)
            if 't1_result' in st.session_state:
                st.write("---")
                if st.button("🔍 Explain Decision (Grad-CAM)"):
                    with st.spinner("Generating Heatmap..."):
                        try:
                            # Use the same file (requires re-seeking or simple re-upload logic in memory)
                            # Simple hack: use the value in session state or re-read
                            uploaded_file.seek(0)
                            files = {"file": uploaded_file.getvalue()}
                            
                            res = requests.post(f"{API_URL}/explain/classification", files=files)
                            if res.status_code == 200:
                                st.session_state['t1_heatmap'] = res.json()['heatmap_base64']
                            else:
                                st.error("Could not generate heatmap.")
                        except Exception as e:
                            st.error(f"Error: {e}")

    with col2:
        st.write("#### 2. AI Analysis Results")
        if 't1_result' in st.session_state:
            res = st.session_state['t1_result']
            
            # Top Result Card
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{res['predicted_class']}</h3>
                    <p>Predicted Class</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{float(res['confidence'])*100:.1f}%</h3>
                    <p>Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Bar Chart
            st.subheader("Probability Distribution")
            probs = res['probabilities']
            df_probs = pd.DataFrame(list(probs.items()), columns=["Mayo Class", "Probability"])
            df_probs['Probability'] = df_probs['Probability'].astype(float)
            st.bar_chart(df_probs.set_index("Mayo Class"), color="#007bff")

            # --- HEATMAP DISPLAY ---
            if 't1_heatmap' in st.session_state:
                st.write("---")
                st.subheader("🧠 AI Visual Reasoning (Grad-CAM)")
                st.write("The **Red/Orange** areas show where the model looked to make this decision.")
                
                # Decode Base64
                import base64
                heatmap_bytes = base64.b64decode(st.session_state['t1_heatmap'])
                st.image(heatmap_bytes, caption="Grad-CAM Visualization", use_container_width=True)

        else:
            st.info("Upload an image and click Analyze to see results.")

# ==========================================
# TASK 2: PATIENT LEVEL DIAGNOSIS (MIL)
# ==========================================
with tab2:
    st.markdown("### Task 2: Patient-Level Diagnosis")
    st.write("Upload multiple frames to get an aggregated patient score.")

    uploaded_files = st.file_uploader("Upload Patient Frames", type=ALLOWED_TYPES, accept_multiple_files=True, key="t2")
    
    if uploaded_files:
        with st.expander(f"📸 View Uploaded Frames ({len(uploaded_files)})"):
            st.image(uploaded_files[:8], width=100)
            if len(uploaded_files) > 8: st.caption("...and more")
        
        # ANALYZE BUTTON
        if st.button(f"Analyze Patient Case", key="btn2"):
            with st.spinner("Aggregating features..."):
                try:
                    files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                    res = requests.post(f"{API_URL}/predict/patient", files=files)
                    
                    if res.status_code == 200:
                        st.session_state['t2_result'] = res.json()
                        # Clear old explanation
                        if 't2_explanation' in st.session_state: del st.session_state['t2_explanation']
                    else:
                        st.error("Server Error")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")

        # SHOW RESULTS
        if 't2_result' in st.session_state:
            result = st.session_state['t2_result']
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            
            diagnosis = result['patient_diagnosis']
            text_color = "#28a745" if "0" in diagnosis or "1" in diagnosis else "#dc3545"
            
            with c1:
                st.markdown(f"<div class='metric-card'><h3 style='color:{text_color} !important'>{diagnosis}</h3><p>Diagnosis</p></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-card'><h3>{float(result['confidence'])*100:.1f}%</h3><p>Confidence</p></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='metric-card'><h3>{result['num_frames_analyzed']}</h3><p>Frames Analyzed</p></div>", unsafe_allow_html=True)
            
            st.success(f"**Clinical Note:** {result.get('clinical_note', '')}")

            # EXPLAIN BUTTON
            st.write("---")
            if st.button("🔍 Reveal Key Key Frames (Reasoning)"):
                with st.spinner("Identifying most important frames..."):
                    try:
                        # Re-prepare files (Streamlit specific hack to reuse uploaded files)
                        files = []
                        for f in uploaded_files:
                            f.seek(0)
                            files.append(("files", (f.name, f.getvalue(), f.type)))
                        
                        res = requests.post(f"{API_URL}/explain/patient", files=files)
                        if res.status_code == 200:
                            st.session_state['t2_explanation'] = res.json()['top_frames']
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")

            # SHOW EXPLANATION
            if 't2_explanation' in st.session_state:
                st.subheader("🧠 Key Evidence Frames")
                st.info("The AI based its diagnosis primarily on these specific frames:")
                
                cols = st.columns(3)
                for i, frame_data in enumerate(st.session_state['t2_explanation']):
                    with cols[i]:
                        st.image(base64.b64decode(frame_data['heatmap_base64']), use_container_width=True)
                        st.caption(f"Importance: {frame_data['importance_score']*100:.1f}%")

# ==========================================
# TASK 3: GRAY ZONE REGRESSION
# ==========================================
with tab3:
    st.markdown("### Task 3: Severity Regression (The Gray Zone)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file_reg = st.file_uploader("Upload Image", type=ALLOWED_TYPES, key="t3")
        if uploaded_file_reg:
            st.image(uploaded_file_reg, use_container_width=True)
            
            # ANALYZE BUTTON
            if st.button("Calculate Severity Score", key="btn3"):
                with st.spinner("Calculating..."):
                    try:
                        files = {"file": uploaded_file_reg.getvalue()}
                        res = requests.post(f"{API_URL}/predict/regression", files=files)
                        
                        if res.status_code == 200:
                            st.session_state['t3_result'] = res.json()
                            if 't3_heatmap' in st.session_state: del st.session_state['t3_heatmap']
                        else:
                            st.error("Server Error")
                    except Exception as e:
                        st.error(f"Error: {e}")

            # EXPLAIN BUTTON
            if 't3_result' in st.session_state:
                st.write("---")
                if st.button("🔍 Explain Score (Grad-CAM)"):
                    with st.spinner("Generating Heatmap..."):
                        try:
                            uploaded_file_reg.seek(0)
                            files = {"file": uploaded_file_reg.getvalue()}
                            res = requests.post(f"{API_URL}/explain/regression", files=files)
                            
                            if res.status_code == 200:
                                st.session_state['t3_heatmap'] = res.json()['heatmap_base64']
                            else:
                                st.error("Error generating heatmap")
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    with col2:
        if 't3_result' in st.session_state:
            data = st.session_state['t3_result']
            score = data['continuous_score']
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{score}</h3>
                <p style='color: #007bff !important'>{data['clinical_status']}</p>
            </div>
            """, unsafe_allow_html=True)
                
            st.write("Severity Scale:")
            my_bar = st.progress(0)
            for percent_complete in range(int(min(score/3.0, 1.0) * 100)):
                time.sleep(0.005)
                my_bar.progress(percent_complete + 1)
            
            st.text(f"0 (Healthy) ............... 1.5 ............... 3.0 (Severe)")
            st.info(f"**Interpretation:** {data['explanation']}")

            # SHOW HEATMAP
            if 't3_heatmap' in st.session_state:
                st.write("---")
                st.subheader("Visual Evidence")
                st.image(base64.b64decode(st.session_state['t3_heatmap']), caption="Regression Focus Area", use_container_width=True)