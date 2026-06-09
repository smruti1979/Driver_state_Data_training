import sys
import ctypes
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import cv2
import csv
import time
from io import BytesIO
from datetime import datetime
from fpdf import FPDF

# Import your cleanly isolated app modules
import config
from engine import DriverSafetySimulator
import atexit

# ==========================================
# 1. WINDOWS VS CODE THREAD-SAFE EXIT HOOK
# ==========================================
if os.name == 'nt':
    PHANDLER_ROUTINE = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)
    def windows_ctrl_handler(ctrl_type):
        if ctrl_type in (0, 1):
            try:
                if "cap" in st.session_state and st.session_state.cap is not None:
                    st.session_state.cap.release()
            except Exception: pass
            os._exit(0)
        return 1
    _global_handler_ref = PHANDLER_ROUTINE(windows_ctrl_handler)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_global_handler_ref, True)

def structural_cleanup_on_exit():
    if "cv_pipeline" in st.session_state:
        st.session_state.cv_pipeline.close_all_resources()
    if "cap" in st.session_state and st.session_state.cap is not None:
        st.session_state.cap.release()

atexit.register(structural_cleanup_on_exit)

# ==========================================
# 2. GLOBAL PAGE CONFIG & PROFESSIONAL THEME
# ==========================================
st.set_page_config(page_title="Fleet Safety Telemetry Dashboard", layout="wide")

st.markdown("""
    <style>
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            color: #2D3748 !important;
            -webkit-font-smoothing: antialiased;
        }
        [data-testid="stSidebar"] {
            background-color: #F8FAFC !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
            max-width: 100% !important;
        }
        h1, h2, h3 {
            font-weight: 600 !important;
            color: #1A365D !important;
            letter-spacing: -0.02em !important;
        }
        div.stButton > button {
            border-radius: 6px !important;
            font-weight: 600 !important;
        }
        div.stButton > button[type="primary"] {
            background-color: #1A365D !important;
            border-color: #1A365D !important;
        }
        div.stButton > button[type="primary"]:hover {
            background-color: #2B6CB0 !important;
            border-color: #2B6CB0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================

# Instantiates simulator within Session state caching
if "cv_pipeline" not in st.session_state:
    st.session_state.cv_pipeline = DriverSafetySimulator()
if "is_running" not in st.session_state:
    st.session_state.is_running = False

pipeline = st.session_state.cv_pipeline

# --- NEW: MACHINE LEARNING DATA LOGGER FUNCTION ---
ML_DATASET_FILE = "driver_fatigue_dataset.csv"

def log_ml_features(ear, mar, head_deviation, label):
    """Appends live engine tracking parameters directly to a custom ML training dataset."""
    file_exists = os.path.isfile(ML_DATASET_FILE)
    with open(ML_DATASET_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "ear", "mar", "head_deviation", "label"])
        writer.writerow([time.time(), ear, mar, head_deviation, label])

def load_data():
    if os.path.exists(config.LOG_FILE):
        try:
            df = pd.read_csv(config.LOG_FILE)
            if not df.empty:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                return df
        except Exception: return None
    return None

def generate_pdf_report(df):
    pdf = FPDF(orientation="P", unit="mm", format="letter")
    pdf.set_margins(left=15, top=15, right=15)
    pdf.add_page()
    c_primary, c_secondary, c_text, c_bg_metrics, c_bg_alert = (26, 54, 93), (43, 108, 176), (45, 55, 72), (226, 232, 240), (254, 235, 200)
    
    pdf.set_font("Helvetica", "B", 22); pdf.set_text_color(*c_primary)
    pdf.cell(0, 10, "Fleet Safety Simulation Summary Report", ln=True)
    pdf.set_font("Helvetica", "", 10); pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 6, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | System Status: Active", ln=True); pdf.ln(8)
    
    total_frames = len(df)
    critical_alerts = len(df[df["Alert_Level"] == 2])
    warning_alerts = len(df[df["Alert_Level"] == 1])
    avg_latency = df["Latency_MS"].mean()
    duration = (df["Timestamp"].max() - df["Timestamp"].min()).total_seconds() if total_frames > 1 else 0
    
    pdf.set_font("Helvetica", "B", 14); pdf.set_text_color(*c_secondary)
    pdf.cell(0, 8, "Executive Performance Metrics", ln=True); pdf.ln(2)
    
    metrics_data = [
        ("Session Run Duration", f"{duration:.1f} seconds"), ("Total Video Frames Processed", f"{total_frames} frames"),
        ("Average Processing Latency", f"{avg_latency:.2f} ms"), ("Drowsiness / Distraction Events (Critical)", f"{critical_alerts} incidents"),
        ("Yawning / Fatigue Events (Warning)", f"{warning_alerts} incidents")
    ]
    
    pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(*c_primary); pdf.set_fill_color(*c_bg_metrics)
    pdf.cell(95, 7, "Metric Indicator", border=1, fill=True)
    pdf.cell(91, 7, "Recorded Value", border=1, fill=True, ln=True)
    
    pdf.set_font("Helvetica", "", 10); pdf.set_text_color(*c_text)
    for label, val in metrics_data:
        pdf.cell(95, 7, label, border=1)
        pdf.cell(91, 7, val, border=1, ln=True)
    
    pdf.ln(10); pdf.set_font("Helvetica", "B", 14); pdf.set_text_color(*c_secondary)
    pdf.cell(0, 8, "Recent Logged Safety Incidents (Max 15)", ln=True); pdf.ln(2)
    
    incidents_df = df[df["Alert_Level"] > 0].tail(15)
    if not incidents_df.empty:
        pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(123, 52, 30); pdf.set_fill_color(*c_bg_alert)
        pdf.cell(55, 7, "Timestamp", border=1, fill=True)
        pdf.cell(81, 7, "Violation Type", border=1, fill=True)
        pdf.cell(50, 7, "Latency", border=1, fill=True, ln=True)
        
        pdf.set_font("Helvetica", "", 10); pdf.set_text_color(*c_text)
        for _, row in incidents_df.iterrows():
            ts_str = row["Timestamp"].strftime("%H:%M:%S.%f")[:-3]
            pdf.cell(55, 7, ts_str, border=1)
            pdf.cell(81, 7, str(row["Alert_Type"]), border=1)
            pdf.cell(50, 7, f"{row['Latency_MS']:.1f} ms", border=1, ln=True)
    else:
        pdf.set_font("Helvetica", "", 10); pdf.set_text_color(*c_text)
        pdf.cell(0, 7, "No safety threshold violations or driver fatigue signatures registered during this session window.", ln=True)
        
    try:
        pdf_output = pdf.output(dest='S')
        return BytesIO(pdf_output.encode('latin1')) if isinstance(pdf_output, str) else BytesIO(pdf_output)
    except Exception:
        return BytesIO(pdf.output())

df = load_data()

# 1. Render Sidebar Components
with st.sidebar:
    st.header("📋 Session Operations")
    
    if not st.session_state.is_running:
        if st.button("🚀 Boot Vision Core Engine", type="primary", use_container_width=True):
            st.session_state.is_running = True
            st.rerun()
    else:
        if st.button("🛑 Shutdown Vision Engine", type="secondary", use_container_width=True):
            st.session_state.is_running = False
            pipeline.stop_alerts()
            st.rerun()

    if st.session_state.is_running:
        st.success("Webcam Pipeline Active")
    else:
        st.error("Webcam Pipeline Offline")

    # --- NEW: SIDEBAR ML RECORDER PANEL ---
    st.markdown("---")
    st.header("🤖 ML Classifier Data Recorder")
    ml_mode = st.radio(
        "Current Training Label Target:",
        ["🔴 Do Not Log Data", "🟢 Label 0: Alert State", "⚠️ Label 1: Fatigued State"]
    )
    
    # Visual counter for the tracking session
    if os.path.exists(ML_DATASET_FILE):
        try:
            samples_df = pd.read_csv(ML_DATASET_FILE)
            st.metric("Total ML Samples Collected", len(samples_df))
        except Exception: pass

    st.markdown("---")
    
    st.subheader("🧹 Database Maintenance")
    if st.button("🗑️ Reset Session Data", type="secondary", use_container_width=True):
        pipeline.stop_alerts()
        try:
            if os.path.exists(config.LOG_FILE):
                os.remove(config.LOG_FILE)
            with open(config.LOG_FILE, mode='w', newline='') as f:
                csv.writer(f).writerow(["Timestamp", "Alert_Type", "Alert_Level", "EAR", "MAR", "Head_Deviation", "Latency_MS"])
            st.toast("Telemetry data reset successfully!", icon="🧼")
            st.rerun()
        except Exception:
            st.sidebar.error("Could not reset database. Ensure the vision engine is stopped first.")

    st.markdown("---")
    if df is not None and len(df) > 0:
        st.info("Telemetry Database Linked")
        pdf_data = generate_pdf_report(df)
        st.download_button(
            label="📥 Export Session Summary PDF", data=pdf_data,
            file_name=f"fleet_safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf", use_container_width=True, key="pdf_download_button"
        )
    else:
        st.warning("No session metrics loaded.")

# 2. Render Main Layout Elements
st.title("📊 Commercial Fleet Driver Fatigue Telemetry Dashboard")
st.subheader("Real-Time Operator Assessment Lab")

col_left, col_right = st.columns(2)

with col_left:
    st.write("### 🎥 Driver Assessment Video Feed")
    video_placeholder = st.empty()

with col_right:
    st.write("### ⏱️ System Telemetry Summary")
    kpi_placeholder = st.empty()

st.markdown("---")
waveform_placeholder = st.empty()
tables_placeholder = st.empty()
map_placeholder = st.empty()

# --- INTERFACE SYNC LOOP ---
if st.session_state.is_running:
    # 1. Thread-safe initialization of hardware resource in session state
    if "cap" not in st.session_state or st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        # Force lower camera resolution to optimize processing bandwidth
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    cap = st.session_state.cap
    
    if not cap.isOpened():
        st.session_state.is_running = False
        st.error("Error: Hardware pipeline could not open webcam interface.")
        st.rerun()

    # 2. Isolated UI Rendering Context Fragment
    # This keeps UI changes confined strictly to the video component to eliminate page flickering
    @st.fragment
    def run_video_pipeline():
        last_chart_update = time.time()
        
        # Localized loop executes safely inside the fragment component container
        while st.session_state.is_running:
            success, raw_frame = cap.read()
            if not success or raw_frame is None:
                time.sleep(0.01)
                continue
                
            # Process single frame safely through your custom engine
            rgb_processed = pipeline.process_single_frame(raw_frame, ml_mode=ml_mode)
            
            # Update the existing placeholder directly (No page reload)
            video_placeholder.image(rgb_processed, channels="RGB", use_container_width=True)
            
            # Throttled background chart calculation
            current_time = time.time()
            if current_time - last_chart_update > 1.5:
                df_refresh = load_data()
                if df_refresh is not None and len(df_refresh) > 0:
                    with kpi_placeholder.container():
                        kpi1, kpi2, kpi3 = st.columns(3)
                        kpi1.metric("Total Frames", len(df_refresh))
                        kpi2.metric("Critical Alerts", len(df_refresh[df_refresh["Alert_Level"] == 2]))
                        kpi3.metric("Warning Alerts", len(df_refresh[df_refresh["Alert_Level"] == 1]))
                last_chart_update = current_time
                
            # Yield minimal execution sleep time to hand control back to the rendering engine
            time.sleep(0.01)

    # Invoke the optimized runtime loop
    run_video_pipeline()

else:
    # 3. Clean up camera allocation when user toggles engine to offline status
    if "cap" in st.session_state and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
