import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os
import base64
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_squared_error, r2_score, f1_score, precision_score, recall_score,
                             silhouette_score, mean_absolute_error)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from prophet import Prophet
from wordcloud import WordCloud
from fpdf import FPDF
import shap
import warnings
import tempfile
import zipfile
import shutil
import requests
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer
import contextlib  # Add this with other imports
import json
import hashlib


# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize session state for model tracking
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'trained_model_name' not in st.session_state:
    st.session_state.trained_model_name = None
if 'trained_model_acc' not in st.session_state:
    st.session_state.trained_model_acc = None
if 'regression_model' not in st.session_state:
    st.session_state.regression_model = None
if 'regression_model_name' not in st.session_state:
    st.session_state.regression_model_name = None
if 'regression_model_score' not in st.session_state:
    st.session_state.regression_model_score = None


# Directory for saving models and metadata
MODEL_DIR = "saved_models"
METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- User Profile Backend Helpers ---
USER_PROFILE_FILE = "user_profile.json"
USER_ACTIVITY_FILE = "user_activity.csv"
UPLOADED_DATASETS_FILE = "uploaded_datasets.csv"
DOWNLOAD_HISTORY_FILE = "download_history.csv"

# --- Per-User Data Helpers ---
USER_PROFILE_DIR = "user_profiles"
USER_DATA_DIR = "user_data"
os.makedirs(USER_PROFILE_DIR, exist_ok=True)
os.makedirs(USER_DATA_DIR, exist_ok=True)

def get_user_profile_path(username):
    return os.path.join(USER_PROFILE_DIR, f"user_{username}.json")

def get_user_data_path(username, dtype):
    return os.path.join(USER_DATA_DIR, f"user_{username}_{dtype}.csv")

def load_user_profile(username):
    path = get_user_profile_path(username)
    if not os.path.exists(path):
        # New user onboarding
        profile = {
            "nickname": username,
            "avatar": None,
            "join_date": datetime.now().strftime("%b %Y"),
            "theme": "🔥 Warm",
            "default_task": "Classification",
            "preview_rows": 10
        }
        with open(path, "w") as f:
            json.dump(profile, f)
        return profile, True  # True = new user
    with open(path, "r") as f:
        return json.load(f), False

def save_user_profile(username, profile):
    path = get_user_profile_path(username)
    with open(path, "w") as f:
        json.dump(profile, f)

def load_user_csv(username, dtype, columns=None):
    path = get_user_data_path(username, dtype)
    if not os.path.exists(path):
        if columns:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
        return pd.DataFrame(columns=columns if columns else [])
    return pd.read_csv(path)

def append_user_csv(username, dtype, row, columns=None):
    path = get_user_data_path(username, dtype)
    df = load_user_csv(username, dtype, columns)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)

def get_user_profile():
    if os.path.exists(USER_PROFILE_FILE):
        with open(USER_PROFILE_FILE, "r") as f:
            return json.load(f)
    # Default profile
    return {
        "nickname": "Anonymous User",
        "avatar": None,
        "join_date": datetime.now().strftime("%b %Y")
    }

def save_user_profile(profile):
    with open(USER_PROFILE_FILE, "w") as f:
        json.dump(profile, f)

# --- Activity Log Helpers ---
def log_user_activity(username, action, emoji, duration=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = {"timestamp": now, "emoji": emoji, "action": action, "duration": duration or ""}
    append_user_csv(username, "activity", row, columns=["timestamp", "emoji", "action", "duration"])

def get_activity_log():
    if not os.path.exists(USER_ACTIVITY_FILE):
        return []
    with open(USER_ACTIVITY_FILE) as f:
        lines = f.readlines()
    return [dict(timestamp=l.split(",")[0], emoji=l.split(",")[1], action=','.join(l.split(",")[2:]).strip()) for l in lines]

# --- Uploaded Datasets Log ---
def log_uploaded_dataset(username, filename, dtype, size):
    now = datetime.now().strftime("%Y-%m-%d")
    row = {"filename": filename, "type": dtype, "size": size, "date": now}
    append_user_csv(username, "datasets", row, columns=["filename", "type", "size", "date"])

def get_uploaded_datasets():
    if not os.path.exists(UPLOADED_DATASETS_FILE):
        return []
    with open(UPLOADED_DATASETS_FILE) as f:
        lines = f.readlines()
    return [dict(filename=l.split(",")[0], type=l.split(",")[1], size=l.split(",")[2], date=l.split(",")[3].strip()) for l in lines]

# --- Download History Log ---
def log_download(username, filename, dtype):
    now = datetime.now().strftime("%Y-%m-%d")
    row = {"filename": filename, "type": dtype, "date": now}
    append_user_csv(username, "downloads", row, columns=["filename", "type", "date"])

def get_download_history():
    if not os.path.exists(DOWNLOAD_HISTORY_FILE):
        return []
    with open(DOWNLOAD_HISTORY_FILE) as f:
        lines = f.readlines()
    return [dict(filename=l.split(",")[0], type=l.split(",")[1], date=l.split(",")[2].strip()) for l in lines]

# --- User Dashboard Section ---
def user_dashboard_section():
    username = st.session_state.get("username", "anonymous")
    profile, is_new = load_user_profile(username)
    nickname = profile.get("nickname", username)
    avatar = profile.get("avatar")
    join_date = profile.get("join_date", "Jan 2023")
    theme = profile.get("theme", "🔥 Warm")
    default_task = profile.get("default_task", "Classification")
    preview_rows = profile.get("preview_rows", 10)

    # Per-user data
    with st.spinner("Loading dashboard..."):
        models_df = load_user_csv(username, "model_metadata", columns=["Model Name","Task","Dataset","Metric","Score","Saved At","File","Duration"])
        datasets_df = load_user_csv(username, "datasets", columns=["filename","type","size","date"])
        downloads_df = load_user_csv(username, "downloads", columns=["filename","type","date"])
        activity_df = load_user_csv(username, "activity", columns=["timestamp","emoji","action","duration"])

    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

    # Onboarding for new users
    if is_new or models_df.empty:
        st.markdown(f"<h2 style='font-weight:600; color:#4F46E5;'>👋 Welcome, {nickname}!</h2>", unsafe_allow_html=True)
        st.info("Get started by uploading a dataset or training your first model. Your dashboard will update as you use the app.")
    else:
        st.markdown(f"<h2 style='font-weight:600; color:#4F46E5;'><i class='fas fa-user-circle'></i> {nickname}’s Dashboard</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown("#### Profile")
        if avatar:
            st.image(avatar, width=100)
        else:
            st.image("https://placehold.co/200x200/?text=User", width=100)
        uploaded_avatar = st.file_uploader("Change Avatar", type=["png","jpg","jpeg"], key="avatar_upload")
        if uploaded_avatar:
            img_bytes = uploaded_avatar.read()
            b64img = f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"
            profile["avatar"] = b64img
            save_user_profile(username, profile)
            log_user_activity(username, "Changed avatar", "🖼️")
            st.success("Avatar updated!")
            st.rerun()
        new_nick = st.text_input("Edit Nickname", value=nickname, key="nickname_input")
        if st.button("Save Nickname"):
            profile["nickname"] = new_nick
            save_user_profile(username, profile)
            log_user_activity(username, "Changed nickname", "📝")
            st.success("Nickname updated!")
            st.rerun()
        st.markdown(f"**Member Since:** {join_date}")
        st.markdown(f"**Models Trained:** {len(models_df)}")
        st.markdown(f"**Datasets Uploaded:** {len(datasets_df)}")
    with col2:
        st.markdown(f"#### Recent Models for {nickname}")
        if not models_df.empty:
            for _, row in models_df.tail(3).iloc[::-1].iterrows():
                with st.container():
                    st.markdown(f"**{row['Model Name']}**  ")
                    st.markdown(f"<span class='badge badge-primary'>{row['Task']}</span>  ", unsafe_allow_html=True)
                    st.markdown(f"Score: <b>{row['Score']}</b> | {row['Metric']} | {row['Saved At']}", unsafe_allow_html=True)
                    if 'Duration' in row and pd.notnull(row['Duration']):
                        st.markdown(f"⏱️ Training Duration: {row['Duration']}")
                    dcol1, dcol2, dcol3 = st.columns([1,1,1])
                    with dcol1:
                        file_path = os.path.join(MODEL_DIR, row['File'])
                        if os.path.exists(file_path):
                            st.markdown(download_model_button(file_path, "Download"), unsafe_allow_html=True)
                    with dcol2:
                        st.button("Reuse", key=f"reuse_{row['File']}")
                    with dcol3:
                        if st.button("Delete", key=f"delete_{row['File']}"):
                            delete_user_model(username, row['File'])
                            st.success(f"Deleted model: {row['Model Name']}")
                            st.rerun()
        else:
            st.info("No models trained yet.")

    with card_container():
        st.markdown(f"#### Activity Timeline for {nickname}")
        if not activity_df.empty:
            for _, item in activity_df.tail(10).iloc[::-1].iterrows():
                duration_str = f" ⏱️ {item['duration']}" if 'duration' in item and pd.notnull(item['duration']) else ""
                st.markdown(f"{item['emoji']} {item['action']}{duration_str} <span style='color:#B0B0B0; font-size:0.9em;'>({item['timestamp']})</span>", unsafe_allow_html=True)
        else:
            st.info("No recent activity.")

    col3, col4 = st.columns(2)
    with col3:
        with card_container():
            st.markdown(f"#### Uploaded Datasets by {nickname}")
            with st.spinner("Loading uploaded datasets..."):
                if not datasets_df.empty:
                    st.dataframe(datasets_df.tail(10).iloc[::-1], use_container_width=True, height=300)
                else:
                    st.info("No datasets uploaded yet.")
    with col4:
        with card_container():
            st.markdown(f"#### Download History for {nickname}")
            with st.spinner("Loading download history..."):
                if not downloads_df.empty:
                    st.dataframe(downloads_df.tail(10).iloc[::-1], use_container_width=True, height=300)
                else:
                    st.info("No downloads yet.")

    with card_container():
        with st.expander("User Settings"):
            theme_opt = st.selectbox("Theme Preference", ["🌚 Dark", "🌞 Light", "🔥 Warm"], index=["🌚 Dark", "🌞 Light", "🔥 Warm"].index(theme))
            default_task_opt = st.selectbox("Default ML Task", ["Classification", "Regression", "Clustering", "Time Series"], index=["Classification", "Regression", "Clustering", "Time Series"].index(default_task))
            preview_rows_opt = st.number_input("Dataset Preview Rows", min_value=5, max_value=50, value=int(preview_rows), step=1)
            if st.button("Save Settings"):
                profile["theme"] = theme_opt
                profile["default_task"] = default_task_opt
                profile["preview_rows"] = int(preview_rows_opt)
                save_user_profile(username, profile)
                log_user_activity(username, "Updated settings", "⚙️")
                st.success("Settings updated!")
                st.rerun()
            st.markdown(f"- **Theme:** {theme_opt}")
            st.markdown(f"- **Default ML Task:** {default_task_opt}")
            st.markdown(f"- **Dataset Preview Rows:** {preview_rows_opt}")
    st.markdown(f"<div class='footer'>Profile last updated on <b>{datetime.now().strftime('%B %d, %Y')}</b></div>", unsafe_allow_html=True)

# ----- Page Configuration -----
st.set_page_config(
    page_title="📊 Advanced AutoML App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Theme Selector Styles (Updated to match new palette) -----
theme_css = {
    "🌚 Dark": """
        body, .stApp {
            background-color: #181818;
            color: #f1f1f1;
            font-family: 'Inter', sans-serif; /* Changed font */
        }
        .stSidebar {
            background-color: #111122;
            color: #cccccc;
        }
        .stButton > button {
            background-color: #3A86FF; /* Primary Blue */
            color: white;
            border: none;
            border-radius: 8px; /* Slightly more rounded */
            padding: 0.6rem 1.2rem; /* Increased padding */
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #2A6CDA; /* Darker blue on hover */
        }
        .stTextInput>div>input,
        .stSelectbox>div>div>div,
        .stNumberInput>div>div>input { /* Added number input */
            background-color: #2b2b2b;
            color: white;
            border-radius: 8px;
            padding: 0.5rem;
            border: 1px solid #444;
        }
        .stExpander {
            border-radius: 8px;
            border: 1px solid #333;
            background-color: #222;
        }
        .stAlert {
            border-radius: 8px;
        }
    """,
    "🌞 Light": """
        body, .stApp {
            background-color: #F9FAFB; /* Light grey background */
            color: #333323; /* Dark text */
            font-family: 'Inter', sans-serif; /* Changed font */
        }
        .stSidebar {
            background-color: #FFFFFF; /* White sidebar */
            color: #333323;
            box-shadow: 2px 0 5px rgba(0,0,0,0.05); /* Subtle shadow */
        }
        .stButton > button {
            background-color: #3A86FF; /* Primary Blue */
            color: #fff;
            border-radius: 8px; /* Slightly more rounded */
            padding: 0.6rem 1.2rem; /* Increased padding */
            border: none;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #2A6CDA; /* Darker blue on hover */
        }
        .stTextInput>div>input,
        .stSelectbox>div>div>div,
        .stNumberInput>div>div>input { /* Added number input */
            background-color: #ffffff;
            color: #333323;
            border-radius: 8px;
            padding: 0.5rem;
            border: 1px solid #E0E0E0; /* Lighter border */
        }
        .stExpander {
            border-radius: 8px;
            border: 1px solid #E0E0E0;
            background-color: #FFFFFF;
        }
        .stAlert {
            border-radius: 8px;
        }
        .stMetric {
            background-color: #FFFFFF;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
    """,
    "🔥 Warm": """
        body, .stApp {
            background-color: #fff5e6;
            color: #4a3f35;
            font-family: 'Inter', sans-serif; /* Changed font */
        }
        .stSidebar {
            background-color: #ffe6cc;
            color: #4a3f35;
        }
        .stButton > button {
            background-color: #f0a500;
            color: #fff;
            border-radius: 8px; /* Slightly more rounded */
            padding: 0.6rem 1.2rem; /* Increased padding */
            border: none;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #d98e00;
        }
        .stTextInput>div>input,
        .stSelectbox>div>div>div,
        .stNumberInput>div>div>input { /* Added number input */
            background-color: #fff9f0;
            color: #4a3f35;
            border-radius: 8px;
            padding: 0.5rem;
            border: 1px solid #e0c095;
        }
        .stExpander {
            border-radius: 8px;
            border: 1px solid #e0c095;
            background-color: #fffaf0;
        }
        .stAlert {
            border-radius: 8px;
        }
    """
}

# ----- Model Save/Load Utilities -----
def save_trained_model_user(model, model_name, dataset_name, task_type, metric_name, metric_value, username, duration=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{model_name}_{task_type}_{timestamp}.pkl"
    filepath = os.path.join(MODEL_DIR, filename)
    import joblib
    try:
        joblib.dump(model, filepath)
        st.success(f"✅ Model file saved: `{filepath}`")
    except Exception as e:
        st.error(f"❌ Failed to save model file: {e}")
        return
    row = {
        "Model Name": model_name,
        "Task": task_type,
        "Dataset": dataset_name,
        "Metric": metric_name,
        "Score": round(metric_value, 4),
        "Saved At": timestamp,
        "File": filename,
        "Duration": duration or ""
    }
    append_user_csv(username, "model_metadata", row, columns=["Model Name","Task","Dataset","Metric","Score","Saved At","File","Duration"])
    st.success("✅ Model metadata updated.")

def load_model_metadata():
    """Load model metadata from CSV file."""
    if os.path.exists(METADATA_FILE):
        return pd.read_csv(METADATA_FILE)
    return pd.DataFrame()

def download_model_button(file_path, label="Download"):
    """Generate a download button for the model file."""
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        href = f'<a href="data:file/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}" style="background-color: #3A86FF; color: white; padding: 0.6rem 1.2rem; border-radius: 8px; text-decoration: none; display: inline-block; transition: all 0.3s ease-in-out;">{label}</a>'
        return href

# -----------------------------
# Helper Functions
# -----------------------------

def load_sample_data(name):
    """Load sample datasets based on the provided name."""
    if name == "Customer Churn (Classification)":
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
    elif name == "House Prices (Regression)":
        url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
        df = pd.read_csv(url)
    elif name == "Mall Customers (Clustering)":
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mall_customers.csv"
        df = pd.read_csv(url)
    elif name == "Air Passengers (Time Series)":
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        df = pd.read_csv(url)
        df.columns = ['ds', 'y']
    elif name == "Sample Image Classification":
        df = pd.DataFrame({
            "image_url": [
                "https://upload.wikimedia.org/wikipedia/commons/9/99/Black_cat_in_sunlight.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/f/f9/Phoenicopterus_ruber_in_São_Paulo_Zoo.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/5/55/Blue_Bird_February_2010-1.jpg"
            ],
            "label": ["Cat", "Cat", "Flamingo", "Bird"]
        })
    elif name == "Sample Text (NLP)":
        df = pd.DataFrame({
            "text": [
                "I love this product, it is amazing!",
                "Worst purchase ever, very disappointing.",
                "This item is okay, not the best but works.",
                "Excellent quality and fast shipping."
            ],
            "sentiment": ["positive", "negative", "neutral", "positive"]
        })
    else:
        df = None
    return df

def preprocess_classification(df, target_col):
    """Preprocess the dataset for classification tasks."""
    df = df.copy()
    thresh = len(df) * 0.6
    df.dropna(thresh=thresh, axis=1, inplace=True)
    df.dropna(inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col:
            df[col] = LabelEncoder().fit_transform(df[col])
    if df[target_col].dtype == 'object':
        df[target_col] = LabelEncoder().fit_transform(df[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def preprocess_regression(df, target_col):
    """Preprocess the dataset for regression tasks."""
    df = df.copy()
    thresh = len(df) * 0.6
    df.dropna(thresh=thresh, axis=1, inplace=True)
    df.dropna(inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def preprocess_clustering(df):
    """Preprocess the dataset for clustering tasks."""
    df = df.copy()
    thresh = len(df) * 0.6
    df.dropna(thresh=thresh, axis=1, inplace=True)
    df.dropna(inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def plot_classification_report(y_true, y_pred):
    """Plot the classification report as a DataFrame."""
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.background_gradient(cmap='Blues'), use_container_width=True, height=400)

def plot_confusion_mat(y_true, y_pred):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig, use_container_width=True)

def plot_regression_results(y_true, y_pred):
    """Plot actual vs predicted values for regression tasks."""
    fig = px.scatter(x=y_true, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted")
    fig.add_shape(type="line", x0=y_true.min(), y0=y_true.min(), x1=y_true.max(), y1=y_true.max(),
                  line=dict(color="red", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

def explain_model_choice(task_type):
    """Provide explanations for different model types based on the task."""
    if task_type == "Classification":
        st.markdown("""
        **Classification models explained:**  
        - Logistic Regression: simple linear classifier  
        - Random Forest: ensemble of decision trees, good general accuracy  
        - XGBoost: powerful gradient boosting, often top performer  
        - CatBoost: gradient boosting optimized for categorical data  
        - SVM: tries to find a hyperplane to separate classes  
        """)
    elif task_type == "Regression":
        st.markdown("""
        **Regression models explained:**  
        - Linear Regression: predicts continuous values linearly  
        - Random Forest Regressor: ensemble tree model for regression  
        - XGBoost Regressor: gradient boosting for regression  
        - CatBoost Regressor: gradient boosting optimized for categorical data  
        - SVR: support vector regression  
        """)
    elif task_type == "Clustering":
        st.markdown("""
        **Clustering algorithms explained:**  
        - KMeans: partition data into k groups  
        - DBSCAN: density-based clustering detecting noise/outliers  
        """)
    elif task_type == "Time Series":
        st.markdown("""
        **Time series forecasting:**  
        - Prophet: additive model by Facebook for easy forecasting  
        """)

def generate_pdf_report(model_name, dataset_name, task_type, metric_name, metric_value):
    """Generate a PDF report for the trained model."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="AutoML Model Report", ln=True, align='C')

    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Model Name: {model_name}", ln=True)
    pdf.cell(200, 10, txt=f"Task Type: {task_type}", ln=True)
    pdf.cell(200, 10, txt=f"Dataset: {dataset_name}", ln=True)
    pdf.cell(200, 10, txt=f"{metric_name}: {round(metric_value, 4)}", ln=True)
    pdf.cell(200, 10, txt=f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    report_path = os.path.join(MODEL_DIR, f"{model_name}_{task_type}_report.pdf")
    pdf.output(report_path)
    return report_path

def save_model(model, filename):
    """Saves a trained model to the MODEL_DIR."""
    filepath = os.path.join(MODEL_DIR, filename)
    try:
        joblib.dump(model, filepath)
        st.success(f"Model saved successfully to {filepath}")
    except Exception as e:
        st.error(f"Error saving model: {e}")

def plot_k_distance(df):
    """Plots the k-distance graph for DBSCAN."""
    from sklearn.neighbors import NearestNeighbors
    # Ensure df contains only numerical data for distance calculation
    df_numeric = df.select_dtypes(include=np.number)
    if df_numeric.empty:
        st.warning("No numeric columns found for k-distance plot.")
        return

    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(df_numeric)
    distances, indices = nbrs.kneighbors(df_numeric)
    distances = np.sort(distances[:, 1], axis=0)

    fig, ax = plt.subplots()
    ax.plot(distances)
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel("Epsilon")
    ax.set_title("K-distance Graph for DBSCAN (Elbow Method)")
    st.pyplot(fig, use_container_width=True)


# -----------------------------
# Workflow Functions
# -----------------------------

def classification_workflow(data, uploaded_file, use_sample):
    """Classification model training and evaluation workflow."""
    st.container()
    st.header("Classification Workflow")

    with st.expander("📋 Classification Help"):
        st.markdown("""
        Train classification models (e.g., Random Forest, XGBoost) using your dataset.
        **Steps:**
        1. **Select Target Column:** Choose the column your model will predict.
        2. **Explain Models:** Get a brief description of available classification algorithms.
        3. **Train Models:** Initiate the training process. The app will automatically tune and compare several models.
        4. **Review Results:** See the accuracy, confusion matrix, and classification report for the best performing model.
        5. **Save Model:** Optionally save the trained model and its metadata for later use or deployment.
        6. **Generate PDF Report:** Create a PDF summary of the model's performance.
        """)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Select target column 🎯", data.columns, help="The column containing the labels your model will predict.")
    with col2:
        if st.button("Explain Models 💡"):
            explain_model_choice("Classification")
    
    st.markdown("---")

    X, y = preprocess_classification(data, target_col)
    st.write(f"Features: {list(X.columns)}")
    st.write(f"Target classes: {list(np.unique(y))}")

    st.markdown("---")
    st.subheader("⚙️ Training Configuration")
    col_config1, col_config2 = st.columns(2)
    with col_config1:
        test_size = st.slider("Test set size (%)", 10, 50, 20, help="Percentage of data to reserve for testing the model.")
    with col_config2:
        random_state = st.number_input("Random seed (for reproducibility)", value=42, step=1, help="Seed for random number generation to ensure reproducible results.")

    st.markdown("---")
    if st.button("🚀 Train Models"):
        with st.spinner("Training models... This may take a few moments."):
            results = {}

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100,
                                                                random_state=random_state,
                                                                stratify=y)

            param_grids = {
                "Logistic Regression": {
                    "model": LogisticRegression(max_iter=200),
                    "params": {
                        'C': [0.01, 0.1, 1, 10],
                        'solver': ['liblinear', 'lbfgs']
                    }
                },
                "Random Forest": {
                    "model": RandomForestClassifier(),
                    "params": {
                        'n_estimators': [100, 200],
                        'max_depth': [5, 10, None],
                        'min_samples_split': [2, 5]
                    }
                },
                "XGBoost": {
                    "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                    "params": {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 5, 7]
                    }
                },
                "CatBoost": {
                    "model": CatBoostClassifier(verbose=0),
                    "params": {
                        'iterations': [100, 200],
                        'depth': [4, 6],
                        'learning_rate': [0.01, 0.1]
                    }
                },
                "SVM": {
                    "model": SVC(probability=True),
                    "params": {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto']
                    }
                }
            }

            for name, mp in param_grids.items():
                st.text(f"Tuning {name}...")
                clf = GridSearchCV(mp["model"], mp["params"], cv=3, scoring='accuracy', n_jobs=-1)
                clf.fit(X_train, y_train)
                model = clf.best_estimator_
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                results[name] = (model, acc, preds)
                st.write(f"✅ {name} best params: {clf.best_params_}")
                st.write(f"✅ {name} accuracy: {acc:.4f}")

            acc_df = pd.DataFrame({k: v[1] for k, v in results.items()}, index=["Accuracy"]).T
            st.subheader("🏆 Model Accuracy Comparison")
            st.dataframe(acc_df, use_container_width=True, height=400)

            best_model_name = acc_df["Accuracy"].idxmax()
            best_model, best_acc, best_preds = results[best_model_name]

            st.session_state.trained_model = best_model
            st.session_state.trained_model_name = best_model_name
            st.session_state.trained_model_acc = best_acc

            st.info(f"Best model: **{best_model_name}** with accuracy: {best_acc:.4f}")
            
            st.markdown("---")
            st.subheader("📊 Best Model Evaluation")
            col_metrics1, col_metrics2 = st.columns(2)
            with col_metrics1:
                st.metric(label="Best Model Accuracy", value=f"{best_acc:.4f}", delta=None)
            with col_metrics2:
                st.subheader("Confusion Matrix")
                plot_confusion_mat(y_test, best_preds)
            
            st.subheader("Classification Report")
            plot_classification_report(y_test, best_preds)

    # Save model functionality
    if st.session_state.trained_model:
        st.markdown("---")
        st.subheader("💾 Save Trained Model")
        if st.button("✅ Confirm Save Model"):
            dataset_name = uploaded_file.name if uploaded_file else use_sample
            save_trained_model_user(
                st.session_state.trained_model,
                st.session_state.trained_model_name,
                dataset_name,
                "Classification",
                "Accuracy",
                st.session_state.trained_model_acc,
                st.session_state.get("username"),
                duration=None
            )
            log_user_activity(st.session_state.get("username"), f"Trained model {st.session_state.trained_model_name}", "🤖")
            st.success("🎉 Model saved and metadata recorded successfully!")
            st.rerun()

    if st.session_state.trained_model_name:
        st.markdown("---")
        st.subheader("📤 Export Model Report as PDF")
    
        if st.button("📄 Generate PDF Report"):
            dataset_name = uploaded_file.name if uploaded_file else use_sample
            report_file = generate_pdf_report(
                model_name=st.session_state.trained_model_name,
                dataset_name=dataset_name,
                task_type="Classification",
                metric_name="Accuracy",
                metric_value=st.session_state.trained_model_acc
            )
            with open(report_file, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.markdown(
                    f'<a href="data:application/pdf;base64,{b64}" download="{os.path.basename(report_file)}" style="background-color: #FFBE0B; color: white; padding: 0.6rem 1.2rem; border-radius: 8px; text-decoration: none; display: inline-block; transition: all 0.3s ease-in-out;">📄 Download Report</a>',
                    unsafe_allow_html=True
                )
    st.container() # End of Classification Workflow container

def regression_workflow(data, uploaded_file, use_sample):
    """Regression model training and evaluation workflow."""
    st.container()
    st.header("Regression Workflow")

    with st.expander("📋 Regression Help"):
        st.markdown("""
        Train regression models (e.g., Linear Regression, XGBoost Regressor) to predict continuous values.
        **Steps:**
        1. **Select Target Column:** Choose the continuous column your model will predict.
        2. **Explain Models:** Get a brief description of available regression algorithms.
        3. **Train Models:** Initiate the training process. The app will automatically tune and compare several models.
        4. **Review Results:** See the R² score, RMSE, and actual vs. predicted plots for the best performing model.
        5. **Save Model:** Optionally save the trained model and its metadata for later use or deployment.
        6. **Generate PDF Report:** Create a PDF summary of the model's performance.
        """)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Select target column 🎯", data.columns, help="The continuous column your model will predict.")
    with col2:
        if st.button("Explain Models 💡"):
            explain_model_choice("Regression")
    
    st.markdown("---")

    X, y = preprocess_regression(data, target_col)
    st.write(f"Features: {list(X.columns)}")

    st.markdown("---")
    st.subheader("⚙️ Training Configuration")
    col_config1, col_config2 = st.columns(2)
    with col_config1:
        test_size = st.slider("Test set size (%)", 10, 50, 20, help="Percentage of data to reserve for testing the model.")
    with col_config2:
        random_state = st.number_input("Random seed (for reproducibility)", value=42, step=1, help="Seed for random number generation to ensure reproducible results.")

    st.markdown("---")
    if st.button("🚀 Train Models"):
        with st.spinner("Training models... This may take a few moments."):
            results = {}

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state
            )

            param_grids = {
                "Linear Regression": {
                    "model": LinearRegression(),
                    "params": {}
                },
                "Random Forest Regressor": {
                    "model": RandomForestRegressor(),
                    "params": {
                        'n_estimators': [100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5]
                    }
                },
                "XGBoost Regressor": {
                    "model": XGBRegressor(),
                    "params": {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 5, 7]
                    }
                },
                "CatBoost Regressor": {
                    "model": CatBoostRegressor(verbose=0),
                    "params": {
                        'iterations': [100, 200],
                        'depth': [4, 6],
                        'learning_rate': [0.01, 0.1]
                    }
                },
                "SVR": {
                    "model": SVR(),
                    "params": {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto']
                    }
                }
            }

            for name, mp in param_grids.items():
                st.text(f"Tuning {name}...")

                if not mp["params"]:
                    model = mp["model"]
                    model.fit(X_train, y_train)
                elif name == "SVR":
                    model = RandomizedSearchCV(mp["model"], mp["params"], n_iter=4,
                                               cv=3, scoring='r2', n_jobs=-1, random_state=random_state)
                    model.fit(X_train, y_train)
                    st.write(f"✅ {name} best params: {model.best_params_}")
                    model = model.best_estimator_
                else:
                    model = GridSearchCV(mp["model"], mp["params"], cv=3,
                                         scoring='r2', n_jobs=-1)
                    model.fit(X_train, y_train)
                    st.write(f"✅ {name} best params: {model.best_params_}")
                    model = model.best_estimator_

                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                results[name] = (model, rmse, r2, preds)

            metrics = {
                "RMSE": {k: v[1] for k, v in results.items()},
                "R2 Score": {k: v[2] for k, v in results.items()}
            }
            metrics_df = pd.DataFrame(metrics)
            st.subheader("🏆 Model Performance Comparison")
            st.dataframe(metrics_df, use_container_width=True, height=400)

            best_model_name = metrics_df["R2 Score"].idxmax()
            best_model, best_rmse, best_r2, best_preds = results[best_model_name]

            st.session_state.regression_model = best_model
            st.session_state.regression_model_name = best_model_name
            st.session_state.regression_model_score = best_r2

            st.info(f"Best model: **{best_model_name}** with R2 score: {best_r2:.4f}")
            
            st.markdown("---")
            st.subheader("📊 Best Model Evaluation")
            col_metrics1, col_metrics2 = st.columns(2)
            with col_metrics1:
                st.metric(label="Best Model R2 Score", value=f"{best_r2:.4f}", delta=None)
                st.metric(label="Best Model RMSE", value=f"{best_rmse:.4f}", delta=None)
            with col_metrics2:
                st.subheader("Actual vs Predicted Plot")
                plot_regression_results(y_test, best_preds)

    # Save model functionality
    if st.session_state.regression_model:
        st.markdown("---")
        st.subheader("💾 Save Trained Regression Model")
        if st.button("✅ Confirm Save Model (Regression)"):
            dataset_name = uploaded_file.name if uploaded_file else use_sample
            save_trained_model_user(
                st.session_state.regression_model,
                st.session_state.regression_model_name,
                dataset_name,
                "Regression",
                "R2 Score",
                st.session_state.regression_model_score,
                st.session_state.get("username"),
                duration=None
            )
            log_user_activity(st.session_state.get("username"), f"Trained regression model {st.session_state.regression_model_name}", "📈")
            st.success("🎉 Regression model saved successfully!")
            st.rerun()

    if st.session_state.regression_model_name:
        st.markdown("---")
        st.subheader("📤 Export Model Report as PDF")
    
        if st.button("📄 Generate PDF Report"):
            dataset_name = uploaded_file.name if uploaded_file else use_sample
            report_file = generate_pdf_report(
                model_name=st.session_state.regression_model_name,
                dataset_name=dataset_name,
                task_type="Regression",
                metric_name="R2 Score",
                metric_value=st.session_state.regression_model_score
            )
            with open(report_file, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.markdown(
                    f'<a href="data:application/pdf;base64,{b64}" download="{os.path.basename(report_file)}" style="background-color: #FFBE0B; color: white; padding: 0.6rem 1.2rem; border-radius: 8px; text-decoration: none; display: inline-block; transition: all 0.3s ease-in-out;">📄 Download Report</a>',
                    unsafe_allow_html=True
                )
    st.container() # End of Regression Workflow container

def clustering_workflow(data):
    """Clustering analysis workflow."""
    st.container()
    st.header("Clustering Workflow")

    with st.expander("📋 Clustering Help"):
        st.markdown("""
        Perform unsupervised clustering to group similar data points.
        **Algorithms:**
        - **KMeans:** Partitions data into a pre-defined number of clusters (k). Use the Elbow Method and Silhouette Score to find the optimal 'k'.
        - **DBSCAN:** Density-based clustering that groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions.
        **Steps:**
        1. **Select Algorithm:** Choose between KMeans and DBSCAN.
        2. **Configure Parameters:** Adjust algorithm-specific parameters (e.g., `max_k` for KMeans, `eps` and `min_samples` for DBSCAN).
        3. **Run Clustering:** Execute the clustering process.
        4. **Visualize Results:** Explore the clustered data using scatter matrices and PCA projections.
        """)
    st.markdown("---")

    df = preprocess_clustering(data)
    st.write(f"Data shape: {df.shape}")

    algo_choice = st.selectbox("Select clustering algorithm", ["KMeans", "DBSCAN"])
    st.markdown("---")

    if algo_choice == "KMeans":
        st.subheader("⚙️ KMeans Configuration")
        col_kmeans1, col_kmeans2 = st.columns(2)
        with col_kmeans1:
            max_k = st.slider("Max clusters to try (Elbow Method)", 2, 15, 10, help="Maximum number of clusters to evaluate for KMeans.")
        with col_kmeans2:
            show_elbow = st.checkbox("Show Elbow Curve & Silhouette Score", value=True)

        if st.button("🚀 Run KMeans Clustering"):
            with st.spinner("Clustering with KMeans..."):
                inertias = []
                silhouette_scores = []

                for k in range(2, max_k + 1):
                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = model.fit_predict(df)
                    inertias.append(model.inertia_)
                    try:
                        sil_score = silhouette_score(df, labels)
                        silhouette_scores.append(sil_score)
                    except:
                        silhouette_scores.append(-1)

                best_k = np.argmax(silhouette_scores) + 2
                st.success(f"✅ Best number of clusters (by silhouette score): **{best_k}**")

                best_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                final_labels = best_model.fit_predict(df)
                df["KMeans Cluster"] = final_labels

                st.markdown("---")
                st.subheader("📊 KMeans Clustered Data Preview")
                st.dataframe(df.head(), use_container_width=True, height=400)

                st.markdown("---")
                st.subheader("📈 KMeans Cluster Visualizations")
                fig = px.scatter_matrix(df,
                                        dimensions=df.columns[:min(5, len(df.columns))],
                                        color="KMeans Cluster",
                                        title="KMeans Cluster Scatter Matrix")
                st.plotly_chart(fig, use_container_width=True)

                pca = PCA(n_components=2)
                components = pca.fit_transform(df.select_dtypes(include=[np.number]))
                fig2 = px.scatter(x=components[:, 0], y=components[:, 1],
                                  color=final_labels.astype(str),
                                  title="KMeans PCA 2D Projection")
                st.plotly_chart(fig2, use_container_width=True)

                if show_elbow:
                    st.markdown("---")
                    st.subheader("📉 Elbow Method & Silhouette Score")
                    fig3, ax = plt.subplots(1, 2, figsize=(12, 4)) # Increased figure size
                    ax[0].plot(range(2, max_k + 1), inertias, marker='o', color='#3A86FF')
                    ax[0].set_title("Inertia (Elbow Curve)")
                    ax[0].set_xlabel("Number of Clusters (k)")
                    ax[0].set_ylabel("Inertia")

                    ax[1].plot(range(2, max_k + 1), silhouette_scores, marker='x', color='#FFBE0B')
                    ax[1].set_title("Silhouette Scores")
                    ax[1].set_xlabel("Number of Clusters (k)")
                    ax[1].set_ylabel("Score")
                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=True)

    elif algo_choice == "DBSCAN":
        st.subheader("⚙️ DBSCAN Configuration")
        col_dbscan1, col_dbscan2 = st.columns(2)
        with col_dbscan1:
            eps_range = st.slider("Max epsilon to try", 0.2, 10.0, 2.0, step=0.2, help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
        with col_dbscan2:
            min_samples_range = st.slider("Max min_samples to try", 3, 20, 5, help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.")

        if st.checkbox("Plot DBSCAN k-distance elbow", help="Helps in visually determining an optimal 'eps' value."):
            plot_k_distance(df)

        if st.button("🚀 Run DBSCAN Tuning"):
            with st.spinner("Tuning DBSCAN..."):
                best_score = -1
                best_params = None
                best_labels = None

                for eps in np.arange(0.2, eps_range + 0.1, 0.2):
                    for min_samples in range(3, min_samples_range + 1):
                        model = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = model.fit_predict(df)
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                        if n_clusters >= 2:
                            try:
                                score = silhouette_score(df, labels)
                                if score > best_score:
                                    best_score = score
                                    best_params = (eps, min_samples)
                                    best_labels = labels
                            except:
                                continue

                if best_params:
                    eps, min_samples = best_params
                    st.success(f"✅ Best DBSCAN parameters: eps={eps:.2f}, min_samples={min_samples}, silhouette={best_score:.4f}")
                    df["DBSCAN Cluster"] = best_labels

                    st.markdown("---")
                    st.subheader("📊 DBSCAN Clustered Data Preview")
                    st.dataframe(df.head(), use_container_width=True, height=400)

                    st.markdown("---")
                    st.subheader("📈 DBSCAN Cluster Visualizations")
                    fig = px.scatter_matrix(df,
                                            dimensions=df.columns[:min(5, len(df.columns))],
                                            color="DBSCAN Cluster",
                                            title="DBSCAN Cluster Scatter Matrix")
                    st.plotly_chart(fig, use_container_width=True)

                    pca = PCA(n_components=2)
                    components = pca.fit_transform(df.select_dtypes(include=[np.number]))
                    fig2 = px.scatter(x=components[:, 0], y=components[:, 1],
                                      color=pd.Series(best_labels).astype(str),
                                      title="DBSCAN PCA 2D Projection")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.error("❌ DBSCAN failed to find at least 2 valid clusters.\n\nTry:\n- Increasing `eps`\n- Decreasing `min_samples`\n- Check k-distance elbow\n- Standardize input features")
    st.container() # End of Clustering Workflow container

def timeseries_workflow(data):
    """Time series forecasting workflow."""
    st.container()
    st.header("Time Series Forecasting Workflow")

    with st.expander("📋 Time Series Help"):
        st.markdown("""
        Forecast future values using time series data with the Prophet model.
        **Requirements:**
        - Your dataset must contain a 'ds' column (for dates/timestamps) and a 'y' column (for the values to forecast).
        **Steps:**
        1. **Configure Forecast:** Set the number of periods to forecast into the future.
        2. **Prophet Options:** Adjust seasonality mode, include holidays, and tune `changepoint_prior_scale` for model flexibility.
        3. **Run Forecasting:** Train the Prophet model and generate predictions.
        4. **Visualize Forecast:** See the forecast plot and its components (trend, seasonality).
        5. **Save Model:** Optionally save the trained Prophet model.
        """)
    st.markdown("---")

    # Check if required columns are present
    if "ds" not in data.columns or "y" not in data.columns:
        st.error("❌ Data must contain 'ds' (date) and 'y' (value) columns.")
        st.stop()

    data['ds'] = pd.to_datetime(data['ds'])
    st.subheader("📊 Sample Data Preview")
    st.dataframe(data.head(), use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("⚙️ Prophet Model Options")
    col_prophet1, col_prophet2 = st.columns(2)
    with col_prophet1:
        periods = st.number_input("🔮 Forecast periods (days)", min_value=1, max_value=365, value=30, help="Number of future periods (days) to forecast.")
        seasonality_mode = st.selectbox("Seasonality mode", ["additive", "multiplicative"], help="How seasonality is modeled (additive or multiplicative).")
    with col_prophet2:
        use_holidays = st.checkbox("Include country holidays?", help="Whether to include country-specific holidays in the model.")
        country = st.selectbox("Select country for holidays", ["IN", "US", "UK", "None"], help="Select a country if including holidays.") if use_holidays else None

    st.markdown("📈 *Tuning changepoint_prior_scale helps control overfitting*")
    cps_values = st.multiselect(
        "Changepoint prior scale values to try (higher = more flexibility)", 
        [0.001, 0.01, 0.05, 0.1, 0.5], default=[0.01, 0.1],
        help="Controls the flexibility of the trend. Higher values allow the trend to be more flexible."
    )

    st.markdown("---")
    if st.button("🚀 Run Forecasting"):
        with st.spinner("⏳ Training Prophet model(s)..."):
            best_model = None
            best_score = float("inf")
            best_forecast = None
            best_params = {}

            for cps in cps_values:
                try:
                    m = Prophet(
                        seasonality_mode=seasonality_mode,
                        changepoint_prior_scale=cps
                    )
                    if use_holidays and country != "None":
                        m.add_country_holidays(country_name=country)

                    m.fit(data.copy())
                    future = m.make_future_dataframe(periods=periods)
                    forecast = m.predict(future)

                    # Evaluate performance on historical data
                    y_true = data["y"]
                    y_pred = forecast.loc[:len(data)-1, "yhat"]
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = mean_squared_error(y_true, y_pred, squared=False)

                    if rmse < best_score:
                        best_score = rmse
                        best_model = m
                        best_forecast = forecast
                        best_params = {"cps": cps, "mae": mae, "rmse": rmse}
                except Exception as e:
                    st.warning(f"⚠️ Model failed for cps={cps}: {e}")

            if best_model:
                st.success(f"✅ Best Model: cps={best_params['cps']} | MAE={best_params['mae']:.2f}, RMSE={best_params['rmse']:.2f}")
                
                st.markdown("---")
                st.subheader("📈 Forecast Output Preview")
                st.dataframe(best_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(), use_container_width=True, height=400)

                st.markdown("---")
                st.subheader("📊 Forecast Plot")
                fig1 = best_model.plot(best_forecast)
                st.pyplot(fig1, use_container_width=True)

                st.markdown("---")
                st.subheader("📉 Forecast Components")
                fig2 = best_model.plot_components(best_forecast)
                st.pyplot(fig2, use_container_width=True)

                # Save option
                st.markdown("---")
                st.subheader("💾 Save Trained Model")
                if st.checkbox("Save Model?", help="Save the trained Prophet model for future use."):
                    filename = st.text_input("Filename (.pkl)", f"prophet_model_{int(best_params['cps'] * 100)}.pkl", help="Enter a filename for the saved model.")
                    if st.button("Save Model"):
                        save_model(best_model, filename)
                        st.success("Model saved!")

            else:
                st.error("❌ No valid model was trained. Please try adjusting parameters.")
    st.container() # End of Time Series Workflow container

def image_workflow(data):
    """Basic image classification workflow."""
    st.container()
    st.header("🖼️ Image Dataset Explorer")

    with st.expander("📋 Image (Basic) Help"):
        st.markdown("""
        Explore your image dataset. This section is for basic visualization and exploration of image datasets.
        **Requirements:**
        - Your dataset must contain an 'image_url' column (for image links) and a 'label' column (for image categories).
        **Features:**
        - **Filter by Label:** View images belonging to specific categories.
        - **Search by URL:** Find images by keywords in their URLs.
        - **Pagination:** Browse through images in manageable pages.
        - **Explore by Index:** View individual images by their row index.
        """)
    st.markdown("---")

    # Basic validation
    if 'image_url' not in data.columns or 'label' not in data.columns:
        st.error("❌ Image dataset must contain 'image_url' and 'label' columns.")
        st.stop()

    st.subheader("🔍 Filter & Search Images")
    col_filter, col_search = st.columns(2)
    with col_filter:
        unique_labels = data['label'].unique()
        selected_label = st.selectbox("Filter by label", ["All"] + sorted(unique_labels.tolist()), help="Filter images by their assigned label.")
    with col_search:
        search_url = st.text_input("Search image by URL (optional)", help="Search for images containing specific keywords in their URL.")
    
    filtered_data = data if selected_label == "All" else data[data['label'] == selected_label]
    if search_url:
        filtered_data = filtered_data[filtered_data['image_url'].str.contains(search_url, case=False)]

    st.success(f"Found {len(filtered_data)} images matching criteria.")
    st.markdown("---")

    st.subheader("📸 Image Gallery")
    page_size = 5
    max_page = max(1, (len(filtered_data) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, help="Navigate through pages of images.")
    start = (page - 1) * page_size
    end = start + page_size

    for i, row in filtered_data.iloc[start:end].iterrows():
        st.image(row['image_url'], width=300, caption=f"Label: {row['label']}")
        st.markdown("---")

    st.subheader("📍 Explore Individual Image by Index")
    if len(data) > 0:
        index = st.slider("Select image index", 0, len(data)-1, 0, help="Select an image by its row index to view it individually.")
        row = data.iloc[index]
        st.image(row['image_url'], width=400, caption=f"Label: {row['label']}")
    st.container() # End of Image Workflow container

def text_workflow(data):
    """Basic text analysis workflow."""
    st.container()
    st.header("📝 Text Dataset Viewer & Sentiment Analysis")

    with st.expander("📋 Text (Basic) Help"):
        st.markdown("""
        Explore your text dataset and visualize sentiment distribution.
        **Requirements:**
        - Your dataset must contain a 'text' column (for the text content) and a 'sentiment' column (for text labels/categories).
        **Features:**
        - **Filter by Sentiment:** View texts belonging to specific sentiment categories.
        - **Search by Keyword:** Find texts containing specific keywords.
        - **Sentiment Distribution:** See a bar chart of sentiment counts.
        - **Word Cloud:** Generate a word cloud from the filtered text data to visualize common words.
        - **Text Explorer:** Browse through individual text entries with pagination.
        """)
    st.markdown("---")

    # Validation
    if 'text' not in data.columns or 'sentiment' not in data.columns:
        st.error("❌ Dataset must have 'text' and 'sentiment' columns.")
        st.stop()

    st.subheader("🔍 Filter & Search Texts")
    col_filter_text, col_search_text = st.columns(2)
    with col_filter_text:
        sentiments = data['sentiment'].unique().tolist()
        selected_sentiment = st.selectbox("Filter by Sentiment", ["All"] + sentiments, help="Filter text entries by their sentiment category.")
    with col_search_text:
        keyword = st.text_input("Search by keyword in text", help="Search for text entries containing specific keywords.")

    filtered_data = data if selected_sentiment == "All" else data[data['sentiment'] == selected_sentiment]
    if keyword:
        filtered_data = filtered_data[filtered_data['text'].str.contains(keyword, case=False)]

    st.success(f"{len(filtered_data)} texts matched.")
    st.markdown("---")

    st.subheader("📊 Sentiment Distribution")
    st.bar_chart(data['sentiment'].value_counts(), use_container_width=True)

    st.markdown("---")
    st.subheader("☁️ Word Cloud (Filtered Data)")
    if not filtered_data.empty:
        wc_text = " ".join(filtered_data['text'].astype(str).tolist())
        wordcloud = WordCloud(width=800, height=300, background_color='white').generate(wc_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("No text found for word cloud based on current filters.")

    st.markdown("---")
    st.subheader("📖 Text Explorer")
    page_size = 5
    max_page = max(1, (len(filtered_data) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, help="Navigate through pages of text entries.")
    start = (page - 1) * page_size
    end = start + page_size

    for i, row in filtered_data.iloc[start:end].iterrows():
        st.markdown(f"**Text {i}**")
        st.write(row['text'])
        st.markdown(f"**Sentiment:** `{row['sentiment']}`")
        st.markdown("---")

    st.subheader("🔎 View Individual Text by Index")
    index = st.slider("Select index", 0, len(data) - 1, 0, help="Select a text entry by its row index to view it individually.")
    row = data.iloc[index]
    st.info(f"**Text:** {row['text']}")
    st.success(f"**Sentiment:** {row['sentiment']}")
    st.container() # End of Text Workflow container

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#features by rohit section
#-----------------------------------------------------------------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------
#shap_analysis_section
#-------------------------------------------------------------------------------------------------

def shap_analysis_section():
    st.container()
    st.title("🧬 SHAP Model Explainability & Feature Impact Simulator")

    with st.expander("📋 SHAP Analysis Help"):
        st.markdown("""
        This section explains model predictions using SHAP (SHapley Additive exPlanations) visualizations.
        SHAP values help you understand how each feature contributes to a model's prediction for a specific instance,
        and also globally across the dataset.
        **Steps:**
        1. **Select a Saved Model:** Choose a model you previously trained and saved.
        2. **Upload Original Dataset:** Upload the exact dataset that was used to train the selected model. This is crucial for accurate SHAP calculations.
        3. **Select Target Column:** Specify the target column from your uploaded dataset.
        4. **View SHAP Visualizations:** Explore various SHAP plots:
            - **Summary Plot:** Shows overall feature importance.
            - **Beeswarm Plot:** Displays the distribution of SHAP values for each feature.
            - **Dependence Plot:** Illustrates the effect of a single feature on the prediction.
            - **Waterfall Plot:** Explains a single prediction by showing how each feature pushes the prediction from the base value to the final output.
        5. **Feature Impact Simulation:** Interactively change feature values for a selected instance and observe the real-time impact on the prediction and its SHAP explanation.
        """)
    st.markdown("---")

    st.markdown("### 📁 Step 1: Select a Saved Model")
    username = st.session_state.get("username", "anonymous")
    meta_df = load_user_csv(username, "model_metadata", columns=["Model Name","Task","Dataset","Metric","Score","Saved At","File","Duration"])

    if meta_df.empty:
        st.warning("No saved models found. Train and save a model first.")
        return

    model_file = st.selectbox("📦 Choose Model File", meta_df["File"].tolist(), help="Select a previously saved model to analyze.")
    model_path = os.path.join(MODEL_DIR, model_file)
    task = meta_df[meta_df["File"] == model_file]["Task"].values[0]

    st.markdown("---")
    st.markdown("### 📂 Step 2: Upload the Dataset Used to Train the Model")
    dataset_file = st.file_uploader("Upload original training dataset (CSV/XLSX)", type=["csv", "xlsx"], help="Upload the exact dataset used to train the selected model for accurate SHAP calculations.")

    if dataset_file:
        try:
            df = pd.read_csv(dataset_file) if dataset_file.name.endswith("csv") else pd.read_excel(dataset_file)
            st.success("✅ Dataset loaded successfully")
            st.dataframe(df.head(), use_container_width=True, height=400)
        except Exception as e:
            st.error(f"❌ Failed to load dataset: {e}")
            return

        st.markdown("---")
        st.markdown("### 🎯 Step 3: Select Target Column")
        target_col = st.selectbox("Select Target Column", df.columns, help="The target column used when training the model.")
        model = joblib.load(model_path)

        if task == "Classification":
            X, y = preprocess_classification(df, target_col)
        elif task == "Regression":
            X, y = preprocess_regression(df, target_col)
        else:
            st.warning("Only classification and regression models are supported for SHAP analysis.")
            return

        st.markdown("---")
        st.markdown("### 📊 SHAP Visualizations")
        
        with st.spinner("Calculating SHAP values... This may take a moment."):
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

        with st.expander("📍 SHAP Summary (Mean Absolute Value)"):
            st.markdown("This plot shows the average impact of each feature on the model's output magnitude. Features are ranked by importance.")
            fig1 = plt.figure(figsize=(10, 6)) # Adjusted figure size
            shap.plots.bar(shap_values, show=False)
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)

        with st.expander("🎯 SHAP Beeswarm Plot"):
            st.markdown("The Beeswarm plot shows how the SHAP values for each feature are distributed across all instances, providing insights into feature impact and direction. Red indicates higher feature values, blue indicates lower.")
            fig2 = plt.figure(figsize=(10, 6)) # Adjusted figure size
            shap.plots.beeswarm(shap_values, show=False)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

        with st.expander("🔍 SHAP Dependence Plot"):
            st.markdown("This plot shows the effect of a single feature on the prediction, often revealing interactions with other features. The vertical axis is the SHAP value for the selected feature.")
            feat = st.selectbox("Select feature for Dependence Plot", X.columns, help="Choose a feature to see its impact on predictions and potential interactions.")
            fig3 = plt.figure(figsize=(10, 6)) # Adjusted figure size
            shap.dependence_plot(feat, shap_values.values, X, show=False)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

        with st.expander("🌊 SHAP Waterfall Plot"):
            st.markdown("The Waterfall plot explains a single prediction by showing how each feature pushes the prediction from the base value (average prediction) to the final output. Features are ordered by their impact.")
            index = st.slider("Pick a sample row for Waterfall Plot", 0, len(X)-1, 0, help="Select a specific data point (row) to see its prediction explained.")
            fig4 = plt.figure(figsize=(10, 6)) # Adjusted figure size
            shap.plots.waterfall(shap_values[index], show=False)
            plt.tight_layout()
            st.pyplot(fig4, use_container_width=True)

        st.markdown("---")
        with st.expander("🔁 Feature Impact Simulation"):
            st.markdown("Change the value of any feature for a selected instance and see the real-time impact on the prediction and its SHAP explanation:")
            
            sim_index = st.slider("Select sample row for simulation", 0, len(X)-1, 0, key="sim_index", help="Choose a data point to simulate changes on.")
            row = X.iloc[[sim_index]].copy()
            st.write("Original Feature Values for selected row:")
            st.dataframe(row, use_container_width=True)

            col_to_change = st.selectbox("Feature to modify", X.columns, key="sim_feat", help="Select the feature whose value you want to change.")
            new_val = st.number_input(f"New value for '{col_to_change}'", value=float(row[col_to_change].values[0]), key="sim_val", help="Enter the new value for the selected feature.")

            new_row = row.copy()
            new_row[col_to_change] = new_val
            
            if st.button("Simulate Impact"):
                new_pred = model.predict(new_row)[0]
                st.success(f"📈 New Prediction: **{new_pred:.4f}**")

                new_shap = explainer(new_row)
                fig5 = plt.figure(figsize=(10, 6)) # Adjusted figure size
                shap.plots.waterfall(new_shap[0], show=False)
                plt.tight_layout()
                st.pyplot(fig5, use_container_width=True)
    st.container() # End of SHAP Analysis container


#-------------------------------------------------------------------------------------------------
#visual_insights_dashboard
#-------------------------------------------------------------------------------------------------
def visual_insights_dashboard():
    st.container()
    st.title("📊 Visual Insights Dashboard")
    st.caption("All-in-one interface for data analysis, model performance, and prediction interpretation.")

    with st.expander("📋 Visual Insights Dashboard Help"):
        st.markdown("""
        This dashboard provides a comprehensive overview of your dataset and the performance of your trained models.
        **Sections:**
        - **Dataset Overview:** View raw data, data types, missing values, and distributions of categorical and numerical features.
        - **Model Training Summary:** Get a quick summary of the selected model's type, task, and overall score. For classification, see Confusion Matrix and Classification Report; for regression, see Actual vs. Predicted plot and R² / RMSE scores.
        - **SHAP Explainability:** Dive deep into model interpretability with various SHAP plots (Feature Importance, Beeswarm, Dependence, Waterfall) and an interactive Feature Impact Simulator.
        **Steps:**
        1. **Select a Model:** Choose one of your previously saved models.
        2. **Upload Original Dataset:** Provide the dataset that was used to train the selected model.
        3. **Select Target Column:** Identify the target column from your dataset.
        4. **Explore:** Navigate through the different sections to gain insights into your data and model.
        """)
    st.markdown("---")

    meta_df = load_user_csv(st.session_state.get("username", "anonymous"), "model_metadata", columns=["Model Name","Task","Dataset","Metric","Score","Saved At","File","Duration"])
    if meta_df.empty:
        st.warning("❌ No saved models available. Please train and save a model first.")
        return

    st.markdown("### 📦 Select a Model to Analyze")
    selected_model_file = st.selectbox("Choose a model file", meta_df["File"].tolist(), help="Select a saved model to load its details and analyze its performance.")
    selected_meta = meta_df[meta_df["File"] == selected_model_file].iloc[0]
    model_path = os.path.join("saved_models", selected_model_file)
    task = selected_meta["Task"]

    model = joblib.load(model_path)

    st.markdown("---")
    st.markdown("### 📁 Upload the Original Training Dataset")
    uploaded_file = st.file_uploader("Upload original training dataset (CSV/XLSX)", type=["csv", "xlsx"], help="Upload the dataset that was used to train the selected model.")
    if not uploaded_file:
        st.info("📥 Please upload the dataset that was used to train the selected model.")
        return

    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    st.markdown("---")
    st.markdown("## 🔍 Dataset Overview")

    with st.expander("View Raw Data & Basic Stats"):
        st.dataframe(df.head(), use_container_width=True, height=400)
        st.markdown("### 📊 Column Data Types")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Type"]), use_container_width=True, height=400)

        st.markdown("### 🧯 Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(8, 4)) # Adjusted figure size
        sns.heatmap(df.isnull(), cbar=False, cmap="YlOrRd", ax=ax)
        plt.title("Missing Values Heatmap")
        st.pyplot(fig, use_container_width=True)

        if df.select_dtypes(include='object').shape[1] > 0:
            st.markdown("### 🧮 Categorical Value Counts")
            cat_col = st.selectbox("Select categorical column for value counts", df.select_dtypes(include='object').columns, help="Choose a categorical column to visualize its distribution.")
            value_counts_df = df[cat_col].value_counts().reset_index()
            value_counts_df.columns = [cat_col, 'count']
            fig = px.bar(value_counts_df, x=cat_col, y='count', color=cat_col, title=f"Value Counts for {cat_col}")
            st.plotly_chart(fig, use_container_width=True)


        st.markdown("### 📈 Numeric Distribution")
        num_col = st.selectbox("Select numeric feature for distribution", df.select_dtypes(include=np.number).columns, help="Choose a numeric column to visualize its distribution.")
        fig = px.histogram(df, x=num_col, nbins=30, color_discrete_sequence=["#3A86FF"], title=f"Distribution of {num_col}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("## 🤖 Model Training Summary")

    target_col = st.selectbox("🎯 Select target column used during training", df.columns, help="The target column that the model was trained to predict.")
    if task == "Classification":
        X, y = preprocess_classification(df, target_col)
    else:
        X, y = preprocess_regression(df, target_col)

    st.markdown("### ⚙️ Model Type & Performance")
    col_model_info1, col_model_info2 = st.columns(2)
    with col_model_info1:
        st.metric(label="Model Name", value=selected_meta['Model Name'])
        st.metric(label="Task Type", value=selected_meta['Task'])
    with col_model_info2:
        st.metric(label=f"Overall Score ({selected_meta['Metric']})", value=f"{selected_meta['Score']:.4f}")
        st.metric(label="Dataset Used", value=selected_meta['Dataset'])

    if task == "Classification":
        preds = model.predict(X)
        st.markdown("### 🧩 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6)) # Adjusted figure size
        sns.heatmap(confusion_matrix(y, preds), annot=True, fmt='d', cmap='YlGnBu', ax=ax)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig, use_container_width=True)

        st.markdown("### 📜 Classification Report")
        st.text(classification_report(y, preds))
    else:
        preds = model.predict(X)
        st.markdown("### 📊 Regression: Actual vs Predicted")
        fig = px.scatter(x=y, y=preds, labels={'x': "Actual Values", 'y': "Predicted Values"}, title="Actual vs Predicted Values")
        fig.add_shape(type='line', x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color='red', dash='dash'))
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"📈 R² Score: {r2_score(y, preds):.4f} | RMSE: {mean_squared_error(y, preds, squared=False):.4f}")

    st.markdown("---")
    st.markdown("## 🧠 SHAP Explainability")

    try:
        with st.spinner("Calculating SHAP values for interpretability..."):
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

        with st.expander("🔹 Feature Importance (Mean Absolute SHAP)"):
            st.markdown("This plot shows the average impact of each feature on the model's output magnitude. Features are ranked by importance.")
            fig = plt.figure(figsize=(10, 6)) # Adjusted figure size
            shap.plots.bar(shap_values, show=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with st.expander("🔹 Beeswarm Plot"):
            st.markdown("The Beeswarm plot displays the distribution of SHAP values for each feature, showing how each feature impacts the prediction for individual instances. Red indicates higher feature values, blue indicates lower.")
            fig = plt.figure(figsize=(10, 6)) # Adjusted figure size
            shap.plots.beeswarm(shap_values, show=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with st.expander("🔹 Dependence Plot"):
            st.markdown("This plot shows the effect of a single feature on the prediction, often revealing interactions with other features. The vertical axis is the SHAP value for the selected feature.")
            feature = st.selectbox("Pick a feature for Dependence Plot", X.columns, help="Choose a feature to see its impact on predictions and potential interactions.")
            fig = plt.figure(figsize=(10, 6)) # Adjusted figure size
            shap.dependence_plot(feature, shap_values[:, feature], X, show=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with st.expander("🔹 Waterfall Plot (Prediction Breakdown)"):
            st.markdown("The Waterfall plot explains a single prediction by showing how each feature pushes the prediction from the base value (average prediction) to the final output. Features are ordered by their impact.")
            index = st.slider("Select row index for Waterfall Plot", 0, len(X)-1, 0, help="Select a specific data point (row) to see its prediction explained.")
            fig = plt.figure(figsize=(10, 6)) # Adjusted figure size
            shap.plots.waterfall(shap_values[index], show=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with st.expander("🎛️ Feature Impact Simulator"):
            st.markdown("Change the value of any feature for a selected instance and see the real-time impact on the prediction and its SHAP explanation:")
            
            sim_index_viz = st.slider("Select sample row for simulation", 0, len(X)-1, 0, key="sim_index_viz", help="Choose a data point to simulate changes on.")
            row_viz = X.iloc[[sim_index_viz]].copy()
            st.write("Original Feature Values for selected row:")
            st.dataframe(row_viz, use_container_width=True)

            col_viz = st.selectbox("Feature to adjust", X.columns, key="sim_feat_viz", help="Select the feature whose value you want to change.")
            val_viz = st.number_input(f"New value for '{col_viz}'", value=float(row_viz[col_viz].values[0]), key="sim_val_viz", help="Enter the new value for the selected feature.")
            
            new_row_viz = row_viz.copy()
            new_row_viz[col_viz] = val_viz

            if st.button("Simulate Impact (Dashboard)"):
                new_pred_viz = model.predict(new_row_viz)[0]
                st.success(f"New Prediction after changing `{col_viz}`: **{new_pred_viz:.4f}**")
                new_shap_viz = explainer(new_row_viz)
                fig_viz = plt.figure(figsize=(10, 6)) # Adjusted figure size
                shap.plots.waterfall(new_shap_viz[0], show=False)
                plt.tight_layout()
                st.pyplot(fig_viz, use_container_width=True)

    except Exception as e:
        st.warning(f"SHAP explainability failed: {e}. Please ensure the dataset matches the model's training features.")
    st.container() # End of Visual Insights Dashboard container

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#feature_engineering_section
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
def feature_engineering_section():
    st.container()
    st.title("🧪 Feature Engineering Playground")
    st.caption("Explore, clean, transform, and evaluate features interactively with live plots and comparisons.")

    with st.expander("📋 Feature Engineering Playground Help"):
        st.markdown("""
        This section allows you to interactively explore, clean, and transform individual features in your dataset.
        **Features:**
        - **Feature Summary:** Get an overview of data types, missing values, and unique values for all columns.
        - **Feature Transformer:** Apply various transformations (Log, Z-Score, Min-Max Scale, Binning) and handle missing values (Mean, Median, Drop Rows) for selected numeric features.
        - **Visualize Before vs After:** See the immediate impact of transformations on feature distributions using histograms and box plots.
        - **Quick Model Performance Check:** Train a simple Random Forest Classifier using the transformed feature to get an idea of its predictive power.
        **Steps:**
        1. **Upload Dataset:** Load your CSV or XLSX file.
        2. **Select Numeric Feature:** Choose a numeric column to work with.
        3. **Apply Imputation/Transformation:** Select desired methods for missing value handling and data transformation.
        4. **Observe Visualizations:** Compare the original and transformed feature distributions.
        5. **(Optional) Check Model Performance:** Select a target column and train a quick model to see the transformed feature's impact on accuracy.
        """)
    st.markdown("---")

    file = st.file_uploader("📁 Upload a dataset (CSV/XLSX)", type=["csv", "xlsx"], help="Upload your dataset to start feature engineering.")
    if not file:
        st.info("Upload a dataset to begin.")
        return

    try:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        return

    st.success("✅ Data loaded.")
    st.markdown("### 📊 Feature Summary")

    with st.expander("🔍 View Data Snapshot and Stats"):
        st.dataframe(df.head(10), use_container_width=True, height=400)

        desc = pd.DataFrame({
            'Type': df.dtypes,
            'Missing %': df.isnull().mean() * 100,
            'Unique Values': df.nunique()
        })
        st.dataframe(desc.style.background_gradient(cmap='YlGn'), use_container_width=True, height=400)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric features found in your dataset.")
        return

    st.markdown("---")
    st.markdown("### 🧰 Feature Transformer")

    selected_feat = st.selectbox("Select a numeric feature to explore", numeric_cols, help="Choose a numeric column to apply transformations and handle missing values.")
    
    col_trans, col_impute = st.columns(2)
    with col_trans:
        transformation = st.radio("Transformation Type", ["None", "Log", "Z-Score", "Min-Max Scale", "Binning"], horizontal=True, help="Apply a mathematical transformation to the feature.")
    with col_impute:
        imputation = st.radio("Missing Value Handling", ["None", "Mean", "Median", "Drop Rows"], horizontal=True, help="Choose how to handle missing values in the selected feature.")

    transformed_df = df.copy()

    # Impute
    if imputation != "None":
        imputer_strategy = 'mean' if imputation == 'Mean' else 'median'
        if imputation == "Drop Rows":
            transformed_df = transformed_df.dropna(subset=[selected_feat])
            st.info(f"Dropped rows with missing values in '{selected_feat}'. New shape: {transformed_df.shape}")
        else:
            imputer = SimpleImputer(strategy=imputer_strategy)
            transformed_df[selected_feat] = imputer.fit_transform(transformed_df[[selected_feat]])
            st.info(f"Missing values in '{selected_feat}' imputed using {imputation}.")

    # Transform
    new_col = selected_feat + "_transformed"
    try:
        if transformation == "Log":
            transformed_df[new_col] = np.log1p(transformed_df[selected_feat])
            st.success(f"Applied Log transformation to '{selected_feat}'.")
        elif transformation == "Z-Score":
            transformed_df[new_col] = StandardScaler().fit_transform(transformed_df[[selected_feat]])
            st.success(f"Applied Z-Score standardization to '{selected_feat}'.")
        elif transformation == "Min-Max Scale":
            transformed_df[new_col] = MinMaxScaler().fit_transform(transformed_df[[selected_feat]])
            st.success(f"Applied Min-Max scaling to '{selected_feat}'.")
        elif transformation == "Binning":
            transformed_df[new_col] = pd.cut(transformed_df[selected_feat], bins=5, labels=False, duplicates='drop')
            st.success(f"Applied Binning (5 bins) to '{selected_feat}'.")
        else:
            transformed_df[new_col] = transformed_df[selected_feat]
            st.info("No transformation applied.")
    except Exception as e:
        st.error(f"Error transforming feature: {e}. Please check data for non-numeric values or zeros/negatives for log transform.")
        return

    st.markdown("---")
    st.markdown("### 📈 Visualize Before vs After")

    vis1, vis2 = st.columns(2)
    with vis1:
        st.markdown("**Original Feature Distribution**")
        fig1 = px.histogram(df, x=selected_feat, nbins=30, title=None, height=250, color_discrete_sequence=["#FFBE0B"])
        st.plotly_chart(fig1, use_container_width=True)
    with vis2:
        st.markdown("**Transformed Feature Distribution**")
        fig2 = px.histogram(transformed_df, x=new_col, nbins=30, title=None, height=250, color_discrete_sequence=["#3A86FF"])
        st.plotly_chart(fig2, use_container_width=True)

    box1, box2 = st.columns(2)
    with box1:
        st.markdown("**Original Feature Box Plot**")
        fig3 = px.box(df, y=selected_feat, height=250, color_discrete_sequence=["#FFBE0B"])
        st.plotly_chart(fig3, use_container_width=True)
    with box2:
        st.markdown("**Transformed Feature Box Plot**")
        fig4 = px.box(transformed_df, y=new_col, height=250, color_discrete_sequence=["#3A86FF"])
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚙️ Optional: Quick Model Performance Check")
    with st.expander("💡 Train a quick Random Forest Classifier with the transformed feature"):
        target_col = st.selectbox("Select target column for quick model", df.columns, help="Choose a target column to quickly assess the predictive power of the transformed feature.")

        if target_col == selected_feat:
            st.warning("⚠️ You selected the same feature for target and input. Please choose a different target column.")
            return

        if df[target_col].isnull().any():
            st.warning("Some missing values in target column. Rows with NaN will be dropped for model training.")
            transformed_df = transformed_df.dropna(subset=[target_col])

        if transformed_df[target_col].nunique() < 2:
            st.error("Target column must have at least two unique classes for classification. Please select a suitable target.")
            return

        model_df = transformed_df[[new_col, target_col]].dropna()
        X = model_df[[new_col]]
        y = model_df[target_col]

        if st.button("Run Quick Model"):
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
                st.success(f"✅ Accuracy with transformed feature: **{acc:.3f}**")
            except Exception as e:
                st.error(f"Failed to train quick model: {e}. Ensure target column is suitable for classification.")
    st.container() # End of Feature Engineering Playground container

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#behavioral_impact_analysis
#----------------------------------------------------------------------------------------------------------------------------------------------------------
def behavioral_impact_analysis():
    st.container()
    st.title("📈 Behavioral Impact Analysis")
    st.caption("Understand how behavioral patterns affect model predictions using comparison and SHAP analysis.")

    with st.expander("📋 Behavioral Impact Analysis Help"):
        st.markdown("""
        This section helps you analyze the impact of specific "behavioral" features on your model's predictions.
        It compares a model trained on all features against a model trained only on selected behavioral features,
        and allows for interactive simulation of behavioral changes.
        **Steps:**
        1. **Upload Dataset:** Load your CSV or XLSX file.
        2. **Select Target Column:** Choose the column your model will predict.
        3. **Select Behavioral Features:** Identify the features that represent behavioral patterns (e.g., user activity, engagement metrics).
        4. **Model Performance Comparison:** See how a model trained only on behavioral features performs compared to a model trained on all features.
        5. **Single Prediction Simulation:** Select a data point, modify its behavioral feature values, and observe how the prediction changes.
        6. **SHAP Explainability:** Use Waterfall plots to understand which features contribute most to the prediction before and after behavioral changes.
        """)
    st.markdown("---")

    file = st.file_uploader("📁 Upload the training dataset (CSV/XLSX)", type=["csv", "xlsx"], help="Upload your dataset for behavioral impact analysis.")
    if not file:
        st.info("Upload a dataset to begin behavioral analysis.")
        return

    try:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    st.success("✅ Dataset loaded successfully.")
    st.dataframe(df.head(), use_container_width=True, height=400)

    st.markdown("---")
    st.markdown("## 🎯 Select Target and Behavioral Features")
    target_col = st.selectbox("Select target column", df.columns, help="The column representing the outcome your model predicts.")
    
    # Check if target is valid
    if df[target_col].isnull().all():
        st.error("❌ All values in the target column are missing. Please check your dataset.")
        return
    elif df[target_col].isnull().any():
        st.warning("⚠️ Some rows with missing target values will be dropped for analysis.")
        df.dropna(subset=[target_col], inplace=True)

    behavior_feats = st.multiselect(
        "Select behavioral features (e.g., login frequency, session duration)",
        df.columns.drop(target_col).tolist(),
        help="Choose features that represent user behavior or actions."
    )

    if not behavior_feats:
        st.warning("Please select at least one behavioral feature to continue.")
        return

    df_processed = df.copy()
    le = LabelEncoder()
    for col in df_processed.select_dtypes(include='object'):
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))

    X_full = df_processed.drop(columns=[target_col])
    X_behave = df_processed[behavior_feats]
    y = df_processed[target_col]

    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_full, y, test_size=0.2, random_state=42)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_behave, y, test_size=0.2, random_state=42)

    model_all = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_behave = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    if st.button("🚀 Run Behavioral Analysis"):
        with st.spinner("Training models and performing analysis..."):
            model_all.fit(X_train_all, y_train_all)
            model_behave.fit(X_train_b, y_train_b)

        preds_all = model_all.predict(X_test_all)
        preds_b = model_behave.predict(X_test_b)

        results_df = pd.DataFrame({
            "Model": ["Full Feature Model", "Behavioral Feature Model"],
            "Accuracy": [accuracy_score(y_test_all, preds_all), accuracy_score(y_test_b, preds_b)],
            "F1 Score": [f1_score(y_test_all, preds_all, average="weighted"), f1_score(y_test_b, preds_b, average="weighted")],
            "Precision": [precision_score(y_test_all, preds_all, average="weighted"), precision_score(y_test_b, preds_b, average="weighted")],
            "Recall": [recall_score(y_test_all, preds_all, average="weighted"), recall_score(y_test_b, preds_b, average="weighted")]
        })

        st.markdown("---")
        st.markdown("## ⚖️ Model Performance Comparison")

        perf_col1, perf_col2 = st.columns([1, 1])

        with perf_col1:
            st.markdown("### 📋 Metrics Table")
            st.dataframe(
                results_df.style.highlight_max(axis=0, props='background-color: #06D6A0; color: white;'), # Highlight best
                use_container_width=True,
                height=250
            )

        with perf_col2:
            st.markdown("### 📊 Bar Comparison")
            fig = px.bar(
                results_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Metric',
                y='Score',
                color='Model',
                barmode='group',
                height=300, # Adjusted height
                title='Performance Metrics Comparison',
                color_discrete_map={"Full Feature Model": "#3A86FF", "Behavioral Feature Model": "#FFBE0B"} # Custom colors
            )
            fig.update_layout(margin=dict(t=30, b=30, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)


        st.markdown("---")
        st.markdown("## 🧪 Single Prediction Simulation")

        row_idx = st.slider("Pick a sample row for simulation", 0, min(50, len(X_test_b)-1), 0, help="Select a data point to observe the impact of behavioral changes.")
        row = X_test_b.iloc[[row_idx]].copy()
        st.write("🧾 Original Behavioral Feature Values:")
        st.dataframe(row, use_container_width=True)

        st.markdown("### Modify Behavioral Features:")
        mod_cols = st.columns(len(behavior_feats))
        new_row = row.copy()
        for i, feat in enumerate(behavior_feats):
            with mod_cols[i]:
                new_val = st.number_input(f"{feat}", value=float(row[feat].values[0]), key=f"mod_feat_{feat}", help=f"Change the value of {feat}.")
                new_row[feat] = new_val

        pred_old = model_behave.predict(row)[0]
        pred_new = model_behave.predict(new_row)[0]

        st.success(f"🎯 Prediction before change: `{pred_old}` → Prediction after change: `{pred_new}`")

        st.markdown("---")
        st.markdown("## 📊 SHAP Explainability of Behavioral Impact")

        with st.spinner("Calculating SHAP values for behavioral impact..."):
            explainer = shap.Explainer(model_behave, X_test_b)
            shap_old = explainer(row)
            shap_new = explainer(new_row)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 🔷 Prediction Explanation: Before Change")
            fig = plt.figure(figsize=(10, 6)) # Adjusted figure size
            shap.plots.waterfall(shap_old[0], show=False)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True, use_container_width=True)

        with col_b:
            st.markdown("### 🔶 Prediction Explanation: After Change")
            fig = plt.figure(figsize=(10, 6)) # Adjusted figure size
            shap.plots.waterfall(shap_new[0], show=False)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True, use_container_width=True)

        st.markdown("### 🔍 SHAP Feature Impact (After Change)")
        fig2 = plt.figure(figsize=(10, 4)) # Adjusted figure size
        shap.plots.bar(shap_new, show=False)
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True, use_container_width=True)

        st.success("✅ Interactive behavioral simulation complete.")
    st.container() # End of Behavioral Impact Analysis container

#--------------------------------------------------------------------------------------------------------------------------------------------------------
#image_model_training_section
#--------------------------------------------------------------------------------------------------------------------------------------------------------
def image_model_training_section():
    st.container()
    st.title("🧠 Image Model Training (AutoML - Vision)")
    st.markdown("Train deep learning models on your image dataset using Transfer Learning or CNN. Upload images, configure training, and evaluate results.")

    with st.expander("📋 Image Model Training Help"):
        st.markdown("""
        Train deep learning models for image classification.
        **Dataset Formats:**
        - **CSV with image URLs:** Provide a CSV with 'image_url' and 'label' columns. Images will be downloaded.
        - **ZIP with folders:** Upload a ZIP file where each folder name is a class label, and images are inside.
        **Steps:**
        1. **Upload Dataset:** Choose your dataset format and upload the file.
        2. **Configure Preprocessing:** Set image size, color mode (RGB/Grayscale), and normalization.
        3. **Model & Training Setup:** Select model type (Basic CNN or ResNet50 for Transfer Learning), epochs, batch size, and train/test split.
        4. **Build & Train Model:** Start the training process. Model summary, training history (accuracy/loss plots), and class distribution will be shown.
        5. **Evaluation Report:** View classification report, confusion matrix, and sample predictions.
        6. **Download Model:** Save the trained Keras model (.h5 file).
        """)
    st.markdown("---")

    temp_dir = tempfile.mkdtemp()

    st.markdown("### 📥 Step 1: Upload Dataset")
    dataset_type = st.radio("Select dataset format", ["CSV with image URLs", "ZIP with folders (label = folder name)"], help="Choose how your image dataset is provided.")

    if dataset_type == "CSV with image URLs":
        csv_file = st.file_uploader("Upload CSV (must contain 'image_url' and 'label' columns)", type=["csv"], help="Upload a CSV file where each row contains an image URL and its corresponding label.")
        if csv_file:
            df = pd.read_csv(csv_file)
            st.dataframe(df.head(), use_container_width=True, height=200)
            col_img_csv, col_label_csv = st.columns(2)
            with col_img_csv:
                img_col = st.selectbox("Image URL column", df.columns, help="Select the column containing image URLs.")
            with col_label_csv:
                label_col = st.selectbox("Label column", df.columns, help="Select the column containing image labels.")

            st.info("Downloading images from URLs into temporary folders... This may take a while for large datasets.")
            progress_bar = st.progress(0)
            total_images = len(df)
            downloaded_count = 0
            for i, row in df.iterrows():
                try:
                    image = Image.open(requests.get(row[img_col], stream=True, timeout=5).raw).convert("RGB")
                    folder = os.path.join(temp_dir, str(row[label_col]))
                    os.makedirs(folder, exist_ok=True)
                    image.save(os.path.join(folder, f"{i}.jpg"))
                    downloaded_count += 1
                except Exception as e:
                    # st.warning(f"Could not download image {row[img_col]}: {e}") # Too verbose for large datasets
                    pass
                progress_bar.progress((i + 1) / total_images)
            st.success(f"Images downloaded and grouped by label. Successfully downloaded {downloaded_count} of {total_images} images.")
        else:
            st.stop()
    else: # ZIP with folders
        zip_file = st.file_uploader("Upload a ZIP file of folders (e.g., Cat/, Dog/, etc.)", type=["zip"], help="Upload a ZIP file where each folder inside represents a class label and contains images for that class.")
        if zip_file:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            st.success("ZIP extracted successfully.")
        else:
            st.stop()

    st.markdown("---")
    st.markdown("### ⚙️ Step 2: Configure Preprocessing")
    col1, col2 = st.columns(2)
    with col1:
        img_size = st.slider("Resize images to (square)", 64, 256, 128, help="Images will be resized to this square dimension.")
        color_mode = st.radio("Color Mode", ["rgb", "grayscale"], help="Choose 'rgb' for color images or 'grayscale' for black and white.")
    with col2:
        normalize = st.checkbox("Normalize pixel values (0-1)", value=True, help="Scale pixel values from 0-255 to 0-1. Recommended for neural networks.")

    st.markdown("---")
    st.markdown("### 🤖 Step 3: Model & Training Setup")
    col3, col4 = st.columns(2)
    with col3:
        model_type = st.selectbox("Model type", ["Basic CNN", "ResNet50 (Transfer Learning)"], help="Choose between a simple Convolutional Neural Network or a pre-trained ResNet50 for transfer learning.")
        epochs = st.slider("Epochs", 1, 50, 10, help="Number of times the model will iterate over the entire training dataset.")
    with col4:
        batch_size = st.slider("Batch Size", 8, 64, 16, help="Number of samples per gradient update.")
        split = st.slider("Train/Test Split", 0.5, 0.95, 0.8, help="Proportion of the dataset to use for training (remaining for validation).")

    # Data Generators
    datagen = ImageDataGenerator(
        rescale=1./255 if normalize else None,
        validation_split=1 - split,
        preprocessing_function=preprocess_input if model_type.startswith("ResNet") else None
    )

    target_size = (img_size, img_size)
    channels = 3 if color_mode == "rgb" else 1
    input_shape = (img_size, img_size, channels)

    try:
        train_gen = datagen.flow_from_directory(
            temp_dir, target_size=target_size, color_mode=color_mode,
            batch_size=batch_size, class_mode="categorical", subset="training"
        )
        val_gen = datagen.flow_from_directory(
            temp_dir, target_size=target_size, color_mode=color_mode,
            batch_size=batch_size, class_mode="categorical", subset="validation"
        )
    except Exception as e:
        st.error(f"Error creating image data generators. Ensure your dataset structure is correct and contains images. Error: {e}")
        st.stop()

    num_classes = len(train_gen.class_indices)
    if num_classes == 0:
        st.error("No classes found in the dataset. Please check your image folders/labels.")
        st.stop()
    if train_gen.samples == 0:
        st.error("No training samples found. Adjust split or check dataset.")
        st.stop()
    if val_gen.samples == 0:
        st.warning("No validation samples found. Adjust split or check dataset.")


    st.markdown("---")
    st.markdown("### 🧠 Step 4: Build & Train Model")

    if model_type.startswith("Basic"):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
    else:
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False
        model = Sequential([
            base_model,
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax")
        ])

    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    st.subheader("Model Summary")
    model.summary(print_fn=lambda x: st.text(x)) # Use st.text for better rendering of summary

    if st.button("🚀 Start Training"):
        with st.spinner("Training in progress... This may take a while depending on epochs and dataset size."):
            history = model.fit(
                train_gen, epochs=epochs, validation_data=val_gen,
                callbacks=[EarlyStopping(patience=3)], verbose=1
            )
        st.success("Training completed!")

        st.markdown("---")
        st.markdown("### 📈 Training History")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history['accuracy'], label='Train Acc', color='#3A86FF')
        ax[0].plot(history.history['val_accuracy'], label='Val Acc', color='#FFBE0B')
        ax[0].legend()
        ax[0].set_title("Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Accuracy")

        ax[1].plot(history.history['loss'], label='Train Loss', color='#3A86FF')
        ax[1].plot(history.history['val_loss'], label='Val Loss', color='#FFBE0B')
        ax[1].legend()
        ax[1].set_title("Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🗂️ Class Distribution")
        labels = list(train_gen.class_indices.keys())
        class_counts = np.bincount(train_gen.classes)
        fig2 = px.bar(x=labels, y=class_counts, labels={"x": "Label", "y": "Count"}, title="Training Samples per Class", color_discrete_sequence=["#06D6A0"])
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Evaluation Report")
        y_true, y_pred = [], []
        # Ensure val_gen is not empty before iterating
        if val_gen.samples > 0:
            val_gen.reset() # Reset generator to ensure consistent order
            for i in range(len(val_gen)):
                X_batch, y_batch = val_gen[i]
                preds_batch = model.predict(X_batch)
                y_true.extend(np.argmax(y_batch, axis=1))
                y_pred.extend(np.argmax(preds_batch, axis=1))
                # Break condition to avoid infinite loop if generator is not finite
                if i >= val_gen.samples // batch_size and val_gen.samples % batch_size == 0:
                    break
                elif i >= val_gen.samples // batch_size and val_gen.samples % batch_size != 0:
                    break
        else:
            st.warning("No validation data available for evaluation. Skipping Classification Report and Confusion Matrix.")
            # Dummy data to prevent error in classification_report if val_gen is empty
            y_true = [0, 1]
            y_pred = [0, 1]
            labels = ["Class 0", "Class 1"] # Placeholder labels

        if val_gen.samples > 0: # Only show if there was actual validation data
            st.text(classification_report(y_true, y_pred, target_names=labels))
            cm = confusion_matrix(y_true, y_pred)
            fig3, ax3 = plt.subplots(figsize=(8, 6)) # Adjusted figure size
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='YlGnBu')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            st.pyplot(fig3, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🔍 Sample Predictions")
            val_gen.reset() # Reset again for sample predictions
            X_sample, y_sample = next(val_gen) # Get one batch
            preds = model.predict(X_sample)
            pred_labels = np.argmax(preds, axis=1)
            true_labels = np.argmax(y_sample, axis=1)

            for i in range(min(5, len(X_sample))): # Show up to 5 samples
                st.image(X_sample[i], width=200, caption=f"Predicted: {labels[pred_labels[i]]} | True: {labels[true_labels[i]]}")
        else:
            st.info("No validation samples to show predictions.")

        st.markdown("---")
        st.markdown("### 💾 Download Trained Model")
        model_path = os.path.join(temp_dir, "image_model.h5")
        model.save(model_path)
        with open(model_path, "rb") as f:
            st.download_button("📥 Download .h5 Model", f, file_name="image_model.h5", help="Download the trained Keras model file (.h5).")

    shutil.rmtree(temp_dir, ignore_errors=True)
    st.container() # End of Image Model Training container

#---------------------------------------------------------------------------------------------------------------------------------------------------------
#smart_prediction_studio
#---------------------------------------------------------------------------------------------------------------------------------------------------------
def smart_prediction_studio():
    st.container()
    st.title("🔍 Model Deployment & Smart Prediction Studio")
    st.caption("Load a trained model, upload new data, and predict with preprocessing and evaluation logic.")

    with st.expander("📋 Smart Prediction Studio Help"):
        st.markdown("""
        This studio allows you to deploy your saved models for making new predictions.
        **Steps:**
        1. **Select a Saved Model:** Choose a model from your saved models list. Its metadata will be displayed.
        2. **Upload Dataset for Prediction:** Upload a new dataset (CSV or Excel) on which you want to make predictions.
        3. **Select Target Column:** Specify the target column in your new dataset. This is used for evaluation if the target is present.
        4. **Run Prediction:** The app will automatically preprocess the new data (using the same logic as training) and generate predictions.
        5. **Evaluation Metrics:** If your uploaded dataset includes the target column, the model's performance on this new data will be evaluated and displayed.
        6. **Download Results:** Download the dataset with the added 'Prediction' column.
        """)
    st.markdown("---")

    meta_df = load_user_csv(st.session_state.get("username", "anonymous"), "model_metadata", columns=["Model Name","Task","Dataset","Metric","Score","Saved At","File","Duration"])
    if meta_df.empty:
        st.warning("❌ No saved models found. Please train and save a model first.")
        return

    # Step 1: Select Model
    st.markdown("### 📦 Step 1: Select a Saved Model")
    model_file = st.selectbox("Choose a model file", meta_df["File"].tolist(), help="Select a previously trained and saved model for deployment.")
    selected_meta = meta_df[meta_df["File"] == model_file].iloc[0]
    model_path = os.path.join(MODEL_DIR, model_file)
    model = joblib.load(model_path)
    task = selected_meta["Task"]

    with st.expander("📋 Model Metadata"):
        st.json({
            "Model": selected_meta["Model Name"],
            "Task": selected_meta["Task"],
            "Dataset": selected_meta["Dataset"],
            "Score": selected_meta["Score"],
            "Saved At": selected_meta["Saved At"]
        })

    # Step 2: Upload Prediction Dataset
    st.markdown("---")
    st.markdown("### 📁 Step 2: Upload Dataset for Prediction")
    predict_file = st.file_uploader("Upload new dataset (CSV or Excel)", type=["csv", "xlsx"], help="Upload the dataset on which you want the model to make predictions.")
    if not predict_file:
        st.info("Please upload a dataset to continue.")
        return

    try:
        df = pd.read_csv(predict_file) if predict_file.name.endswith("csv") else pd.read_excel(predict_file)
        st.success("✅ Dataset loaded successfully.")
        st.dataframe(df.head(), use_container_width=True, height=400)
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return

    # Step 3: Target Column
    st.markdown("---")
    st.markdown("### 🎯 Step 3: Select Target Column (Optional for Evaluation)")
    target_col_options = ['None'] + df.columns.tolist()
    target_col = st.selectbox("Select the target column (optional, for evaluation)", target_col_options, index=0, help="If your new dataset contains the target column, select it here to evaluate the model's performance on this data.")
    if target_col == 'None':
        target_col = None
    
    # Step 4: Preprocess & Predict
    st.markdown("---")
    st.markdown("### 🧠 Step 4: Preprocess and Predict")
    if st.button("🚀 Run Prediction"):
        with st.spinner("Preprocessing data and generating predictions..."):
            try:
                df_for_prediction = df.copy()
                
                y_actual = None
                if task == "Classification":
                    X_processed, y_actual = preprocess_classification(df_for_prediction, target_col)
                elif task == "Regression":
                    X_processed, y_actual = preprocess_regression(df_for_prediction, target_col)
                else:
                    st.warning("Unsupported model type for prediction.")
                    return

                predictions = model.predict(X_processed)
                result_df = df.copy()
                result_df["Prediction"] = predictions
                st.success("✅ Prediction completed.")
                st.dataframe(result_df, use_container_width=True, height=400)

                # Step 5: Evaluation
                st.markdown("---")
                st.markdown("### 📊 Evaluation Metrics")
                if target_col and y_actual is not None and len(y_actual) == len(predictions):
                    if task == "Classification":
                        acc = accuracy_score(y_actual, predictions)
                        st.metric(label="Accuracy on New Data", value=f"{acc:.4f}", delta=None)
                        st.markdown("**Confusion Matrix**")
                        plot_confusion_mat(y_actual, predictions)
                        st.markdown("**Classification Report**")
                        plot_classification_report(y_actual, predictions)
                    else:
                        r2 = r2_score(y_actual, predictions)
                        rmse = mean_squared_error(y_actual, predictions, squared=False)
                        st.metric(label="R² Score on New Data", value=f"{r2:.4f}", delta=None)
                        st.metric(label="RMSE on New Data", value=f"{rmse:.4f}", delta=None)
                        st.markdown("**Actual vs Predicted**")
                        plot_regression_results(y_actual, predictions)
                else:
                    st.warning("Cannot perform evaluation: Target column not provided or mismatch in data length. Predictions are shown above.")

                # Step 6: Download Result
                st.markdown("---")
                st.markdown("### 📥 Download Results")
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv", help="Download the dataset with the new 'Prediction' column.")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}. Please ensure the uploaded dataset's features match the model's training features.")
    st.container() # End of Smart Prediction Studio container


# -----------------------------
# Main Application
# -----------------------------
def main():
    """Main application function."""
    st.markdown("""
        <div style='text-align:center; margin-bottom:1.5rem;'>
            <h1 style='font-family:Poppins,Inter,sans-serif; font-weight:600; font-size:2.5rem;'>📊 Advanced AutoML App</h1>
            <div style='font-family:Inter,sans-serif; font-weight:400; font-size:1.2rem; color:#333;'>Beginner Friendly & Powerful</div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.title("🔧 App Controls")
    nav_options = [
        "🏁 Upload & Setup",
        "📊 Classification",
        "📈 Regression",
        "🧪 Feature Engineering",
        "🧬 SHAP",
        "📤 Visual Dashboard",
        "👁 Behavioral Impact",
        "🧠 Image Model Trainer",
        "🔍 Smart Prediction Studio",
        "👤 User Dashboard"
    ]
    nav = st.sidebar.radio("Navigate", nav_options, help="Go to different sections of the app.")

    # Theme toggle
    if "theme" not in st.session_state:
        st.session_state["theme"] = "🔥 Warm"  # Set default theme here (now warm)

    st.sidebar.subheader("🎨 Theme")
    st.session_state["theme"] = st.sidebar.radio("", ["🌚 Dark", "🌞 Light", "🔥 Warm"], index=["🌚 Dark", "🌞 Light", "🔥 Warm"].index(st.session_state["theme"]))
    st.markdown(f"<style>{theme_css[st.session_state['theme']]}</style>", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("🧑‍💻 Developed by <b>Rohit Kumar</b><br><span style='font-size:0.9rem;'>v1.0 &copy; 2024</span>", unsafe_allow_html=True)

    # Load data (only for Upload & Setup, Classification, Regression, Feature Engineering)
    data = None
    uploaded_file = None
    use_sample = None
    if nav in ["🏁 Upload & Setup", "📊 Classification", "📈 Regression", "🧪 Feature Engineering"]:
        with card_container():
            st.subheader("Upload Dataset")
            uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], help="Upload your own dataset (CSV or XLSX format).")
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)
                    st.success(f"Loaded your {uploaded_file.name} dataset")
                except Exception as e:
                    st.error(f"Error loading your file: {e}")
                    return
            else:
                st.info("No dataset loaded. Please upload a file to continue.")
            if data is not None:
                st.dataframe(data.head(), height=300, use_container_width=True)

    # Section Routing
    if nav == "🏁 Upload & Setup":
        pass  # Already handled above
    elif nav == "📊 Classification":
        if data is not None:
            with card_container():
                classification_workflow(data, uploaded_file, use_sample)
    elif nav == "📈 Regression":
        if data is not None:
            with card_container():
                regression_workflow(data, uploaded_file, use_sample)
    elif nav == "🧪 Feature Engineering":
        with card_container():
            feature_engineering_section()
    elif nav == "🧬 SHAP":
        with card_container():
            shap_analysis_section()
    elif nav == "📤 Visual Dashboard":
        with card_container():
            visual_insights_dashboard()
    elif nav == "👁 Behavioral Impact":
        with card_container():
            behavioral_impact_analysis()
    elif nav == "🧠 Image Model Trainer":
        with card_container():
            image_model_training_section()
    elif nav == "🔍 Smart Prediction Studio":
        with card_container():
            smart_prediction_studio()
    elif nav == "👤 User Dashboard":
        user_dashboard_section()

    # Footer
    st.markdown("<div class='footer'>🧑‍💻 Rohit Kumar &mdash; Advanced AutoML App &copy; 2024</div>", unsafe_allow_html=True)


# Inject Google Fonts and custom CSS for premium UI
st.markdown(
    '''<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
    html, body, .stApp {
        font-family: 'Inter', 'Poppins', sans-serif !important;
        background: #F9FAFB;
        color: #333333;
    }
    .card-container {
        background: #fff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    .stButton > button {
        border-radius: 0.5rem !important;
        font-weight: 600;
        transition: box-shadow 0.2s, background 0.2s;
        box-shadow: 0 2px 8px rgba(58,134,255,0.08);
    }
    .stButton > button:hover {
        box-shadow: 0 4px 16px rgba(58,134,255,0.15);
        filter: brightness(0.95);
    }
    .stSidebar {
        background: #fff !important;
        border-radius: 0 16px 16px 0;
        box-shadow: 2px 0 8px rgba(0,0,0,0.04);
    }
    .stDataFrame, .stTable {
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    .stExpander {
        border-radius: 12px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    .stMetric {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        padding: 1rem;
    }
    .stTextInput>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        border-radius: 8px !important;
        padding: 0.5rem !important;
        border: 1px solid #E0E0E0 !important;
        background: #fff !important;
        color: #333 !important;
    }
    .stRadio > label, .stCheckbox > label {
        font-weight: 500;
    }
    .stSlider > div {
        padding: 0.5rem 0;
    }
    .footer {
        color: #B0B0B0;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }
    </style>''',
    unsafe_allow_html=True
)

@contextlib.contextmanager
def card_container():
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    yield
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# User Authentication System
# -----------------------------
USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    users = load_users()
    if username in users and users[username] == hash_password(password):
        return True
    return False

def register_user(username, password):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = hash_password(password)
    save_users(users)
    return True, "Registration successful."

def logout():
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""

# Authentication UI
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""

if not st.session_state["logged_in"]:
    st.markdown("""
        <style>
        .centered-card {
            max-width: 400px;
            margin: 8vh auto 0 auto;
            background: #fff;
            border-radius: 18px;
            box-shadow: 0 4px 24px rgba(60,60,60,0.10), 0 1.5px 6px rgba(58,134,255,0.08);
            padding: 2.5rem 2rem 2rem 2rem;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .centered-card h2 {
            font-family: 'Poppins', 'Inter', sans-serif;
            font-weight: 700;
            color: #3A86FF;
            margin-bottom: 1.2rem;
        }
        .centered-card .stTabs, .centered-card .stTextInput, .centered-card .stButton {
            width: 100% !important;
            margin-bottom: 1.1rem;
        }
        .centered-card .stTabs {
            margin-top: 1.2rem;
        }
        .centered-card label {
            font-weight: 500;
        }
        .stApp > header {display: none;}
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="centered-card">', unsafe_allow_html=True)
    st.markdown("<h2>🔒 User Login</h2>", unsafe_allow_html=True)
    # Place everything inside the card
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if authenticate_user(login_user, login_pass):
                st.session_state["logged_in"] = True
                st.session_state["username"] = login_user
                st.success(f"Welcome, {login_user}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with tab2:
        reg_user = st.text_input("New Username", key="reg_user")
        reg_pass = st.text_input("New Password", type="password", key="reg_pass")
        reg_pass2 = st.text_input("Confirm Password", type="password", key="reg_pass2")
        if st.button("Register"):
            if not reg_user or not reg_pass:
                st.warning("Please fill all fields.")
            elif reg_pass != reg_pass2:
                st.warning("Passwords do not match.")
            else:
                ok, msg = register_user(reg_user, reg_pass)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()
else:
    st.sidebar.markdown(f"**Logged in as:** `{st.session_state['username']}`")
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()

def delete_user_model(username, file_name):
    # Remove model file
    file_path = os.path.join(MODEL_DIR, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    # Remove from model_metadata
    meta_path = get_user_data_path(username, "model_metadata")
    if os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        df = df[df["File"] != file_name]
        df.to_csv(meta_path, index=False)
    log_user_activity(username, f"Deleted model {file_name}", "🗑️")

if __name__ == "__main__":
    main()

