import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import h5py
import time
import platform
import psutil
import re
import string
import contractions

import plotly.express as px
import plotly.graph_objects as go

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

# Keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, InputLayer

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DeepCSAT AI",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# LOAD CSS
# ============================================================

if os.path.exists("styles.css"):
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ============================================================
# LOADING SCREEN
# ============================================================

with st.spinner("Loading AI System..."):
    time.sleep(1)

# ============================================================
# NLTK SETUP
# ============================================================

packages = ["punkt","stopwords","wordnet","averaged_perceptron_tagger"]

for p in packages:
    try:
        nltk.data.find(p)
    except:
        nltk.download(p)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ============================================================
# PATH CONFIG
# ============================================================

DATA_FILE = "data/processed/csat_feature_engineered.csv"
MODEL_PATH = "models"
OUTPUT_PATH = "outputs"

SCALER_FILE = f"{MODEL_PATH}/feature_scaler.pkl"
VECT_FILE = f"{MODEL_PATH}/tfidf_vectorizer.pkl"
BEST_FILE = f"{OUTPUT_PATH}/best_model.json"
METRICS_FILE = os.path.join(OUTPUT_PATH, "final_model_comparison.csv")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()

# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_FILE)

scaler = load_scaler()

# ============================================================
# LOAD TFIDF
# ============================================================

@st.cache_resource
def load_vectorizer():
    return joblib.load(VECT_FILE)

vectorizer = load_vectorizer()

# ============================================================
# LOAD MODEL METRICS
# ============================================================

@st.cache_data
def load_model_metrics():

    if os.path.exists(METRICS_FILE):
        return pd.read_csv(METRICS_FILE)

    return None


metrics_df = load_model_metrics()

# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_model(path):

    with h5py.File(path,"r") as f:

        config=json.loads(f.attrs["model_config"])

        for layer in config["config"]["layers"]:
            if "config" in layer:
                layer["config"].pop("quantization_config",None)

        model=model_from_json(
            json.dumps(config),
            custom_objects={
                "Sequential":Sequential,
                "Dense":Dense,
                "Dropout":Dropout,
                "BatchNormalization":BatchNormalization,
                "InputLayer":InputLayer
            }
        )

        model.load_weights(path)

    return model

# ============================================================
# SIDEBAR : COMPLETE SYSTEM CONTROL PANEL
# ============================================================

st.sidebar.title("DeepCSAT AI - Control Panel ")

st.sidebar.caption(
    "AI powered customer satisfaction intelligence dashboard using deep learning and NLP analysis."
)


st.sidebar.divider()

# ============================================================
# MODEL SELECTION
# ============================================================

st.sidebar.markdown("### Model Selection")

mode = st.sidebar.radio(
    "Select Mode",
    ["Automatic", "Manual"]
)

model_files = {
    "Baseline ANN Default": "baseline_ann_default.h5",
    "BaselineANN Tuned": "baseline_ann_tuned.h5",
    "Deep ANN Default": "deep_ann_default.h5",
    "Deep ANN Tuned": "deep_ann_tuned.h5",
    "Dropout ANN Default": "dropout_ann_default.h5",
    "Dropout ANN Tuned": "dropout_ann_tuned.h5"
}

# ------------------------------------------------------------
# AUTOMATIC MODEL SELECTION
# ------------------------------------------------------------

if mode == "Automatic":

    if os.path.exists(BEST_FILE):

        with open(BEST_FILE) as f:
            data = json.load(f)

        model_name = data["model_name"]
        model_path = data["model_path"]

    else:

        st.sidebar.error("best_model.json missing")
        model_name = None
        model_path = None

# ------------------------------------------------------------
# MANUAL MODEL SELECTION
# ------------------------------------------------------------

else:

    model_name = st.sidebar.selectbox(
        "Choose Model",
        list(model_files.keys())
    )

    model_path = os.path.join(
        MODEL_PATH,
        model_files[model_name]
    )

# ============================================================
# LOAD MODEL
# ============================================================

if model_path and os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = None


# ============================================================
# ACTIVE MODEL METRICS
# ============================================================

st.sidebar.divider()
st.sidebar.markdown("### Active Model")

if model_name:

    st.sidebar.write("**Model Name:**", model_name)

    if metrics_df is not None and model_path:

        filename = os.path.basename(model_path)

        model_map = {
            "baseline_ann_default.h5": ("Baseline ANN", "Default"),
            "baseline_ann_tuned.h5": ("Baseline ANN", "Tuned"),
            "deep_ann_default.h5": ("Deep ANN", "Default"),
            "deep_ann_tuned.h5": ("Deep ANN", "Tuned"),
            "dropout_ann_default.h5": ("Dropout ANN", "Default"),
            "dropout_ann_tuned.h5": ("Dropout ANN", "Tuned"),
        }

        if filename in model_map:

            model_base, version = model_map[filename]

            model_row = metrics_df[
                (metrics_df["Model"] == model_base) &
                (metrics_df["Version"] == version)
            ]

            if not model_row.empty:

                accuracy = float(model_row["Accuracy"].values[0]) * 100
                precision = float(model_row["Precision"].values[0]) * 100
                recall = float(model_row["Recall"].values[0]) * 100
                f1 = float(model_row["F1 Score"].values[0]) * 100

                # Row 1
                col1, col2 = st.sidebar.columns(2)

                with col1:
                    st.metric("Accuracy", f"{accuracy:.1f}%")

                with col2:
                    st.metric("Precision", f"{precision:.1f}%")

                # Row 2
                col3, col4 = st.sidebar.columns(2)

                with col3:
                    st.metric("Recall", f"{recall:.1f}%")

                with col4:
                    st.metric("F1 Score", f"{f1:.1f}%")

            else:
                st.sidebar.warning("Metrics not found for this model")

        else:
            st.sidebar.warning("Model not recognized")

    else:
        st.sidebar.warning("Metrics CSV not loaded")

else:
    st.sidebar.warning("No model selected")

# ============================================================
# LOADING STATUS SECTION
# ============================================================

st.sidebar.divider()
st.sidebar.markdown("### Loading Status")

def status_icon(status):
    return "✅" if status else "❌"


# ------------------------------------------------------------
# CHECK FILE STATUS
# ------------------------------------------------------------

model_status = {}

for m in model_files.values():

    model_path_check = os.path.join(MODEL_PATH, m)

    model_status[m] = os.path.exists(model_path_check)


dataset_loaded = os.path.exists(DATA_FILE)
vectorizer_loaded = os.path.exists(VECT_FILE)
scaler_loaded = os.path.exists(SCALER_FILE)
json_loaded = os.path.exists(BEST_FILE)

runtime_status = {
    "Model Object": model is not None,
    "Dataset Object": df is not None,
    "Vectorizer": vectorizer is not None,
    "Scaler": scaler is not None
}


# ------------------------------------------------------------
# GLOBAL STATUS
# ------------------------------------------------------------

all_models_loaded = all(model_status.values())

all_files_loaded = all([
    dataset_loaded,
    vectorizer_loaded,
    scaler_loaded,
    json_loaded
])

all_runtime_loaded = all(runtime_status.values())

system_ok = all_models_loaded and all_files_loaded and all_runtime_loaded

if system_ok:
    st.sidebar.success("All components loaded successfully")
else:
    st.sidebar.warning("Some components failed to load")


# ------------------------------------------------------------
# MODELS STATUS
# ------------------------------------------------------------

with st.sidebar.expander("Models Loaded", expanded=False):

    for name, status in model_status.items():

        st.write(
            status_icon(status),
            name
        )


# ------------------------------------------------------------
# SYSTEM FILES
# ------------------------------------------------------------

with st.sidebar.expander("System Files", expanded=False):

    st.write(status_icon(dataset_loaded), "csat_feature_engineered.csv")
    st.write(status_icon(vectorizer_loaded), "tfidf_vectorizer.pkl")
    st.write(status_icon(scaler_loaded), "feature_scaler.pkl")
    st.write(status_icon(json_loaded), "best_model.json")


# ------------------------------------------------------------
# RUNTIME OBJECTS
# ------------------------------------------------------------

with st.sidebar.expander("Runtime Objects", expanded=False):

    for name, status in runtime_status.items():

        st.write(
            status_icon(status),
            name
        )


# ============================================================
# SYSTEM HEALTH
# ============================================================

st.sidebar.divider()
st.sidebar.markdown("### System Health")

cpu_usage = psutil.cpu_percent()
ram_usage = psutil.virtual_memory().percent

dataset_rows = df.shape[0]
feature_count = df.shape[1]
tfidf_vocab = len(vectorizer.get_feature_names_out())

# Row 1
col1, col2 = st.sidebar.columns(2)

with col1:
    st.metric("Dataset Rows", dataset_rows)

with col2:
    st.metric("Features", feature_count)

# Row 2
col3, col4 = st.sidebar.columns(2)

with col3:
    st.metric("TFIDF Vocab", tfidf_vocab)

with col4:
    st.metric("CPU Usage", f"{cpu_usage}%")

# Row 3
col5, col6 = st.sidebar.columns(2)

with col5:
    st.metric("RAM Usage", f"{ram_usage}%")


# ============================================================
# RUNTIME INFO
# ============================================================

st.sidebar.divider()
st.sidebar.markdown("### Runtime Info")

st.sidebar.write("OS:", platform.system())
st.sidebar.write("Python:", platform.python_version())

st.sidebar.divider()

st.sidebar.caption("DeepCSAT AI v1.0 • Customer Intelligence Platform")
# ============================================================
# TEXT PROCESSING
# ============================================================

def clean_text(text):

    text=contractions.fix(str(text))
    text=text.lower()

    text=re.sub(r"http\S+|www\S+"," ",text)
    text=re.sub(r"\d+"," ",text)

    text=text.translate(str.maketrans("","",string.punctuation))

    return text


def preprocess_text(text):

    text=clean_text(text)

    tokens=word_tokenize(text)

    tokens=[w for w in tokens if w not in stop_words]

    pos_tags=pos_tag(tokens)

    lemmas=[
        lemmatizer.lemmatize(word)
        for word,tag in pos_tags
    ]

    return " ".join(lemmas)

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="glass-card">',unsafe_allow_html=True)

st.title("DeepCSAT AI — Customer Satisfaction Intelligence Platform")
st.caption("An AI-powered analytics dashboard that predicts customer satisfaction using deep learning and NLP. It analyzes customer interactions, service performance, and feedback to deliver real-time CSAT insights for data-driven support improvement.")



st.markdown('</div>',unsafe_allow_html=True)

# ============================================================
# INPUT FORM
# ============================================================

st.markdown("## Customer Interaction")

c1,c2=st.columns(2)

channels=["Email","Inbound","Outcall"]

categories=[
c.replace("category_","")
for c in df.columns if c.startswith("category_")
]

products=[
c.replace("Product_category_","")
for c in df.columns if c.startswith("Product_category_")
]

shifts=[
c.replace("Agent Shift_","")
for c in df.columns if c.startswith("Agent Shift_")
]

tenures=[
c.replace("Tenure Bucket_","")
for c in df.columns if c.startswith("Tenure Bucket_")
]

with c1:

    channel=st.selectbox("Service Channel",channels)
    category=st.selectbox("Issue Category",categories)
    product=st.selectbox("Product Category",products)

with c2:

    shift=st.selectbox("Agent Shift",shifts)
    tenure=st.selectbox("Agent Tenure",tenures)

item_price=st.number_input("Item Price",0.0,100000.0,1000.0)

response_time=st.slider("Response Time",0,1000,60)

survey_delay=st.slider("Survey Delay",0,30,2)

remark=st.text_area("Customer Remark")

predict=st.button("Predict CSAT")

# ============================================================
# FEATURE BUILDER
# ============================================================

def build_vector():

    features=df.drop("CSAT Score",axis=1).columns

    X=pd.DataFrame(
        np.zeros((1,len(features))),
        columns=features
    )

    if "Item_price" in X.columns:
        X["Item_price"]=item_price

    if "response_time_minutes" in X.columns:
        X["response_time_minutes"]=response_time

    if "survey_delay_days" in X.columns:
        X["survey_delay_days"]=survey_delay

    if channel=="Inbound" and "channel_name_Inbound" in X.columns:
        X["channel_name_Inbound"]=1

    if channel=="Outcall" and "channel_name_Outcall" in X.columns:
        X["channel_name_Outcall"]=1

    col=f"category_{category}"
    if col in X.columns:
        X[col]=1

    col=f"Product_category_{product}"
    if col in X.columns:
        X[col]=1

    col=f"Agent Shift_{shift}"
    if col in X.columns:
        X[col]=1

    col=f"Tenure Bucket_{tenure}"
    if col in X.columns:
        X[col]=1

    if remark.strip()!="":

        processed=preprocess_text(remark)

        tfidf=vectorizer.transform([processed])

        vocab=vectorizer.get_feature_names_out()

        tfidf_array=tfidf.toarray()[0]

        for i,word in enumerate(vocab):

            if word in X.columns:
                X.loc[0,word]=tfidf_array[i]

    X=X.reindex(columns=features,fill_value=0)

    return X


st.markdown(
    """
    <hr style="
        border: none;
        height: 1px;
        background: linear-gradient(90deg,#6366f1,#22c55e);
        margin: 30px 0;
    ">
    """,
    unsafe_allow_html=True
)

# ============================================================
# PREDICTION
# ============================================================

if predict:

    X=build_vector()

    try:
        input_scaled=scaler.transform(X.values)
    except:
        input_scaled=X.values

    pred=model.predict(input_scaled)

    probs=pred[0] if len(pred.shape)>1 else pred

    score=int(np.argmax(probs))+1

    conf=float(np.max(probs)*100)
    
    

# ============================================================
# CSAT GAUGE
# ============================================================

    st.markdown("## CSAT Score")

    gauge=go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text':"CSAT"},
        gauge={
            'axis':{'range':[1,5]},
            'bar':{'color':"#22c55e"}
        }
    ))

    st.plotly_chart(gauge,use_container_width=True)
  
   

# ============================================================
# CONFIDENCE TABLE
# ============================================================
    st.markdown(
    """
    <hr style="
        border: none;
        height: 1px;
        background: linear-gradient(90deg,#6366f1,#22c55e);
        margin: 30px 0;
    ">
    """,
    unsafe_allow_html=True
)
    st.markdown("## Confidence breakdown:")

    stars = ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]

    prob_percent = [round(p * 100, 1) for p in probs]

    confidence_level = []

    for p in prob_percent:
        if p >= 70:
            confidence_level.append("Very High")
        elif p >= 50:
            confidence_level.append("High")
        elif p >= 30:
            confidence_level.append("Medium")
        else:
            confidence_level.append("Low")

    confidence_df = pd.DataFrame({
        "Star Rating": stars,
        "Probability": [f"{p}%" for p in prob_percent],
        "Confidence": confidence_level
    })

    st.dataframe(
        confidence_df,
        use_container_width=True
    )

    # Overall model confidence
    confidence = round(max(prob_percent),1)

    if confidence > 80:
        color = "#22c55e"
        label = "Very High Confidence"
    elif confidence > 60:
        color = "#3b82f6"
        label = "High Confidence"
    elif confidence > 40:
        color = "#f59e0b"
        label = "Moderate Confidence"
    else:
        color = "#ef4444"
        label = "Low Confidence"

    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.05);
            border-left: 5px solid {color};
            padding:16px;
            border-radius:10px;
            margin-top:10px;
            font-size:16px;
        ">
            <b>{label}</b><br>
            Model confidence in this prediction: <b>{confidence}%</b>
        </div>
        <br>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
    """
    <hr style="
        border: none;
        height: 1px;
        background: linear-gradient(90deg,#6366f1,#22c55e);
        margin: 30px 0;
    ">
    """,
    unsafe_allow_html=True
)
# ============================================================
# PROBABILITY CHART
# ============================================================

    prob_df=pd.DataFrame({
        "Score":[1,2,3,4,5],
        "Probability":probs
    })

    fig=px.bar(prob_df,x="Score",y="Probability")

    st.plotly_chart(fig,use_container_width=True)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

    non_zero=X.loc[:,(X!=0).any(axis=0)]

    text_features=non_zero[
    non_zero.columns.intersection(
    vectorizer.get_feature_names_out()
    )
    ]

    if text_features.shape[1]>0:

        top=text_features.T.sort_values(
        0,ascending=False
        ).head(10)

        fig3=px.bar(top,orientation="h")

        st.plotly_chart(fig3,use_container_width=True)
   
    st.markdown(
    """
    <hr style="
        border: none;
        height: 1px;
        background: linear-gradient(90deg,#6366f1,#22c55e);
        margin: 30px 0;
    ">
    """,
    unsafe_allow_html=True
)
    
st.markdown(
    """
    <div style="
        position:fixed;
        bottom:0;
        left:0;
        width:100%;
        text-align:center;
        padding:12px;
        background:rgba(2,6,23,0.95);
        backdrop-filter:blur(6px);
        border-top:1px solid rgba(255,255,255,0.08);
        color:#94a3b8;
        font-size:13px;
        z-index:100;
    ">
        <b style="color:#e2e8f0;">DeepCSAT AI</b> • Customer Satisfaction Intelligence Platform |
        Built with ❤️ using Deep Learning, NLP & Streamlit |
        <span style="color:#64748b;">
        DeepCSAT Analytics • AI-Powered Customer Experience Insights
        </span>
    </div>
    """,
    unsafe_allow_html=True
)