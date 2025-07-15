import base64
import os
from typing import Dict, Any

import streamlit as st
import requests
import threading
import time

# ---- Start Backend in Separate Thread ----
def run_backend_thread():
    # Import here to avoid circular import
    from backend import start_backend
    thread = threading.Thread(target=start_backend, daemon=True)
    thread.start()
    time.sleep(1)  # Give backend time to start


# ---- Streamlit UI ----
st.set_page_config(page_title="Narrative Mapper", layout="wide")

# UI config
def _default_config() -> Dict[str, Any]:
    """
    Return the default configuration for the pipeline.
    """
    return {
        'example_input': "Aboard Air Force One en route to the Nato summit in the Netherlands, Trump shared a personal text message from a somewhat unlikely source. "
                         "It was sent by Nato boss Mark Rutte, who praised the American president for what he had accomplished in using US bombers to attack Iran's nuclear facilities. "
                         "Congratulations and thank you for your decisive action in Iran,  wrote Rutte in a message the president posted to his Truth Social account. "
                         "The warm words, and the president's eagerness to share them to the world, illustrated just how much the diplomatic equation in the Middle East and among US allies has changed for Trump. "
                         "Last week he left the G7 summit in Canada a day early, as conflict raged between Israel and Iran and it appeared increasingly likely the US would join the fight. America's allies were anxious. "
                         "Now, it appears Trump is heading to Europe with the intention of basking in their praise. But the outlook, however, is more complicated than that."
    }

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = _default_config()

# Load configuration
config = st.session_state.config


# Function to load local image as base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

# Load base64 logo
logo_base64 = get_base64_image("logo.png")  # Adjust path as needed

# Custom CSS for layout control
st.markdown("""
    <style>
    /* Center the header content */
    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }

    /* Full-width container for the graph */
    .full-width-container {
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        padding: 0;
    }

    /* Style for the graph iframe */
    .full-width-container iframe {
        width: 100% !important;
        max-width: 100% !important;
        border: none;
        display: block;
    }

    /* Hide default Streamlit padding for full-width sections */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Center form elements */
    .stTextArea > div > div > textarea {
        background-color: #2d3748;
        color: white;
        border: 1px solid #4a5568;
        border-radius: 4px;
    }

    .stSelectbox > div > div > select {
        background-color: #2d3748;
        color: white;
        border: 1px solid #4a5568;
    }

    /* Button styling */
    .stButton > button {
        background-color: transparent;
        color: #4299e1;
        border: 2px solid #4299e1;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }

    .stButton > button:hover {
        background-color: #4299e1;
        color: white;
    }

    /* Dark theme styling */
    .stApp {
        background-color: #1a202c;
        color: white;
    }

    /* Title styling */
    h1 {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }

    /* Labels styling */
    .stTextArea > label, .stSelectbox > label {
        color: white;
        font-weight: 500;
    }

    /* Loading animation */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #4299e1;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        animation: spin 1s linear infinite;
        margin: 50px auto;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Highlighted text container */
    .highlighted-text {
        background-color: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 4px;
        padding: 12px;
        margin: 10px 0;
        min-height: 100px;
        max-height: 300px;
        overflow-y: auto;
        font-family: monospace;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# Header section - centered
st.markdown('<div class="header-container">', unsafe_allow_html=True)

# Display logo at the top
st.markdown(f"""
    <div style="text-align: center; margin-bottom: 0px;">
        <img src="{logo_base64}" width="100"/>
    </div>
""", unsafe_allow_html=True)

st.title("Narrative Mapper")

# Place loading message and spinner
loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown("**Loading backend...**")
    st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
    # Run backend
    run_backend_thread()

# --- Fetch options from backend ---
corpuses_resp = requests.get("http://127.0.0.1:8000/corpuses/")
corpus_options = corpuses_resp.json() if corpuses_resp.status_code == 200 else ["MultiClaim-v2", "MediaContent-Library"]

# --- Fetch embedder options from backend ---
embedder_resp = requests.get("http://127.0.0.1:8000/embedder_labels/")
embedder_options = embedder_resp.json() if embedder_resp.status_code == 200 else ["Multilingual E5 Large", "Paraphrase XLM-R Multilingual v1", "Paraphrase Multilingual MPNet Base v2"]

# LLMs (static, for UI only)
llms_resp = requests.get("http://127.0.0.1:8000/llms/")
llms_options = llms_resp.json() if llms_resp.status_code == 200 else ["llama3:8b", "gemma3:12b", "gemma3:4b"]


# --- UI Layout ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    user_text = st.text_area("Enter your text here:", value=config['example_input'], height=150)

    option_col1, option_col2, option_col3 = st.columns([1, 1, 1])
    with option_col1:
        corpus_option = st.selectbox("Corpus dataset:", corpus_options)
    with option_col2:
        embedder_option = st.selectbox("Embedder:", embedder_options)
    with option_col3:
        llms_option = st.selectbox("LLMs:", llms_options)

    button_col1, button_col2, option_col1, option_col2 = st.columns([1, 1, 1, 1])
    with button_col1:
        extract_claims_clicked = st.button("Extract claims")
    with button_col2:
        generate_clicked = st.button("Generate Visualization")
    with option_col1:
        micro_size = st.slider("Minimum Micro-cluster size", 0, 30, 5)
    with option_col2:
        macro_size = st.slider("Minimum Macro-cluster size", 0, 150, 50)

# --- Results and actions ---
if extract_claims_clicked:
    # Load corpus (for preview, optional)
    corpus_resp = requests.post(url="http://127.0.0.1:8000/corpus/", json={"corpus": corpus_option})
    if corpus_resp.status_code == 200:
        corpus_preview = corpus_resp.json()
        st.write("**Corpus Preview:**")
        st.dataframe(corpus_preview.get("preview", []))
    else:
        st.error("Failed to load corpus.")

    # Extract claims
    claims_resp = requests.post(
        url="http://127.0.0.1:8000/extract_claims/",
        json={"text": user_text, "embedder": embedder_option}
    )
    if claims_resp.status_code == 200:
        claims_data = claims_resp.json()
        st.success("Claims extracted!")
        st.write(claims_data)
    else:
        st.error("Failed to extract claims.")

    # Extract entities
    entities_resp = requests.post(
        "http://127.0.0.1:8000/extract_entities/",
        json={"text": user_text}
    )
    if entities_resp.status_code == 200:
        entities_data = entities_resp.json()
        st.write("**Extracted Entities:**")
        st.write(entities_data["entities"])
    else:
        st.error("Failed to extract entities.")

if generate_clicked:
    st.info("Generating visualization... (implement your visualization logic here)")

# --- Visualization preview (optional, as in demo.py) ---
html_path = os.path.join("results", 'media-content', 'narrative_map.html')
if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.markdown('<div class="full-width-container">', unsafe_allow_html=True)
    st.components.v1.html(html_content, height=800, scrolling=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.error(f"HTML file not found at: {html_path}")
