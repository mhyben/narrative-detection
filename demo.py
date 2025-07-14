import os
import base64
from typing import Dict, Any

import streamlit as st

from corpus import load_multi_claim, load_media_content


def _default_config() -> Dict[str, Any]:
    """
    Return the default configuration for the pipeline.
    """
    return {
        # Resource management
        'save_resources': True,  # If True, load/unload models as needed

        # Entity matching
        'enable_entity_matching': True,  # Enable Wikipedia-based entity normalization

        # Clustering thresholds
        'min_macro_cluster_size': 50,  # Minimum size for macro (topic) clusters
        'min_micro_cluster_size': 5,  # Minimum size for micro (sub) clusters

        # Caching
        'entity_cache_dir': 'cache/entities',  # Directory for cached entity data
        'corpus_cache_dir': 'cache/corpus',  # Directory for cached corpus data
        'wikipedia_cache_dir': 'cache/wiki',  # Directory for Wikipedia lookups
        'llm_cache_dir': 'cache/llm',  # Directory for LLM responses

        # Claim detection model paths
        'cw_mdeberta_model': 'mdb-multicw-updated-2e6-5e',
        'cw_xlm_roberta_model': 'xlm-multicw-updated-2e6-5e',
        'cw_lesa_model': 'lesa-multicw-updated-2e6-5e',

        # Embedding models
        'embed_multilingual-e5': 'intfloat/multilingual-e5-large',
        'embed_xlm_roberta_model': 'sentence-transformers/paraphrase-xlm-r-multilingual-v1',
        'embed_mpnet_v2': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',

        # LLM
        'ollama_model': 'llama3:8b',  # Default Ollama LLM model for topic labeling

        # UI/Visualization
        'highlight_checkworthy': True,  # Highlight check-worthy sentences in UI

        'example_input': "Aboard Air Force One en route to the Nato summit in the Netherlands, Trump shared a personal text message from a somewhat unlikely source. "
                         "It was sent by Nato boss Mark Rutte, who praised the American president for what he had accomplished in using US bombers to attack Iran's nuclear facilities. "
                         "Congratulations and thank you for your decisive action in Iran,  wrote Rutte in a message the president posted to his Truth Social account. "
                         "The warm words, and the president's eagerness to share them to the world, illustrated just how much the diplomatic equation in the Middle East and among US allies has changed for Trump. "
                         "Last week he left the G7 summit in Canada a day early, as conflict raged between Israel and Iran and it appeared increasingly likely the US would join the fight. America's allies were anxious. "
                         "Now, it appears Trump is heading to Europe with the intention of basking in their praise. But the outlook, however, is more complicated than that."
    }


# Initialize session state
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'config' not in st.session_state:
    st.session_state.config = _default_config()
if 'highlighted_text' not in st.session_state:
    st.session_state.highlighted_text = ""

# Load configuration
config = st.session_state.config


# Function to load local image as base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{encoded}"


# Streamlit UI config - use wide layout for full-width capability
st.set_page_config(page_title="Narrative Mapper", layout="wide")

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

# App title
st.title("Narrative Mapper")

# Create columns for centered form
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Input area
    user_text = st.text_area("Enter your text here:", value=config['example_input'], height=150)

    # Show highlighted text if available
    if st.session_state.highlighted_text:
        st.markdown("**Extracted Check-worthy claims:**")
        st.markdown(f'<div class="highlighted-text">{st.session_state.highlighted_text}</div>', unsafe_allow_html=True)

    option_col1, option_col2, option_col3 = st.columns([1, 1, 1])

    # Combobox
    with option_col1:
        corpuses = ["MultiClaim v2", "Media Content Library"]
        corpus_option = st.selectbox("Corpus dataset:", corpuses)
        match corpus_option:
            case 'MultiClaim v2':
                corpus = load_multi_claim()
                # st.success("Corpus loaded!")
            case 'Media Content Library':
                corpus = load_media_content()
                # st.success("Corpus loaded!")

    with option_col2:
        embedders = ["multilinugal-e5", "XLM-RoBERTa", "all-mpnet-base-v2"]
        embedders_option = st.selectbox("Embedder:", embedders)

    with option_col3:
        llms = ["llama3:8b", "gemma3:12b", "gemma3:4b"]
        llms_option = st.selectbox("LLMs:", llms)

    # Button row with two buttons together
    button_col1, button_col2, option_col1, option_col2 = st.columns([1, 1, 1, 1])

    with button_col1:
        extract_claims_clicked = st.button("Extract claims")

    with button_col2:
        generate_clicked = st.button("Generate Visualization")

    with option_col1:
        micro_size = st.slider("Minimum Micro-cluster size", 0, 30, 5)

    with option_col2:
        macro_size = st.slider("Minimum Macro-cluster size", 0, 150, 50)

st.markdown('</div>', unsafe_allow_html=True)

# Place loading message and spinner
loading_placeholder = st.empty()

# Full-width graph section - show on startup
html_path = os.path.join("results", 'media-content', 'narrative_map.html')

# Check if file exists and display it
if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Create full-width container
    st.markdown('<div class="full-width-container">', unsafe_allow_html=True)

    # Display the HTML content with full width
    st.components.v1.html(html_content, height=800, scrolling=True)

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.error(f"HTML file not found at: {html_path}")

# Handle Extract Claims button click
if extract_claims_clicked:
    # Show loading message and spinner
    with loading_placeholder.container():
        st.markdown("**Loading claim detection model...**")
        st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)

    # Import and initialize preprocessor
    try:
        from preprocessing import InputPreprocessor

        st.session_state.preprocessor = InputPreprocessor(model_name=config['cw_mdeberta_model'])
        loading_placeholder.empty()  # Clear loading animation
        st.success("Model loaded successfully!")
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    # Show loading message and spinner
    with loading_placeholder.container():
        st.markdown("**Running classification...**")
        st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)

    # Process the text
    assert user_text, "Invalid user input"
    try:
        highlighted_text, claims, _ = st.session_state.preprocessor.process_user_text(user_text)
        st.session_state.highlighted_text = highlighted_text
        loading_placeholder.empty()  # Clear loading animation
        st.success("Classification successful!")
        st.rerun()  # Refresh to show highlighted text
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")

# Handle Generate Visualization button click
if generate_clicked:
    # Add your visualization generation logic here
    st.info("Generating visualization... (implement your visualization logic here)")