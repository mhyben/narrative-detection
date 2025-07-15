import base64
import os
from typing import Dict, Any

import streamlit as st
import requests
import threading
import time

# Function to highlight claims in text
def highlight_claims(text, positions):
    """Highlight claims in the text using HTML."""
    if not positions:
        return text
    
    # Create a copy of the text
    highlighted = text
    
    # Sort positions by start index (descending to avoid index shifting)
    sorted_pos = sorted(positions, key=lambda x: x['start'], reverse=True)
    
    # Insert HTML tags for highlighting
    for pos in sorted_pos:
        start, end = pos['start'], pos['end']
        highlighted = highlighted[:start] + f'<span style="background-color: yellow; color: black;">{highlighted[start:end]}</span>' + highlighted[end:]
    
    return highlighted

# ---- Start Backend in Separate Thread ----
def run_backend_thread():
    # Import here to avoid circular import
    from backend import start_backend
    
    # Check if backend is already running
    try:
        # Try to connect to the backend
        response = requests.get("http://127.0.0.1:8000/corpuses/", timeout=0.5)
        if response.status_code == 200:
            print("Backend is already running.")
            return
    except requests.exceptions.ConnectionError:
        # Backend is not running, start it
        pass
    except Exception as e:
        print(f"Error checking backend status: {str(e)}")
    
    # Start backend in a separate thread
    thread = threading.Thread(target=start_backend, daemon=True)
    thread.start()
    
    # Wait for backend to start (with timeout)
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.get("http://127.0.0.1:8000/corpuses/", timeout=0.5)
            if response.status_code == 200:
                print("Backend started successfully.")
                break
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                time.sleep(1)  # Wait before retrying
            else:
                print("Warning: Backend may not have started properly.")
        except Exception as e:
            print(f"Error checking backend status: {str(e)}")
            break


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
if 'highlighted_text' not in st.session_state:
    st.session_state.highlighted_text = None
if 'original_text' not in st.session_state:
    st.session_state.original_text = None
if 'editing_mode' not in st.session_state:
    st.session_state.editing_mode = True

# Load configuration
config = st.session_state.config

# Function to toggle editing mode
def toggle_editing_mode():
    st.session_state.editing_mode = not st.session_state.editing_mode


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

# Clear the loading placeholder after backend is loaded
loading_placeholder.empty()

# --- Fetch options from backend ---
try:
    corpuses_resp = requests.get("http://127.0.0.1:8000/corpuses/", timeout=5)
    corpus_options = corpuses_resp.json() if corpuses_resp.status_code == 200 else ["MultiClaim-v2", "MediaContent-Library"]
except Exception as e:
    st.warning(f"Could not connect to backend: {str(e)}. Using default options.")
    corpus_options = ["MultiClaim-v2", "MediaContent-Library"]

# --- Fetch embedder options from backend ---
try:
    embedder_resp = requests.get("http://127.0.0.1:8000/embedders/", timeout=5)
    embedder_options = embedder_resp.json() if embedder_resp.status_code == 200 else ["Multilingual E5 Large", "Paraphrase XLM-R Multilingual v1", "Paraphrase Multilingual MPNet Base v2"]
except Exception as e:
    embedder_options = ["Multilingual E5 Large", "Paraphrase XLM-R Multilingual v1", "Paraphrase Multilingual MPNet Base v2"]

# LLMs (static, for UI only)
try:
    llms_resp = requests.get("http://127.0.0.1:8000/llms/", timeout=5)
    llms_options = llms_resp.json() if llms_resp.status_code == 200 else ["llama3:8b", "gemma3:12b", "gemma3:4b"]
except Exception as e:
    llms_options = ["llama3:8b", "gemma3:12b", "gemma3:4b"]


# --- UI Layout ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Text input area - show either editable text area or highlighted text
    if st.session_state.editing_mode:
        # If we have original text saved, use that, otherwise use default
        initial_value = st.session_state.original_text if st.session_state.original_text else config['example_input']
        user_text = st.text_area("Enter your text here:", value=initial_value, height=150, key="text_input")
        st.session_state.original_text = user_text  # Save the current text
    else:
        # Display highlighted text
        st.markdown("**Text with highlighted claims:**")
        st.markdown('<div class="highlighted-text">', unsafe_allow_html=True)
        st.markdown(st.session_state.highlighted_text, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add edit button
        if st.button("Edit Text", key="edit_button"):
            st.session_state.editing_mode = True
            st.rerun()
        
        # Use the saved original text
        user_text = st.session_state.original_text

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
        discover_clicked = st.button("Discover Narrative Map")
    with option_col1:
        micro_size = st.slider("Minimum Micro-cluster size", 0, 30, 5)
    with option_col2:
        macro_size = st.slider("Minimum Macro-cluster size", 0, 150, 50)

# --- Results and actions ---
if extract_claims_clicked:
    try:
        with st.spinner("Loading corpus..."):
            # Load corpus (for preview, optional)
            corpus_resp = requests.post(
                url="http://127.0.0.1:8000/corpus/",
                json={"corpus": corpus_option},
                timeout=10
            )
            if corpus_resp.status_code == 200:
                corpus_preview = corpus_resp.json()
                st.write("**Corpus Preview:**")
                st.dataframe(corpus_preview.get("preview", []))
            else:
                st.warning(f"Failed to load corpus: {corpus_resp.text}")
        
        with st.spinner("Extracting claims..."):
            # Extract claims
            claims_resp = requests.post(
                url="http://127.0.0.1:8000/extract_claims/",
                json={"text": user_text, "embedder": embedder_option, "corpus": corpus_option},
                timeout=30
            )
            if claims_resp.status_code == 200:
                claims_data = claims_resp.json()
                st.success("Claims extracted!")
                
                # Highlight claims in the text
                if "claim_positions" in claims_data:
                    highlighted_text = highlight_claims(user_text, claims_data["claim_positions"])
                    
                    # Save highlighted text to session state and switch to display mode
                    st.session_state.highlighted_text = highlighted_text
                    st.session_state.editing_mode = False
                    
                    # Force a rerun to update the UI
                    st.rerun()
                
                # Add to corpus
                with st.spinner("Adding to corpus..."):
                    add_resp = requests.post(
                        url="http://127.0.0.1:8000/add_to_corpus/",
                        json={"text": user_text, "corpus": corpus_option, "embedder": embedder_option},
                        timeout=10
                    )
                    
                    if add_resp.status_code == 200:
                        add_data = add_resp.json()
                        st.success(f"Added {add_data.get('added_claims', 1)} claims to corpus. New corpus size: {add_data.get('corpus_size', 'unknown')}")
                    else:
                        st.warning(f"Failed to add to corpus: {add_resp.text}")
                
                # Display extracted entities
                if "entities" in claims_data:
                    st.write("**Extracted Entities:**")
                    st.write(claims_data["entities"])
            else:
                st.error(f"Failed to extract claims: {claims_resp.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to backend. Please make sure the backend is running.")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The operation might be taking too long.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if discover_clicked:
    try:
        with st.spinner("Discovering Narratives..."):
            # First generate without descriptions (faster)
            pipeline_resp = requests.post(
                url="http://127.0.0.1:8000/run_pipeline/",
                json={
                    "corpus": corpus_option,
                    "embedder": embedder_option,
                    "min_macro_cluster_size": macro_size,
                    "min_micro_cluster_size": micro_size
                },
                timeout=300  # Allow up to 5 minutes for clustering
            )
            
            if pipeline_resp.status_code == 200:
                pipeline_data = pipeline_resp.json()
                st.success("Initial visualization generated!")
                
                # Display initial visualization
                output_dir = pipeline_data.get("output_dir", f"results/{corpus_option.lower().replace('-', '_')}")
                html_path = os.path.join(output_dir, 'narrative_map.html')
                
                # Debug information
                st.info(f"Looking for visualization at: {html_path}")
                st.info(f"Current working directory: {os.getcwd()}")
                st.info(f"Directory exists: {os.path.exists(os.path.dirname(html_path))}")
                
                if os.path.exists(os.path.dirname(html_path)):
                    files_in_dir = os.listdir(os.path.dirname(html_path))
                    st.info(f"Files in directory: {files_in_dir}")
                else:
                    st.warning(f"Directory not found: {os.path.dirname(html_path)}")
                
                if os.path.exists(html_path):
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.markdown('<div class="full-width-container">', unsafe_allow_html=True)
                    st.components.v1.html(html_content, height=800, scrolling=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Start generating descriptions
                    st.info("Generating cluster descriptions... This may take a while.")
                    
                    try:
                        desc_resp = requests.post(
                            url="http://127.0.0.1:8000/generate_descriptions/",
                            json={"corpus": corpus_option},
                            timeout=600  # Allow up to 10 minutes for description generation
                        )
                        
                        if desc_resp.status_code == 200:
                            st.success("Cluster descriptions generated!")
                            
                            # Refresh visualization
                            with open(html_path, "r", encoding="utf-8") as f:
                                html_content = f.read()
                            st.markdown('<div class="full-width-container">', unsafe_allow_html=True)
                            st.components.v1.html(html_content, height=800, scrolling=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning(f"Failed to generate descriptions: {desc_resp.text}")
                            st.info("You can still view the visualization without descriptions.")
                    except requests.exceptions.Timeout:
                        st.warning("Description generation timed out. You can still view the visualization without descriptions.")
                    except Exception as e:
                        st.warning(f"Error generating descriptions: {str(e)}. You can still view the visualization without descriptions.")
                else:
                    st.error(f"HTML file not found at: {html_path}")
                    st.info("Generating descriptions to create visualization...")
                    
                    # Try to generate descriptions to create the visualization
                    try:
                        desc_resp = requests.post(
                            url="http://127.0.0.1:8000/generate_descriptions/",
                            json={"corpus": corpus_option},
                            timeout=600  # Allow up to 10 minutes for description generation
                        )
                        
                        if desc_resp.status_code == 200:
                            desc_data = desc_resp.json()
                            if desc_data.get("html_exists", False):
                                st.success("Visualization created successfully!")
                                # Try to load the HTML file again
                                if os.path.exists(html_path):
                                    with open(html_path, "r", encoding="utf-8") as f:
                                        html_content = f.read()
                                    st.markdown('<div class="full-width-container">', unsafe_allow_html=True)
                                    st.components.v1.html(html_content, height=800, scrolling=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.error(f"Still could not find HTML file at: {html_path}")
                            else:
                                st.error("Failed to create visualization HTML file.")
                        else:
                            st.error(f"Failed to generate descriptions: {desc_resp.text}")
                    except Exception as e:
                        st.error(f"Error generating descriptions: {str(e)}")
            else:
                st.error(f"Failed to generate visualization: {pipeline_resp.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to backend. Please make sure the backend is running.")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The clustering operation is taking too long.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# --- Visualization preview (if not generated in this session) ---
if not discover_clicked:
    # Try to find the visualization for the selected corpus
    corpus_dir = corpus_option.lower().replace('-', '_')
    html_path = os.path.join("results", corpus_dir, 'narrative_map.html')
    if os.path.exists(html_path):
        st.subheader("Previously Discovered Narrative Map:")
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.markdown('<div class="full-width-container">', unsafe_allow_html=True)
        st.components.v1.html(html_content, height=800, scrolling=True)
        st.markdown('</div>', unsafe_allow_html=True)
