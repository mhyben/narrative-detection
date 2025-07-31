# Narrative Detection Project

## Overview

This project provides a pipeline and interactive UI for detecting, clustering, and visualizing narratives in large text corpora. It leverages advanced NLP models for claim extraction, entity recognition, and topic clustering, and presents results in an interactive web interface.

## Features
- Claim and entity extraction from text
- Topic and narrative clustering using BERTopic and hierarchical methods
- Interactive visualization of narrative maps
- Modular backend (FastAPI) and frontend (Streamlit)

## Project Structure
- `main.py` – Streamlit UI entry point
- `backend.py` – FastAPI backend for processing and clustering
- `pipeline.py` – Main pipeline logic for narrative detection
- `entities.py`, `mdeberta.py`, `preprocessing.py` – NLP utilities
- `datasets/` – Datasets (not included in repo, see below)
- `models/` – Pretrained models (not included in repo, see below)
- `results/` – Output results and visualizations (not included in repo, see below)

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mhyben/narrative-detection
   cd narrative-detection
   ```

2. **Install dependencies:**
   In order to run the clustering on GPU (preferable), it is necessary to install RAPIDS using Conda environment.
   ```bash
   conda create -n narrative-detection -c rapidsai -c conda-forge -c nvidia      rapids=25.06 python=3.10 'cuda-version>=12.0,<=12.8'
   conda activate narrative-detection
   pip install -r requirements.txt
   ```

3. **Download required data and models:**
   The `datasets/`, `models/`, and `results/` directories are **not included** in this repository due to size and licensing constraints. You can obtain them via a provided Google Drive link.

   - **Google Drive Link:** _[Insert your Google Drive link here]_  
   Download the following folders/files:
     - `datasets/` [link to download](https://drive.google.com/drive/folders/1e9-x7R2JXqFSGdgbuB88VNX_hUT4uryl?usp=sharing)
     - `models/` [link to download](https://drive.google.com/drive/folders/1nUA05XnT2oPTyiOdHV3s3gb7AFLA5ShE?usp=drive_link)
     - `results/` [link to download](https://drive.google.com/drive/folders/17SCXEghT7NLue3sXuA8Z5s-G0lXegHc9?usp=sharing)

   After downloading, your project structure should look like:
   ```
   Narrative detection/
     ├── backend.py
     ├── corpus.py
     ├── datasets/
     ├── entities.py
     ├── main.py
     ├── mdeberta.py
     ├── models/
     ├── pipeline.py
     ├── preprocessing.py
     ├── requirements.txt
     ├── results/
     └── ...
   ```

## Running the Project

1. **Start the application:**
   The easiest way to run the full pipeline and UI is via Streamlit:
   ```bash
   streamlit run main.py
   ```
   This will automatically start the backend server in a separate thread if it is not already running.

2. **Access the UI:**
   Open your browser and go to the local Streamlit address (usually [http://localhost:8501](http://localhost:8501)).

3. **Using the App:**
   - Select a corpus and embedder from the dropdowns.
   - Enter or paste text to analyze.
   - Use the buttons to extract claims or discover narrative maps.
   - Visualizations and results will be generated and displayed interactively.

## Notes
- If you encounter issues with missing files, ensure you have downloaded and placed the `datasets/`, `models/`, and `results/` folders as described above.
- For large-scale processing or custom datasets, see `pipeline.py` for programmatic usage.

## Acknowledge
This work has received funding by the European Union under the Horizon Europe vera.ai project, Grant Agreement number 101070093.

## Contact
For any questions, please contact [martin.hyben@kinit.sk].
