# New imports for narrative detection
import os
from typing import Optional, List

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from corpus import load_multi_claim, load_media_content
from entities import NamedEntitiesExtractor
from pipeline import NarrativeDetectionPipeline

app = FastAPI()

# --- Pydantic Models ---
class CorpusRequest(BaseModel):
    corpus: str

class ClaimExtractionRequest(BaseModel):
    text: str
    embedder: Optional[str] = None
    corpus: Optional[str] = None

class EntityExtractionRequest(BaseModel):
    text: str

class PipelineRequest(BaseModel):
    corpus: str
    embedder: Optional[str] = None
    min_macro_cluster_size: int = 50
    min_micro_cluster_size: int = 5
    bertopic_runs: int = 3

# --- In-memory cache for loaded corpus (for demo) ---
corpus_cache = {}
embedding_cache = {"Multilingual E5 Large": "intfloat/multilingual-e5-large",
                   "Paraphrase XLM-R Multilingual v1": "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
                   "Paraphrase Multilingual MPNet Base v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        }

# --- Endpoints ---
@app.get("/corpuses/")
def list_corpora():
    return ["MediaContent-Library", "MultiClaim-v2"]

@app.get("/embedders/")
def list_embedders():
    return ["Multilingual E5 Large", "Paraphrase XLM-R Multilingual v1", "Paraphrase Multilingual MPNet Base v2"]

@app.get("/llms/")
def list_llms():
    return ["gemma3:4b", "gemma3:12b", "llama3:8b"]

@app.post("/corpus/")
def load_corpus(req: CorpusRequest):
    try:
        if req.corpus == "MultiClaim-v2":
            df = load_multi_claim()
        elif req.corpus == "MediaContent-Library":
            df = load_media_content()
        else:
            return {"error": "Unknown corpus"}
        # Cache for session (optional)
        corpus_cache[req.corpus] = df
        
        # Convert DataFrame to dict with primitive types
        preview_records = []
        for record in df.head(5).to_dict(orient="records"):
            clean_record = {}
            for key, value in record.items():
                if key == 'embedding':
                    # Skip embedding as it's a large numpy array
                    continue
                elif key == 'entities':
                    # Ensure entities are strings
                    if isinstance(value, list):
                        clean_record[key] = [str(e) for e in value]
                    else:
                        clean_record[key] = str(value)
                else:
                    # Convert other values to primitive types
                    clean_record[key] = str(value) if not isinstance(value, (int, float, bool, type(None))) else value
            preview_records.append(clean_record)
        
        # Return preview with only primitive types
        return {
            "columns": [str(col) for col in df.columns],
            "preview": preview_records
        }
    except Exception as e:
        # Log the error and return a simple error response
        print(f"Error in load_corpus: {str(e)}")
        return {"error": str(e)}

@app.post("/extract_claims/")
def extract_claims(req: ClaimExtractionRequest):
    try:
        # Extract claims from text
        pipeline = NarrativeDetectionPipeline()
        
        # Split text into sentences (simple approach)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', req.text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Map embedder name to model name
        model_name = None
        if req.embedder and req.embedder in embedding_cache:
            model_name = embedding_cache[req.embedder]
        
        # Embed all sentences
        embeddings, _ = pipeline.embed_texts(sentences, model_name=model_name)
        
        # Extract entities
        extractor = NamedEntitiesExtractor()
        entities = extractor.extract_entities(pd.Series(sentences))
        
        # Find claim positions for highlighting
        claim_positions = []
        current_pos = 0
        for sentence in sentences:
            # Find the position of this sentence in the original text
            start = req.text.find(sentence, current_pos)
            if start >= 0:
                end = start + len(sentence)
                claim_positions.append({
                    'text': sentence,
                    'start': start,
                    'end': end
                })
                current_pos = end
        
        # Safely process entities to ensure they're serializable
        processed_entities = []
        for entity_list in entities:
            # Ensure each entity is a simple string
            processed_entities.extend([str(entity) for entity in entity_list])
        
        # Convert numpy array shape to a list of integers
        embedding_shape = [int(dim) for dim in embeddings.shape]
        
        # Create a simple dictionary with only primitive types
        result = {
            "claims": [str(s) for s in sentences],
            "embedding_shape": embedding_shape,
            "entities": processed_entities,
            "claim_positions": [
                {"text": str(pos["text"]), "start": int(pos["start"]), "end": int(pos["end"])}
                for pos in claim_positions
            ]
        }
        
        return result
    except Exception as e:
        # Log the error and return a simple error response
        print(f"Error in extract_claims: {str(e)}")
        return {"error": str(e)}

@app.post("/add_to_corpus/")
def add_to_corpus(req: ClaimExtractionRequest):
    # Get the corpus name from the request
    corpus = getattr(req, 'corpus', None)
    if not corpus:
        return {"error": "Corpus name not provided"}
    
    # Load or get the corpus
    if corpus not in corpus_cache:
        # Load the corpus if not in cache
        try:
            if corpus == "MultiClaim-v2":
                df = load_multi_claim()
            elif corpus == "MediaContent-Library":
                df = load_media_content()
            else:
                return {"error": f"Unknown corpus: {corpus}"}
            corpus_cache[corpus] = df
        except Exception as e:
            return {"error": f"Failed to load corpus: {str(e)}"}
    
    # Get the current corpus
    df = corpus_cache[corpus]
    
    # Split text into sentences (simple approach)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', req.text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Extract entities for each sentence
    extractor = NamedEntitiesExtractor()
    entities = extractor.extract_entities(pd.Series(sentences))
    
    # Process entities to ensure they're serializable
    processed_entities = []
    for entity_list in entities:
        # Convert each entity list to a list of strings
        processed_entities.append([str(entity) for entity in entity_list])
    
    # Create a DataFrame for the new claims
    new_claims = pd.DataFrame({
        'text': sentences,
        'lang': ['en'] * len(sentences),  # Assuming English, could be detected
        'published': [pd.Timestamp.now().strftime('%d-%m-%Y')] * len(sentences),
        'entities': processed_entities
    })
    
    # Add to corpus
    updated_corpus = pd.concat([df, new_claims], ignore_index=True)
    corpus_cache[corpus] = updated_corpus
    
    # Save the updated corpus to disk
    try:
        # Determine the file path based on corpus name
        if corpus == "MultiClaim-v2":
            file_path = os.path.join('datasets', 'multi-claim.csv')
        elif corpus == "MediaContent-Library":
            file_path = os.path.join('datasets', 'media-content.csv')
        else:
            file_path = os.path.join('datasets', f"{corpus.lower().replace('-', '_')}.csv")
        
        # Save to CSV
        updated_corpus.to_csv(file_path, index=False)
        print(f"Updated corpus saved to {file_path} with {len(updated_corpus)} samples")
    except Exception as e:
        print(f"Error saving updated corpus: {str(e)}")
    
    # Create a simple dictionary with only primitive types
    result = {
        "success": True,
        "message": f"Added {len(sentences)} claims to corpus",
        "corpus_size": int(len(updated_corpus)),
        "added_claims": int(len(sentences))
    }
    
    return result

@app.post("/extract_entities/")
def extract_entities(req: EntityExtractionRequest):
    try:
        extractor = NamedEntitiesExtractor()
        # Use a pandas Series for compatibility
        entities = extractor.extract_entities(pd.Series([req.text]))
        
        # Process entities to ensure they're serializable
        if entities and len(entities) > 0:
            # Convert each entity to a string
            processed_entities = [str(entity) for entity in entities[0]]
            return {"entities": processed_entities}
        else:
            return {"entities": []}
    except Exception as e:
        # Log the error and return a simple error response
        print(f"Error in extract_entities: {str(e)}")
        return {"error": str(e)}

@app.post("/run_pipeline/")
def run_pipeline(req: PipelineRequest):
    # Load corpus
    if req.corpus == "MultiClaim-v2":
        df = load_multi_claim()
    elif req.corpus == "MediaContent-Library":
        df = load_media_content()
    else:
        return {"error": "Unknown corpus"}
    
    # Add custom text if available in corpus_cache
    if req.corpus in corpus_cache:
        df = corpus_cache[req.corpus]
    
    # Ensure entities column is properly formatted
    if 'entities' in df.columns:
        # Check if any entities need to be evaluated
        if df['entities'].iloc[0] is not None and isinstance(df['entities'].iloc[0], str) and '[' in df['entities'].iloc[0]:
            try:
                df['entities'] = df['entities'].apply(eval)
            except Exception as e:
                print(f"Warning: Could not evaluate entities: {str(e)}")
    
    # Create output directory
    import os
    output_dir = f"results/{req.corpus.lower().replace('-', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Map embedder name to model name
    model_name = None
    if req.embedder and req.embedder in embedding_cache:
        model_name = embedding_cache[req.embedder]
    
    # Create pipeline config
    pipeline_config = {
        "data": df,
        "min_cluster_size": req.min_micro_cluster_size,
        "max_iterations": req.bertopic_runs,
        "output_dir": output_dir,
        "generate_descriptions": False,  # Don't generate descriptions yet
        "use_cache": False  # Force regeneration of clusters and visualization
    }
    
    # Run pipeline without descriptions
    pipeline = NarrativeDetectionPipeline()
    print(f"Running pipeline with config: {pipeline_config}")
    result_df = pipeline.run_pipeline(**pipeline_config)
    
    # Return a preview or summary with only primitive types
    try:
        # Convert DataFrame to dict with primitive types
        preview_records = []
        for record in result_df.head(10).to_dict(orient="records"):
            clean_record = {}
            for key, value in record.items():
                if key == 'embedding':
                    # Skip embedding as it's a large numpy array
                    continue
                elif key == 'entities':
                    # Ensure entities are strings
                    if isinstance(value, list):
                        clean_record[key] = [str(e) for e in value]
                    else:
                        clean_record[key] = str(value)
                else:
                    # Convert other values to primitive types
                    clean_record[key] = str(value) if not isinstance(value, (int, float, bool, type(None))) else value
            preview_records.append(clean_record)
        
        return {
            "columns": [str(col) for col in result_df.columns],
            "preview": preview_records,
            "n_macro_clusters": int(len(result_df['macro_cluster'].unique())),
            "n_micro_clusters": int(len(result_df['micro_cluster'].unique())),
            "output_dir": str(output_dir)
        }
    except Exception as e:
        # Log the error and return a simple error response
        print(f"Error in run_pipeline_endpoint: {str(e)}")
        return {"error": str(e)}

@app.post("/generate_descriptions/")
def generate_descriptions(req: CorpusRequest):
    # Load corpus
    if req.corpus == "MultiClaim-v2":
        df = load_multi_claim()
    elif req.corpus == "MediaContent-Library":
        df = load_media_content()
    else:
        return {"error": "Unknown corpus"}
    
    # Add custom text if available in corpus_cache
    if req.corpus in corpus_cache:
        df = corpus_cache[req.corpus]
    
    # Ensure entities column is properly formatted
    if 'entities' in df.columns:
        # Check if any entities need to be evaluated
        if df['entities'].iloc[0] is not None and isinstance(df['entities'].iloc[0], str) and '[' in df['entities'].iloc[0]:
            try:
                df['entities'] = df['entities'].apply(eval)
            except Exception as e:
                print(f"Warning: Could not evaluate entities: {str(e)}")
    
    # Create output directory
    import os
    output_dir = f"results/{req.corpus.lower().replace('-', '_')}"
    
    # Load existing results
    result_path = os.path.join(output_dir, "narrative_analysis.csv")
    if not os.path.exists(result_path):
        return {"error": "No results found. Run the pipeline first."}
    
    result_df = pd.read_csv(result_path)
    
    # Generate descriptions
    pipeline = NarrativeDetectionPipeline()
    print(f"Generating descriptions for corpus: {req.corpus} in directory: {output_dir}")
    
    try:
        # Load the 2D embeddings if they exist
        embedding_path = os.path.join(output_dir, "narrative_map.npy")
        if os.path.exists(embedding_path):
            embedding_2d = np.load(embedding_path)
            print(f"Loaded existing 2D embeddings from {embedding_path}")
        else:
            print(f"No existing 2D embeddings found at {embedding_path}, will generate new ones")
            embedding_2d = None
        
        # Generate descriptions and update visualization
        updated_df = pipeline.generate_cluster_descriptions(result_df, df, output_dir, embedding_2d)
        
        # Verify the HTML file was created
        html_path = os.path.join(output_dir, "narrative_map.html")
        if os.path.exists(html_path):
            print(f"Visualization HTML file created at: {html_path}")
        else:
            print(f"Warning: Visualization HTML file not found at: {html_path}")
            
        return {
            "success": True,
            "message": "Descriptions generated successfully",
            "output_dir": output_dir,
            "html_exists": os.path.exists(html_path)
        }
    except Exception as e:
        print(f"Error generating descriptions: {str(e)}")
        return {"error": f"Failed to generate descriptions: {str(e)}"}


# For running as a thread
def start_backend():
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except OSError as e:
        if "address already in use" in str(e).lower():
            print(f"Warning: Port 8000 is already in use. Backend may already be running.")
        else:
            print(f"Error starting backend: {str(e)}")
