# New imports for narrative detection
from typing import Optional, List

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

class EntityExtractionRequest(BaseModel):
    text: str

class PipelineRequest(BaseModel):
    corpus: str
    embedder: Optional[str] = None
    min_macro_cluster_size: int = 50
    min_micro_cluster_size: int = 5
    # Add more options as needed (e.g., LLM, etc.)

# --- In-memory cache for loaded corpus (for demo) ---
corpus_cache = {}
embedding_cache = {"Multilingual E5 Large": "intfloat/multilingual-e5-large",
                   "Paraphrase XLM-R Multilingual v1": "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
                   "Paraphrase Multilingual MPNet Base v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        }

# --- Endpoints ---
@app.get("/corpuses/")
def list_corpora():
    return ["MultiClaim-v2", "MediaContent-Library"]

@app.get("/embedders/")
def list_embedders():
    return ["Multilingual E5 Large", "Paraphrase XLM-R Multilingual v1", "Paraphrase Multilingual MPNet Base v2"]

@app.get("/llms/")
def list_llms():
    return ["llama3:8b", "gemma3:12b", "gemma3:4b"]

@app.post("/corpus/")
def load_corpus(req: CorpusRequest):
    if req.corpus == "multi-claim":
        df = load_multi_claim()
    elif req.corpus == "media-content":
        df = load_media_content()
    else:
        return {"error": "Unknown corpus"}
    # Cache for session (optional)
    corpus_cache[req.corpus] = df
    # Return preview
    return {"columns": list(df.columns), "preview": df.head(5).to_dict(orient="records")}

@app.post("/extract_claims/")
def extract_claims(req: ClaimExtractionRequest):
    # For demo: just return the text and embedding shape
    pipeline = NarrativeDetectionPipeline()
    texts = [req.text]
    embeddings, _ = pipeline.compute_embeddings(texts, model_name=req.embedder)
    return {"claims": texts, "embedding_shape": list(embeddings.shape)}

@app.post("/extract_entities/")
def extract_entities(req: EntityExtractionRequest):
    extractor = NamedEntitiesExtractor()
    # Use a pandas Series for compatibility
    entities = extractor.extract_entities(pd.Series([req.text]))
    return {"entities": entities[0] if entities else []}

@app.post("/run_pipeline/")
def run_pipeline_endpoint(req: PipelineRequest):
    # Load corpus
    if req.corpus == "MultiClaim-v2":
        df = load_multi_claim()
    elif req.corpus == "MediaContent-Library":
        df = load_media_content()
    else:
        return {"error": "Unknown corpus"}
    
    # Run pipeline
    pipeline = NarrativeDetectionPipeline()
    result_df = pipeline.run_pipeline(
        data=df,
        min_cluster_size=req.min_micro_cluster_size,
        output_dir="results"
    )
    # Return a preview or summary
    return {
        "columns": list(result_df.columns),
        "preview": result_df.head(10).to_dict(orient="records"),
        "n_macro_clusters": len(result_df['macro_cluster'].unique()),
        "n_micro_clusters": len(result_df['micro_cluster'].unique())
    }


# For running as a thread
def start_backend():
    uvicorn.run(app, host="127.0.0.1", port=8000)
