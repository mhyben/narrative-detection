# pipeline.py
"""
Narrative Detection Pipeline

Implements the main pipeline for narrative detection, integrating claim extraction, entity extraction, clustering, and visualization.
See narrative_detection_pipeline_plan.md for detailed design.
"""
import ast
import os
import pickle
from typing import Optional, Any, Dict, List

import networkx as nx
import numpy as np
import requests
import umap
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from httpcore import ReadTimeout
from pandas import DataFrame
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from entities import NamedEntitiesExtractor
from visualization import visualize


class NarrativeDetectionPipeline:
    """
    Main pipeline class for narrative detection.
    Handles user/corpus input, entity extraction, clustering, and visualization.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline and its components. Loads configuration and sets up resource management.
        """
        self.entity_extractor = None  # GLiNER
        self.topic_clusterer = None  # BERTopic
        self.macro_clusterer = None  # Louvain
        self.micro_clusterer = None  # HDBSCAN
        self.llm_client = None  # Ollama API client
        self.save_resources = False

    def run_pipeline(self, data: DataFrame, min_cluster_size=5, max_iterations=3, output_dir="results",
                    use_cache=True, generate_descriptions=True) -> Any:
        """
        Run the complete pipeline with optional caching and description generation.

        Args:
            data: Dataset containing claims, entities and their language
            min_cluster_size: Minimum cluster size for BERTopic and HDBSCAN
            max_iterations: Maximum number of BERTopic iterations for handling outliers
            output_dir: Directory to save outputs
            use_cache: Whether to use cached results if available
            generate_descriptions: Whether to generate cluster descriptions

        Returns:
            Processed DataFrame with hierarchical cluster assignments
        """
        assert 'text' in data.columns, "'text' column not found in DataFrame. Please provide claims as 'text'."
        assert 'lang' in data.columns, "'lang' column not found in DataFrame. Please provide language for each claim as 'lang'."
        
        # Check if entities column exists, if not, create an empty one
        if 'entities' not in data.columns:
            print("Warning: 'entities' column not found. Extracting entities.")
            self.entity_extractor = NamedEntitiesExtractor()
            data['entities'] = self.entity_extractor.extract_entities(data['text'])

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if cached results exist
        cache_file = os.path.join(output_dir, "cached_clusters.pkl")
        if use_cache and os.path.exists(cache_file):
            print("Loading cached clusters...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
                
                # Check if we need to update with new data
                cached_texts = set(cached_results['text'].tolist())
                new_texts = set(data['text'].tolist())
                
                if cached_texts == new_texts:
                    print("Using cached results - no new data detected")
                    
                    # If descriptions are needed but not in cache, generate them
                    if generate_descriptions and 'cluster_summary' not in cached_results.columns:
                        print("Generating descriptions for cached results...")
                        return self.generate_cluster_descriptions(cached_results, data, output_dir)
                    
                    return cached_results
                else:
                    print(f"Found {len(new_texts - cached_texts)} new texts, updating clusters...")
                    # For simplicity, we'll just run the full pipeline again
                    # In a production system, you would implement incremental updates
            except Exception as e:
                print(f"Error loading cache: {str(e)}. Running full pipeline...")

        # 1. Generate embeddings from text
        embeddings, model = self.embed_texts(data['text'].tolist())

        # 2. Optionally perform BERTopic clustering
        if max_iterations == 0:
            print("Skipping BERTopic clustering (max_iterations=0). Assigning all to a single macro cluster.")
            df = data.copy()
            df['macro_cluster'] = 0  # or -1 if you prefer
            df['topic_probability'] = 1.0
            topic_model = None
        else:
            df, topic_model = self.iterative_bertopic_clustering(
                df=data,
                embeddings=embeddings,
                embedding_model=model,
                min_cluster_size=min_cluster_size,
                max_iterations=max_iterations
            )

        # 3. Perform hierarchical sub-clustering
        df = self.hierarchical_subclustering(
            df=df,
            embeddings=embeddings,
            min_cluster_size=min_cluster_size
        )

        # 4. Add embeddings column
        df['embedding'] = list(embeddings)

        # 5. Keep only required columns
        result_columns = ['text', 'published', 'lang', 'embedding', 'macro_cluster', 'micro_cluster']
        result_df = df[result_columns].copy()

        # 6. Create 2D narrative map for visualization
        embedding_2d = self.narrative_map(result_df, model)

        # 7. Save the embedding for later use
        np.save(os.path.join(output_dir, "narrative_map.npy"), embedding_2d)
        result_df.to_csv(os.path.join(output_dir, "narrative_analysis.csv"), index=False)
        
        # Cache the results
        if use_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result_df, f)
                print(f"Cached results saved to {cache_file}")
            except Exception as e:
                print(f"Error caching results: {str(e)}")

        # 8. Generate descriptions if requested
        if generate_descriptions:
            print("Generating cluster descriptions...")
            result_df = self.generate_cluster_descriptions(result_df, data, output_dir, embedding_2d)
        else:
            # Create empty descriptions for visualization
            result_df['cluster_summary'] = ""
            visualize(embedding_2d, result_df, output_dir, open_browser=False)

        print("\n=== IMPROVED CLUSTERING ANALYSIS SUMMARY ===")
        print(f"Total texts: {len(result_df)}")
        print(f"Macro clusters: {len(result_df['macro_cluster'].unique())}")
        print(f"Micro clusters: {len(result_df['micro_cluster'].unique())}")

        print("Pipeline completed successfully!")
        print(f"Results saved to {output_dir}/narrative_analysis.csv")
        print(f"Visualization saved to {output_dir}/narrative_map.html")

        return result_df
        
    def generate_cluster_descriptions(self, result_df, data, output_dir, embedding_2d=None):
        """
        Generate descriptions for clusters and update visualization.
        
        Args:
            result_df: DataFrame with cluster assignments
            data: Original data with entities
            output_dir: Directory to save outputs
            embedding_2d: 2D embeddings for visualization (optional)
            
        Returns:
            Updated DataFrame with cluster descriptions
        """
        # Load embedding_2d if not provided
        if embedding_2d is None:
            try:
                embedding_2d = np.load(os.path.join(output_dir, "narrative_map.npy"))
            except Exception as e:
                print(f"Error loading embeddings: {str(e)}. Creating new embeddings...")
                embedding_2d = self.narrative_map(result_df, SentenceTransformer("sentence-transformers/paraphrase-xlm-r-multilingual-v1"))
        
        # Generate macro cluster summaries
        print("Generating macro cluster summaries...")
        macro_summaries = {}
        for macro_id in tqdm(result_df['macro_cluster'].unique()):
            cluster_texts = result_df[result_df['macro_cluster'] == macro_id]['text'].tolist()
            # Get entities from original data
            cluster_entities = data[data['text'].isin(cluster_texts)]['entities'].tolist()
            summary = self.hybrid_cluster_summary(cluster_texts, cluster_entities)
            macro_summaries[macro_id] = summary
        
        # Generate micro cluster summaries
        print("Generating micro cluster summaries...")
        micro_summaries = {}
        for micro_id in tqdm(result_df['micro_cluster'].unique()):
            if micro_id == -1:
                continue
            cluster_texts = result_df[result_df['micro_cluster'] == micro_id]['text'].tolist()
            # Get entities from original data
            cluster_entities = data[data['text'].isin(cluster_texts)]['entities'].tolist()
            summary = self.hybrid_cluster_summary(cluster_texts, cluster_entities)
            micro_summaries[micro_id] = summary
        
        # Update with descriptions (prefer micro cluster descriptions)
        result_df['cluster_summary'] = result_df['micro_cluster'].map(micro_summaries)
        
        # Save updated results
        result_df.to_csv(os.path.join(output_dir, "narrative_analysis.csv"), index=False)
        
        # Update visualization
        visualize(embedding_2d, result_df, output_dir, open_browser=False)
        print(f"Updated visualization saved to {output_dir}/narrative_map.html")
        
        return result_df

    @staticmethod
    def embed_texts(texts: List[str],
                    model_name: str=None,
                    batch_size = 64) -> Any:
        """
        Compute embeddings for BERTopic or HDBScan.
        """

        print("[1] Generating embeddings...")
        if model_name is None:
            model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        return embeddings, model

    @staticmethod
    def build_similarity_graph(embeddings, k=15):
        """Build a k-NN similarity graph from embeddings."""
        print(f"Building {k}-NN similarity graph...")

        # Compute similarity matrix
        cosine_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()

        # Build k-NN graph
        G = nx.Graph()
        for i in tqdm(range(len(cosine_matrix))):
            top_k_indices = np.argsort(-cosine_matrix[i])[1:k + 1]  # Exclude self
            for j in top_k_indices:
                G.add_edge(i, j, weight=cosine_matrix[i][j])

        return G, cosine_matrix

    def iterative_bertopic_clustering(self, df, embeddings, embedding_model, min_cluster_size=5,
                                      n_neighbors=15, n_components=5, nr_topics="auto", max_iterations=3):
        """Perform iterative BERTopic clustering on text until no outliers remain or max iterations reached."""
        print("=== Iterative BERTopic Clustering ===")

        # Configure UMAP for dimensionality reduction
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=13
        )

        # Configure vectorizer for better multilingual support
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words=None,
            max_features=5000,
            min_df=0.0,
            max_df=0.5
        )

        # Configure representation model for better topic descriptions
        representation_model = KeyBERTInspired()

        # Initialize BERTopic
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            min_topic_size=min_cluster_size,
            nr_topics=nr_topics,
            calculate_probabilities=True,
            verbose=True
        )

        # Make a copy of the dataframe to work with
        working_df = df.copy().reset_index(drop=True)
        working_embeddings = embeddings.copy()

        # Initialize arrays to store results
        topics_result = np.zeros(len(working_df), dtype=int) - 1  # Initialize all as -1 (outliers)
        probs_result = np.zeros(len(working_df))  # Initialize probabilities

        # Keep track of documents that still need processing
        docs_to_process = np.ones(len(working_df), dtype=bool)
        iteration = 0

        while iteration < max_iterations and np.any(docs_to_process):
            print(f"BERTopic iteration {iteration + 1}/{max_iterations}")

            # Get indices and data for documents that still need processing
            current_indices = np.where(docs_to_process)[0]
            current_texts = [working_df.iloc[i]['text'] for i in current_indices]
            current_embeddings = working_embeddings[docs_to_process]

            # Skip if no documents to process
            if len(current_texts) == 0:
                break

            # Fit the model and get topics
            topics, probs = topic_model.fit_transform(
                current_texts,
                embeddings=current_embeddings
            )

            # Process results
            for i, (topic, prob) in enumerate(zip(topics, probs)):
                idx = current_indices[i]

                if topic != -1:  # Non-outlier
                    topics_result[idx] = topic
                    # Use the first probability value if probs is multi-dimensional
                    if isinstance(prob, (list, np.ndarray)):
                        probs_result[idx] = prob[0]
                    else:
                        probs_result[idx] = prob
                    docs_to_process[idx] = False
                elif iteration == max_iterations - 1:  # Last iteration, mark remaining outliers
                    topics_result[idx] = -100  # Special outlier cluster
                    docs_to_process[idx] = False

            iteration += 1

        # Add topic information to dataframe
        df['macro_cluster'] = topics_result
        df['topic_probability'] = probs_result

        # Get topic information
        topic_info = topic_model.get_topic_info()
        print(f"BERTopic found {len(topic_info)} topics")
        print(f"Topic distribution:\n{df['macro_cluster'].value_counts().head(10)}")

        return df, topic_model

    def hierarchical_subclustering(self, df, embeddings, min_cluster_size=5):
        """Perform hierarchical sub-clustering using Louvain + HDBScan on each macro cluster."""
        import community.community_louvain as community_louvain
        import hdbscan

        print("=== Fine-grained Hierarchical Sub-clustering ===")

        # Initialize results
        result_df = df.copy()
        result_df['micro_cluster'] = -1

        # Process each macro cluster
        for macro_id in tqdm(sorted(df['macro_cluster'].unique()), desc="Processing macro clusters"):
            # Skip the outlier cluster if it exists
            if macro_id == -100:
                continue

            # Get cluster data
            cluster_mask = df['macro_cluster'] == macro_id
            cluster_df = df[cluster_mask]
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = df[cluster_mask].index.tolist()

            # Skip small clusters
            if len(cluster_df) < min_cluster_size:
                continue

            try:
                # Build similarity graph
                G, _ = self.build_similarity_graph(cluster_embeddings, k=min(15, len(cluster_embeddings) - 1))

                # Apply Louvain community detection
                partition = community_louvain.best_partition(G, weight='weight')
                louvain_labels = np.array(list(partition.values()))

                # Process each Louvain community
                for louvain_id in np.unique(louvain_labels):
                    # Get community data
                    community_mask = louvain_labels == louvain_id
                    community_embeddings = cluster_embeddings[community_mask]
                    community_indices = [cluster_indices[i] for i, m in enumerate(community_mask) if m]

                    # Skip small communities
                    if len(community_embeddings) < min_cluster_size:
                        continue

                    # Apply HDBScan
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        metric='euclidean',
                        cluster_selection_method='leaf',
                        allow_single_cluster=True
                    )
                    micro_labels = clusterer.fit_predict(community_embeddings)

                    # Assign micro cluster labels
                    for idx, micro_id in zip(community_indices, micro_labels):
                        if micro_id == -1:  # HDBScan outlier
                            result_df.at[idx, 'micro_cluster'] = f"{macro_id}_{louvain_id}_outlier"
                        else:
                            result_df.at[idx, 'micro_cluster'] = f"{macro_id}_{louvain_id}_{micro_id}"

            except Exception as e:
                print(f"Error processing macro cluster {macro_id}: {str(e)}")
                # Assign all to same micro cluster on error
                for idx in cluster_indices:
                    result_df.at[idx, 'micro_cluster'] = f"{macro_id}_0_0"

        return result_df

    def narrative_map(self, df, model):
        """Create a 2D narrative map using UMAP."""
        print("Creating 2D narrative map with UMAP...")

        texts = df['text'].tolist()
        embeddings_np = model.encode(texts, convert_to_tensor=False)

        # Normalize & reduce to 2D
        scaled = StandardScaler().fit_transform(embeddings_np)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding_2d = reducer.fit_transform(scaled)

        return embedding_2d

    def hybrid_cluster_summary(self, texts, entities, model="gemma3:4b"):
        """Generate a hybrid summary using both entity extraction and LLM."""
        if len(texts) == 0:
            return 'Empty cluster'

        # --- Normalize and count entities ---
        all_entities = []
        for entity_list in entities:
            if isinstance(entity_list, str):
                try:
                    entity_list = ast.literal_eval(entity_list)
                except (ValueError, SyntaxError):
                    entity_list = [entity_list]

            if isinstance(entity_list, list):
                all_entities.extend(entity_list)
            else:
                all_entities.append(str(entity_list))

        from collections import Counter
        entity_counts = Counter(all_entities)
        top_entities = [e for e, _ in entity_counts.most_common(5) if e]
        entity_context = ", ".join(top_entities)
        
        # If we have no entities, extract key terms from the texts
        if not entity_context:
            # Simple keyword extraction from texts

            import re
            
            # Combine all texts
            combined_text = " ".join(texts)
            
            # Remove punctuation and convert to lowercase
            words = re.findall(r'\b\w+\b', combined_text.lower())
            
            # Remove common stop words (simplified list)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about',
                         'as', 'of', 'is', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                         'did', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their'}
            
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Count word frequencies
            word_counts = Counter(filtered_words)
            
            # Get top 5 keywords
            top_keywords = [word for word, _ in word_counts.most_common(5)]
            entity_context = ", ".join(top_keywords)

        # --- Truncate texts for prompt ---
        short_texts = [t.strip()[:300] for t in texts[:5]]
        
        # Create a summary based on the first few texts if Ollama is not available
        fallback_summary = f"Cluster about: {entity_context}" if entity_context else "Cluster: " + ", ".join(short_texts[:2])
        
        # Truncate fallback summary to a reasonable length
        if len(fallback_summary) > 50:
            fallback_summary = fallback_summary[:47] + "..."

        prompt = f"""
        Summarize the common narrative in the following texts.
        Focus on these key entities if relevant: {entity_context}
        Label it clearly, in maximum 5 words and without any commentary.

        Texts:
        {' '.join(short_texts)}
        """

        # --- Send to Ollama ---
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60  # Increased timeout
            )
            if response.status_code == 200:
                summary = response.json()['response'].strip()
                # Ensure the summary is not too long
                if len(summary) > 50:
                    summary = summary[:47] + "..."
                return summary
            else:
                print(f"Cluster (HTTP Error: {response.status_code})")
                return fallback_summary
        except ReadTimeout:
            print("Cluster (Timeout: Ollama API took too long to respond)")
            return fallback_summary
        except requests.exceptions.ConnectionError:
            print("Cluster (Connection Error: Could not connect to Ollama API)")
            return fallback_summary
        except Exception as e:
            print(f"Cluster (Error: {str(e)})")
            return fallback_summary

    def summarize_cluster(self, texts, model="gemma3:4b"):
        """Summarize a cluster using Ollama LLM API."""
        if len(texts) == 0:
            return ''

        # Truncate to first 5 texts, each max 300 chars
        short_texts = [t.strip()[:300] for t in texts[:5]]

        prompt = f"""
        Summarize the common narrative in the following texts. Label it clearly, in maximum 5 words and without any commentary.

        Texts:
        {' '.join(short_texts)}
        """

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                print(f"Cluster (HTTP Error: {response.status_code})")
                return 'No description available'
        except requests.exceptions.ReadTimeout:
            print("Cluster (Timeout: Ollama API took too long to respond)")
            return 'Summary timed out'
        except Exception as e:
            print(f"Cluster (Error: {str(e)})")
            return 'No description available'



