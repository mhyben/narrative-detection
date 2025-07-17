import pandas as pd
import requests
import json
from ast import literal_eval
from tqdm import tqdm
import hashlib


def upload_to_kinit(
    analysis_csv_path="results/media-content/narrative_analysis.csv",
    entities_csv_path="datasets/media-content.csv",
    endpoint="https://dbkf.ontotext.com/elastic/mcl_tiktok_export"
):
    """
    Uploads each row from narrative_analysis.csv to the KInIT endpoint,
    mapping entities from media-content.csv by 'text', using hashed ID and _update route.
    """

    # Read CSVs
    analysis_df = pd.read_csv(analysis_csv_path)
    entities_df = pd.read_csv(entities_csv_path)

    # Build a mapping from text to serialized entities
    entities_map = dict(zip(entities_df['text'], entities_df['entities']))

    for idx, row in tqdm(analysis_df.iterrows(), total=len(analysis_df), desc="Uploading to https://dbkf.ontotext.com/elastic:"):
        text = row['text']
        if not pd.notnull(text):
            continue

        # Generate SHA-256 hash of the text as unique document ID
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        url = f"{endpoint}/_update/{text_hash}"

        # Build document payload
        doc = {}

        # KINIT_entity
        entities_raw = entities_map.get(text)
        if pd.notnull(entities_raw):
            try:
                parsed_entities = literal_eval(entities_raw)
                if isinstance(parsed_entities, list):
                    doc["KINIT_entity"] = [{"id": ent, "name": ent} for ent in parsed_entities]
                else:
                    doc["KINIT_entity"] = [{"id": str(parsed_entities), "name": str(parsed_entities)}]
            except Exception:
                doc["KINIT_entity"] = [{"id": str(entities_raw), "name": str(entities_raw)}]

        # KINiT_cluster_micro
        micro_cluster = row.get('micro_cluster')
        cluster_summary = row.get('cluster_summary')
        if pd.notnull(micro_cluster):
            micro_obj = {"id": str(micro_cluster)}
            if pd.notnull(cluster_summary):
                micro_obj["description"] = str(cluster_summary)
            doc["KINiT_cluster_micro"] = micro_obj

        # KINIT_cluster_macro
        macro_cluster = row.get('macro_cluster')
        if pd.notnull(macro_cluster):
            doc["KINIT_cluster_macro"] = str(macro_cluster)

        # Final payload with upsert logic
        payload = {
            "doc": doc,
            "doc_as_upsert": True
        }

        # Send POST request to _update/{id}
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=10
            )
            if response.status_code not in (200, 201):
                print(f"Failed to upload row {idx}: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Exception uploading row {idx}: {e}")


# Run the upload
upload_to_kinit()
