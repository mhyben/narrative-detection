import ast
import os
from os.path import join, exists

import pandas as pd
from pandas import Series
from tqdm import tqdm

from entities import NamedEntitiesExtractor


def load_multi_claim() -> pd.DataFrame:
    """Load or preprocess the multi-claim dataset."""

    def parse_date(col: Series):
        def parse_string(val):
            return ast.literal_eval(val) if isinstance(val, str) else val

        # Parse column to a list lists
        fc = col.map(parse_string, na_action="ignore")
        # Extract only the timestamp from each inner list
        dates = []
        for i in tqdm(range(len(fc))):
            if fc[i][0][0]:
                dates.append(int(fc[i][0][0]))
            else:
                dates.append(946684800) # default 1.1. 2020 timestamp

        # Convert the extracted timestamps to a datetime format
        return pd.to_datetime(dates, unit='s', origin='unix').strftime('%d-%m-%Y')

    print('Loading multi-claim dataset: ', end='')
    mc_path = join('datasets', 'multi-claim.csv')

    if exists(mc_path):
        # Check if entities column exists in the CSV
        columns = pd.read_csv(mc_path, nrows=0).columns
        converters = {}
        if 'embedding' in columns:
            converters['embedding'] = eval
        if 'entities' in columns:
            converters['entities'] = eval
        
        data = pd.read_csv(mc_path, converters=converters)
        print('ok')
    else:
        print('\nPreprocessing multi-claim dataset:')
        # Create a dataset subset
        data = pd.read_csv(join('datasets', 'MultiClaim v2', 'fact_checks.csv'))
        data = data[['claim', 'instances', 'claim_detected_language']]
        data['published'] = parse_date(data['instances'])
        data = data.drop('instances', axis=1)
        data = data.dropna(subset=['claim'])
        data.columns = ['text', 'lang', 'published']

        # Extract named entities
        gliner = NamedEntitiesExtractor()
        data['entities'] = gliner.extract_entities(data['text'])

        # Save the preprocessed data
        os.makedirs('datasets', exist_ok=True)
        data.to_csv(mc_path, index=False, header=True)
        print('Saved to multi-claim.csv')

    return data


def load_media_content() -> pd.DataFrame:
    """Load or preprocess the media-content dataset."""

    print('Loading media-content dataset: ', end='')
    mcl_path = join('datasets', 'media-content.csv')

    if exists(mcl_path):
        # Check if entities column exists in the CSV
        columns = pd.read_csv(mcl_path, nrows=0).columns
        converters = {}
        if 'embedding' in columns:
            converters['embedding'] = eval
        if 'entities' in columns:
            converters['entities'] = eval
        
        data = pd.read_csv(mcl_path, converters=converters)
        print('ok')
    else:
        print('\nPreprocessing media-content dataset:')
        # Combine the files into a single dataset
        de = pd.read_csv(join("datasets", 'MCL', 'MCL_sample_4narr.detect_germany.csv'))[['text', 'creation_time']]
        ro = pd.read_csv(join("datasets", 'MCL', 'MCL_sample_4narr.detect_romania.csv'))[['text', 'creation_time']]
        ru = pd.read_csv(join("datasets", 'MCL', 'MCL_sample_4narr.detect_russia.csv'))[['text', 'creation_time']]
        tiktok = pd.read_csv(join("datasets", 'MCL', 'TikTok_sample_dataset.csv'))[['video_description', 'create_time']]

        de.columns = ['text', 'date']
        ro.columns = ['text', 'date']
        ru.columns = ['text', 'date']
        tiktok.columns = ['text', 'date']

        # Add language column
        de['lang'] = 'de'
        ro['lang'] = 'ro'
        ru['lang'] = 'ru'
        tiktok['lang'] = 'en'

        # Convert the publication datetime into just year
        data = pd.concat([de, ro, ru, tiktok]).dropna(subset=['text'])
        data = data.drop_duplicates(subset=['text'])
        data['date'] = pd.to_datetime(data['date'], dayfirst=True, utc=True).dt.strftime('%d-%m-%Y')
        data = data[['text', 'lang', 'date']]

        # Extract named entities
        gliner = NamedEntitiesExtractor()
        data['entities'] = gliner.extract_entities(data['text'])

        # Save the preprocessed data
        os.makedirs('datasets', exist_ok=True)
        data.to_csv(mcl_path, index=False, header=True)
        print('Saved to media-content.csv')

    return data



# load_multi_claim()
# load_media_content()