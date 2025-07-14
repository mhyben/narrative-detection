from collections import defaultdict
from difflib import SequenceMatcher
from typing import List

import Levenshtein
import wikipedia
from gliner import GLiNER
from pandas import Series, DataFrame
from tqdm import tqdm


class NamedEntitiesExtractor:

    def __init__(self):
        self.gliner = None
        self.gliner_categories = [
            'name',
            'nationality',
            'religious or political group',
            'building',
            'location',
            'company',
            'agency',
            'institution',
            'country',
            'city',
            'state',
            'event'
        ]

        self.load_model()

    def load_model(self):
        """Loading Named Entity Recognition model (GLiNER)"""
        self.gliner = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to('cuda')

    @staticmethod
    def levenshtein_dist(claim1, claim2, threshold=0.2):
        """Check if two claims are similar based on Levenshtein distance ratio."""
        distance = Levenshtein.distance(claim1, claim2)
        max_length = max(len(claim1), len(claim2))
        return distance / max_length <= threshold

    def extract_entities(self, claims: Series) -> list:
        """Extract named entities using GLiNER."""
        entity_lists = []

        if claims.shape[0] == 0:
            return entity_lists

        print('Extracting Named Entities:')
        for text in tqdm(claims):
            extracted_entities = []

            try:
                entities = self.gliner.predict_entities(text, self.gliner_categories)
                entities = [entity['text'] for entity in entities]
                extracted_entities.extend(entities)
            except:
                extracted_entities.extend([])

            unique_list = list(dict.fromkeys(extracted_entities))
            entity_lists.append(unique_list)

        return entity_lists

    def entity_histogram(self, entity_lists: list, lang: str) -> DataFrame:
        """Compute histogram from extracted entity lists."""
        text_lang_pairs = defaultdict(int)

        for extracted_entities in entity_lists:
            for entity in extracted_entities:
                is_unique = True
                for pair in text_lang_pairs:
                    if self.levenshtein_dist(entity, pair[0]) and lang == pair[1]:
                        text_lang_pairs[pair] += 1
                        is_unique = False
                        break
                if is_unique:
                    text_lang_pairs[(entity, lang)] += 1

        entity_lang_freq = [(entity[0], entity[1], freq) for entity, freq in text_lang_pairs.items()]
        hist = DataFrame(entity_lang_freq, columns=['entity', 'language', 'frequency'])
        hist = hist.sort_values('frequency', ascending=False)

        return hist

    @staticmethod
    def match_entities(entities: List[str]) -> List[str]:
        """Match entities with Wikipedia entries and normalize references using Wikipedia titles."""
        normalized = []

        for entity in entities:
            try:
                # Search for the most relevant Wikipedia page
                search_results = wikipedia.search(entity)
                if not search_results:
                    normalized.append(entity)
                    continue

                # Fetch the Wikipedia page title for the top result
                best_match = search_results[0]
                page = wikipedia.page(best_match, auto_suggest=False)

                # Check if the title is sufficiently similar to the original entity
                similarity = SequenceMatcher(None, entity.lower(), page.title.lower()).ratio()
                if similarity > 0.6:
                    normalized.append(page.title)
                else:
                    normalized.append(entity)
            except (wikipedia.DisambiguationError, wikipedia.PageError, Exception):
                normalized.append(entity)

        return normalized

    def unload_model(self):
        del self.gliner