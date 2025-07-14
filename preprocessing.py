from typing import Optional, List, LiteralString

from langdetect import detect, LangDetectException
from pandas import DataFrame
from wtpsplit import SaT

from mdeberta import MDeBertaModel


class InputPreprocessor:
    """
    Performs the sentence splitting on the provided text and classify each sentence in terms of check-worthiness.
    """

    def __init__(self, model_name: str):
        """
        Initialize the components.
        """
        self.claim_extractor = None
        self.model_path = model_name

        self.load_model()

    def load_model(self):
        """
        Load check-worthy claim classification model.
        """

        self.claim_extractor = MDeBertaModel(self.model_path)
        if not self.claim_extractor.final_model:
            raise RuntimeError(f"Failed to load MDeBertaModel from {self.model_path}.")

    def process_user_text(self, text: str) -> str | tuple[LiteralString, DataFrame, str | None]:
        """
        Process user-provided text. Return highlighted check-worthy sentences.
        """

        if not text.strip():
            return ""

        # Detect the text language
        language = self.detect_language(text)

        # Split text into sentences
        sentences = self.split_sentences(text, language)
        print(sentences)

        if not sentences:
            return text

        # Extract claims for each sentence
        claims = self.extract_claims(sentences)

        # Create highlighted HTML
        highlighted_html = ""
        for sentence, is_claim in zip(sentences, claims['check-worthy']):
            if is_claim:
                highlighted_html += f'<span style="background-color: #4ade80; color: #1f2937; padding: 2px 4px; border-radius: 3px; margin: 1px;">{sentence}</span>. '
            else:
                highlighted_html += f'<span style="color: #e5e7eb;">{sentence}</span>. '

        return highlighted_html.strip(), claims, language

    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of the input text using langdetect.
        Returns ISO 639-1 language code (e.g., 'en', 'de'), or None if detection fails.
        """
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            # Return None if language cannot be detected
            return None

    def split_sentences(self, text: str, lang_code: str) -> List[str]:
        """
        Split text into sentences using wtpsplit with the detected language.
        Returns a list of sentences.
        """
        try:
            splitter = SaT("sat-3l", style_or_domain="ud", language=lang_code)
            sentences = splitter.split(text, )
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            raise RuntimeError(f"Sentence splitting failed: {e}")

    def extract_claims(self, sentences: List[str]) -> DataFrame:
        """
        Classify each sentence as check-worthy or not using MDeBertaModel.
        Returns a list of dicts with sentence and classification result (True/False for check-worthy).
        """

        assert(len(sentences) > 0), 'No sentences provided.'

        # Load model if not already loaded
        if self.claim_extractor is None:
            self.load_model()

        try:
            # Classify sentences
            classification = self.claim_extractor.inference(sentences)
            print(classification)
            results = DataFrame({'text': sentences, 'check-worthy': classification})
        except Exception as e:
            print(f"Failed to classify sentences: {e}")
            results = DataFrame({'text': sentences})

        return results

    def unload_model(self):
        """
        Unload the model to free resources.
        """
        del self.claim_extractor  # delete the variable entirely
