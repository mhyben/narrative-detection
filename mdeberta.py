import os

import keras
import keras_hub
import numpy as np

class MDeBertaModel:

    def __init__(self, model_name: str = None):
        self.max_len = 256
        self.final_model = None

        self.preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset(
            'deberta_v3_base_multi',
            sequence_length=self.max_len
        )

        if model_name:
            self.load_model(model_name)

    def load_model(self, model_name) -> bool:
        try:
            model_path = os.path.join('models', model_name + '.keras')

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at: {model_path}")

            print(f"Loading full model from: {model_path}")
            self.final_model = keras.models.load_model(model_path)
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def inference(self, texts: list[str]) -> list[bool]:
        """Returns a list of booleans: True if classified as a claim (class 1), else False."""
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings.")
        if not all(isinstance(t, str) for t in texts):
            raise ValueError("All items in input list must be strings.")

        predictions = self.final_model.predict(texts)
        classifications = np.argmax(predictions, axis=1)

        return (classifications == 1).tolist()