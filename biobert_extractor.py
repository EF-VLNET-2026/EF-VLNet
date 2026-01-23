import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BioBERTFeatureExtractor:
    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-v1.1",
        device: Optional[torch.device] = None,
        max_length: int = 512,
        cache_dir: Optional[str] = "/path/to/your/own/cache_directory"
    ):
        if cache_dir == "/path/to/your/own/cache_directory" or cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), 'biobert_cache')

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length

        logger.info(f"Loading BioBERT model: {model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model cache directory: {cache_dir}")

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True,
            )

            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True,
            ).to(self.device)

            self.model.eval()

            self.feature_dim = self.model.config.hidden_size
            logger.info(f"BioBERT model loaded successfully")
            logger.info(f"Feature dimension: {self.feature_dim}")

        except Exception as e:
            logger.error(f"Failed to load BioBERT model: {e}")
            logger.error(f"Please check path: {cache_dir}")
            logger.error(f"Ensure model files exist at: {cache_dir}/models--dmis-lab--biobert-v1.1/")
            raise

    @torch.no_grad()
    def extract_features(
        self,
        texts: List[str],
        pooling: str = 'mean',
    ) -> torch.Tensor:
        if not texts:
            raise ValueError("Text list cannot be empty")

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if pooling == 'cls':
            features = outputs.last_hidden_state[:, 0, :]

        elif pooling == 'mean':
            token_embeddings = outputs.last_hidden_state
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            features = sum_embeddings / sum_mask

        elif pooling == 'max':
            token_embeddings = outputs.last_hidden_state
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[attention_mask_expanded == 0] = -1e9
            features = torch.max(token_embeddings, dim=1)[0]

        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")

        return features

    def extract_features_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        pooling: str = 'mean',
    ) -> torch.Tensor:
        all_features = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_features = self.extract_features(batch_texts, pooling=pooling)
            all_features.append(batch_features.cpu())

        return torch.cat(all_features, dim=0)

    def preprocess_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = text.strip()
        text = ' '.join(text.split())

        return text

    def get_feature_dim(self) -> int:
        return self.feature_dim


class CachedBioBERTExtractor:
    def __init__(
        self,
        extractor: BioBERTFeatureExtractor,
        cache_file: str = "/path/to/your/own/cache_file.npz",
    ):
        self.extractor = extractor
        self.cache_file = Path(cache_file)

        if str(self.cache_file) == "/path/to/your/own/cache_file.npz":
             self.cache_file = Path("./biobert_features_cache.npz")

        self.cache: Dict[str, np.ndarray] = {}
        self.load_cache()

    def load_cache(self):
        if self.cache_file.exists():
            try:
                data = np.load(self.cache_file, allow_pickle=True)
                self.cache = data['cache'].item()
                logger.info(f"Loaded feature cache: {len(self.cache)} items")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
        else:
            logger.info("Cache file not found, creating new cache")

    def save_cache(self):
        try:
            np.savez(
                self.cache_file,
                cache=self.cache,
            )
            logger.info(f"Saved feature cache: {len(self.cache)} items")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def get_features(
        self,
        text: str,
        pooling: str = 'mean',
    ) -> torch.Tensor:
        cache_key = f"{hash(text)}_{pooling}"

        if cache_key in self.cache:
            features = torch.from_numpy(self.cache[cache_key])
        else:
            features = self.extractor.extract_features([text], pooling=pooling)[0]
            self.cache[cache_key] = features.cpu().numpy()

        return features

    def get_features_batch(
        self,
        texts: List[str],
        pooling: str = 'mean',
    ) -> torch.Tensor:
        features_list = []
        texts_to_extract = []
        extract_indices = []

        for i, text in enumerate(texts):
            cache_key = f"{hash(text)}_{pooling}"
            if cache_key in self.cache:
                features_list.append(torch.from_numpy(self.cache[cache_key]))
            else:
                texts_to_extract.append(text)
                extract_indices.append(i)
                features_list.append(None)

        if texts_to_extract:
            new_features = self.extractor.extract_features_batch(
                texts_to_extract,
                pooling=pooling
            )

            for idx, text, feat in zip(extract_indices, texts_to_extract, new_features):
                features_list[idx] = feat
                cache_key = f"{hash(text)}_{pooling}"
                self.cache[cache_key] = feat.cpu().numpy()

        return torch.stack(features_list)

    def __del__(self):
        self.save_cache()


def test_biobert_extractor():
    print("="*60)
    print("Testing BioBERT Feature Extractor (Using Local Model)")
    print("="*60)

    extractor = BioBERTFeatureExtractor()

    texts = [
        "Normal left ventricular size and systolic function. No significant valvular abnormalities.",
        "Dilated cardiomyopathy with severely reduced left ventricular systolic function.",
    ]

    print(f"\nExtracting features for {len(texts)} texts...")

    features = extractor.extract_features(texts, pooling='mean')

    print(f"\nFeatures shape: {features.shape}")
    print(f"Features range: [{features.min().item():.4f}, {features.max().item():.4f}]")
    print(f"Features mean: {features.mean().item():.4f}")
    print(f"Features std: {features.std().item():.4f}")

    print("\nTesting different pooling methods:")
    for pooling in ['cls', 'mean', 'max']:
        feat = extractor.extract_features([texts[0]], pooling=pooling)
        print(f"  {pooling:4s}: shape={feat.shape}, mean={feat.mean().item():.4f}")

    print("\nTest completed!")


if __name__ == '__main__':
    test_biobert_extractor()