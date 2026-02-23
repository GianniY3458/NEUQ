# storage.py
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Gallery:
    features: np.ndarray          # (N, D) float32
    paths: List[str]              # len N


class GalleryStorage:
    def __init__(self, artifact_dir: str = "artifacts"):
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

        self.feat_path = os.path.join(self.artifact_dir, "gallery_features.npy")
        self.paths_path = os.path.join(self.artifact_dir, "gallery_paths.pkl")

    def exists(self) -> bool:
        return os.path.exists(self.feat_path) and os.path.exists(self.paths_path)

    def save(self, features: np.ndarray, paths: List[str]) -> None:
        features = np.asarray(features, dtype=np.float32)
        np.save(self.feat_path, features)
        with open(self.paths_path, "wb") as f:
            pickle.dump(paths, f)

    def load(self) -> Gallery:
        if not self.exists():
            raise FileNotFoundError("Gallery artifacts not found. Please build index first.")
        features = np.load(self.feat_path).astype(np.float32)
        with open(self.paths_path, "rb") as f:
            paths = pickle.load(f)
        if len(paths) != features.shape[0]:
            raise RuntimeError("Gallery mismatch: paths length != features rows.")
        return Gallery(features=features, paths=paths)