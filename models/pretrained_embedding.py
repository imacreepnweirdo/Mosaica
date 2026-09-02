"""Tiny pretrained-CNN wrapper for inspecting image embeddings.

This is the Phase 2 introduction to PyTorch for this project: instead of trying
to optimize a custom matcher immediately, we ask a pretrained ResNet for a
feature vector and inspect what that vector looks like. The pipeline is still
separate from the matching code; this module is intentionally just a model that
turns a PIL image into a numeric embedding.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


class PretrainedEmbeddingModel:
    """ResNet-18 feature extractor with the classifier head removed.

    ResNet18 produces a 512-dimensional embedding after the final average-pool
    stage. That is a usable feature vector for similarity search, and it is the
    same conceptual idea as the mean-RGB vector in the color matcher: a compact
    representation of "what the image looks like".
    """

    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        weights = ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Convert one PIL image into a 1D NumPy feature vector."""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_batch(self, images):
        """Convert a list of PIL images into an (N, D) NumPy array."""
        tensors = torch.stack([self.transform(img).to(self.device) for img in images])
        with torch.no_grad():
            embeddings = self.model(tensors)
        return embeddings.cpu().numpy().astype(np.float32)
