"""Identidad facial: matching contra galeria de embeddings."""
from inference.identity.matcher import FaceGalleryMatcher
from inference.identity.types import GalleryEntry, IdentityMatch

__all__ = ["FaceGalleryMatcher", "GalleryEntry", "IdentityMatch"]
