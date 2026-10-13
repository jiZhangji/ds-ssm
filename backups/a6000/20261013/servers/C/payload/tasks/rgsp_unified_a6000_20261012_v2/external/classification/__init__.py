"""Scratch classification adapters; importing does not initialize PyTorch."""

from .factory import ModelNotReadyError, available_methods, create_model

__all__ = ["ModelNotReadyError", "available_methods", "create_model"]
