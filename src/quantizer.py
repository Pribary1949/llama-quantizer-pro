import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaQuantizer:
    """
    A professional-grade quantization suite for Llama-based models.
    Supports 4-bit and 8-bit quantization using BitsAndBytes.
    """
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None

    def setup_quantization_config(self, bits: int = 4) -> BitsAndBytesConfig:
        """
        Configure BitsAndBytes for optimal quantization.
        """
        if bits == 4: