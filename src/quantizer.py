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
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError("Only 4-bit and 8-bit quantization are supported.")

    def load_model(self, bits: int = 4):
        """
        Load and quantize the model from HuggingFace Hub.
        """
        logger.info(f"Loading model {self.model_id} with {bits}-bit quantization...")
        bnb_config = self.setup_quantization_config(bits)
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            logger.info("Model and tokenizer loaded successfully.")