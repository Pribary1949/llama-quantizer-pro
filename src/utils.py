import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def quantize_model(model_id, bits=4):
    """
    Quantizes a pre-trained model to specified bit precision.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

if __name__ == "__main__":
    m, t = quantize_model("meta-llama/Llama-2-7b-hf")
    print("Model quantized successfully.")