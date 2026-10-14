"""Model and tokenizer loading."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CausalLM(torch.nn.Module):
    """Thin wrapper exposing the Hugging Face model as ``.base_model`` (the attribute every
    script reads hidden states, logits and the output head from)."""

    def __init__(self, config):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=getattr(torch, config.dtype),
            trust_remote_code=getattr(config, "trust_remote_code", True),
        )
        self.config = self.base_model.config


def load_tokenizer(config):
    """Load the tokenizer with RIGHT padding. Answer positions are located as the last
    non-padding token, which is correct for either side, but right padding is the default all
    results were produced with (Gemma-2 tokenizers otherwise pad on the left)."""
    tok = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=getattr(config, "trust_remote_code", True),
    )
    tok.padding_side = "right"
    return tok
