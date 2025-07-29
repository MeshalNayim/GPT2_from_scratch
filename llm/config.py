# llm/config.py

class GPTConfig:
    def __init__(
        self,
        vocab_size=50257,
        context_length=256,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias
        
    def to_dict(self):
        return self.__dict__


# Predefined config for 124M model (GPT-2 small)
GPT_CONFIG_124M = GPTConfig(
    vocab_size=50257,
    context_length=256,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate=0.1,
    qkv_bias=False
)
