import torch
import torch.nn as nn
import tiktoken
import pyttsx3

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architecture classes
class MultiHeadAttention(nn.Module):
    def __init__(self,input_dim,output_dim,context_length,dropout_prob,num_heads,use_qkv_bias=False):
        super().__init__()
        assert (output_dim%num_heads==0),"output_dim must be divisible by num_heads"

        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.query_proj = nn.Linear(input_dim,output_dim,bias=use_qkv_bias)
        self.key_proj = nn.Linear(input_dim,output_dim,bias=use_qkv_bias)
        self.value_proj = nn.Linear(input_dim,output_dim,bias=use_qkv_bias)
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer(
            "causal_mask",torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,inputs):
        batch_size,num_tokens,input_dim=inputs.shape

        keys = self.key_proj(inputs)
        queries = self.query_proj(inputs)
        values = self.value_proj(inputs)

        keys = keys.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1,2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores= queries @ keys.transpose(2,3)
        mask = self.causal_mask.bool()[:num_tokens,:num_tokens]
        attn_scores.masked_fill_(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (self.head_dim**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ values).transpose(1,2)
        attn_output = attn_output.contiguous().view(batch_size,num_tokens,self.output_dim)
        attn_output = self.output_proj(attn_output)

        return attn_output

class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True,unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        return self.scale* norm_x +self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"]),
        )

    def forward(self,x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            input_dim=cfg["emb_dim"],
            output_dim=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout_prob=cfg["drop_rate"],
            use_qkv_bias=cfg["qkv_bias"])

        self.ff=FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self,x):
        shortcut = x
        x =self.norm1(x)
        x =self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x =self.norm2(x)
        x =self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# Helper functions
def format_input(entry):
    # Format to match the training data style
    if 'output' in entry:
        # Use the casual style starter if provided
        return f"{entry['output']} {entry['input']}"
    # Fallback to simple format if no output template
    return entry['input']

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# Configuration
GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-medium (355M)"
NEW_CONFIG = GPT_CONFIG_355M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"qkv_bias": True})

# Tokenizer setup
tokenizer = tiktoken.get_encoding("gpt2")

def main():    
    # Load model
    model = GPTModel(NEW_CONFIG)
    model.load_state_dict(torch.load('trained_model.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Initialize text-to-speech engine
    engine = pyttsx3.init()
    # Set properties for fast-paced chess commentary
    engine.setProperty('rate', 200)  # Increased speed for real-time commentary
    engine.setProperty('volume', 1.0)  # Full volume for clarity
    
    # Try to get a more natural voice if available
    voices = engine.getProperty('voices')
    if voices:
        for voice in voices:
            if 'david' in voice.name.lower() or 'mark' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break

    print("\nType your input for chess commentary (or press Enter to exit):")
    while True:
        user_input = input("Input: ").strip()
        if not user_input:
            break
            
        entry = {"instruction": "Provide real-time chess commentary. Be concise and casual. Mention checks, mistakes, and who is winning based on checkmate.", "input": user_input, "output": "Okay, here we go!"}
        input_text = format_input(entry)
        input_ids = text_to_token_ids(input_text, tokenizer).to(device)
        
        with torch.no_grad():
            token_ids = generate(
                model=model,
                idx=input_ids,
                max_new_tokens=60,  # Adjusted for more concise responses
                context_size=NEW_CONFIG["context_length"],
                top_k=40,  # Fine-tuned for better vocabulary variety
                temperature=1.2,  # Adjusted for more natural commentary
                eos_id=50256
            )
            
        generated_text = token_ids_to_text(token_ids, tokenizer)
        # Extract only the model's response, not the input or instruction
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        print("\n--- MODEL COMMENTARY ---\n" + response_text + "\n")
        
        # Only speak the model's commentary response
        if response_text:
            engine.say(response_text)
            engine.runAndWait()

if __name__ == "__main__":
    main()
