import torch
# import torch.nn as nn
from cs336_basics.model import Transformer, AdamW, load_checkpoint
from cs336_basics.tokenizer import Tokenizer


def softmax_temperature(x, t=1.0):
    x = x / t
    x_exp = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
    return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)


def top_p_sampling(probs, p):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    mask = (cumulative - sorted_probs) >= p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices[sampled_idx]
    return next_token.item()


def decode(model, prompt, top_p, special_tokens=None, max_token=None, temperature=1.0):
    if special_tokens is None:
        special_tokens = [0]
    
    count = 0
    next_token = None
    while (next_token not in special_tokens) and (max_token is None or count < max_token):
        logits = model(prompt)[-1]
        probs = softmax_temperature(logits, temperature)
        next_token = top_p_sampling(probs, top_p)
        # print(next_token)
        count += 1
        prompt = prompt + [next_token]
    return prompt


if __name__ == "__main__":
    print("Start...")
    model = Transformer(10000, 16, 512, 16, 1344, 256, 10000)
    optimizer = AdamW(model.parameters(), 0.001, (0.9, 0.999), 1e-4)
    print("Model and Optimizer Initialized")
    step = load_checkpoint('outputs/chechpoints/model_step_99.pt', model, optimizer)
    print("Model and Optimizer Loaded")
    tokenizer = Tokenizer.from_files('cs336_basics/vocab_ts_10k.json', 'cs336_basics/merges_ts_10k.json', ['<|endoftext|>'])
    prompt = "Once upon a time, "
    print(f"Prompt is: {prompt}")
    prompt = tokenizer.encode(prompt)
    print(f"Encoded to: {prompt}")
    with torch.no_grad():
        output = decode(model, prompt, 0.3)
    print(f"Output: {tokenizer.decode(output)}")