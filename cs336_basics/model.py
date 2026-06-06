import torch
import torch.nn as nn
from einops import einsum, rearrange
from collections.abc import Callable, Iterable
from typing import Optional
import math
import numpy as np


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-3*std, b=3*std)
        self.W = nn.Parameter(weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        emb = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(emb, mean=0, std=1, a=-3, b=3)
        self.emb = nn.Parameter(emb)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device=None, dtype=None):
        super().__init__()
        gain = torch.ones(d_model, device=device, dtype=dtype)
        self.gain = nn.Parameter(gain)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (torch.mean(x**2, dim=-1)+self.eps)**0.5
        result = x * self.gain / rearrange(rms, "... -> ... 1")
        return result.to(in_dtype)
    

def SiLU(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = ((8 * d_model // 3) // 64) * 64
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        return self.W2(SiLU(self.W1(x)) * self.W3(x))
        # return self.W2(SiLU(self.W1(x))) # NoGLU


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        angles = [[(i/theta**((2*k-2)/d_k)) for k in range(1, d_k//2+1)] for i in range(max_seq_len)]
        angles = torch.tensor(angles)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        R = torch.stack([sin, cos], dim=-1).to(device)
        self.register_buffer("R", R, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_pair = rearrange(x, "... (a b) -> ... a b", b=2)
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        result = self.R[token_positions]
        y0 = result[..., 1] * x0 - result[..., 0] * x1
        y1 = result[..., 0] * x0 + result[..., 1] * x1
        result = torch.stack([y0, y1], dim=-1)
        return rearrange(result, "... a b -> ... (a b)")
        # return x # NoPE


def softmax(x, i=None):
    if i is None:
        i=-1
    x_exp = torch.exp(x - torch.max(x, dim=i, keepdim=True).values)
    return x_exp / torch.sum(x_exp, dim=i, keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    if mask is None:
        mask = torch.tril(torch.ones(Q.shape[-2], K.shape[-2], dtype=torch.bool, device=Q.device))
    result = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k")
    result /= (Q.shape[-1])**0.5
    result = result.masked_fill(~mask, float('-inf'))
    result = softmax(result, -1)
    result = einsum(result, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")
    return result
    

class MHA(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None):
        super().__init__()
        self.h = num_heads
        # NOTE: this std may not be correct, may need to consider the num_heads
        self.WQ, self.WK, self.WV, self.WO = [Linear(d_model, d_model) for _ in range(4)]
        self.RoPE = RoPE(theta, d_model//num_heads, max_seq_len) if None not in (max_seq_len, theta) else None

    def forward(self, in_features, token_positions=None):
        Q, K, V = self.WQ(in_features), self.WK(in_features), self.WV(in_features)
        Q = rearrange(Q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.h)
        K = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.h)
        V = rearrange(V, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.h)
        if self.RoPE is not None:
            if token_positions is None:
                token_positions = torch.arange(in_features.shape[-2])
            Q, K = self.RoPE(Q, token_positions), self.RoPE(K, token_positions)
        att = scaled_dot_product_attention(Q, K, V)
        att = rearrange(att, "... h seq_len d_v -> ... seq_len (h d_v)")
        return self.WO(att)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=None, theta=None):
        super().__init__()
        self.RMSNorm1 = RMSNorm(d_model)
        self.MHA = MHA(d_model, num_heads, max_seq_len=max_seq_len, theta=theta)
        self.RMSNorm2 = RMSNorm(d_model)
        self.SwiGLU = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.MHA(self.RMSNorm1(x))
        return x + self.SwiGLU(self.RMSNorm2(x))
        # x = self.RMSNorm1(x + self.MHA(x)) # post-norm
        # return self.RMSNorm2(x + self.SwiGLU(x))
        

class Transformer(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, num_heads, d_ff, context_length, theta):
        super().__init__()
        self.Embedding = Embedding(vocab_size, d_model)
        self.TransformerBlocks = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)])
        self.RMSNorm = RMSNorm(d_model)
        self.output_emb = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.TransformerBlocks(x)
        x = self.RMSNorm(x)
        x = self.output_emb(x)
        return x
        

def cross_entropy(logits, targets):
    logits -= torch.max(logits, dim=-1, keepdim=True).values
    log_exp_sum = torch.log(torch.sum(torch.exp(logits), dim=-1))
    result = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    result = log_exp_sum - result
    return torch.mean(result)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


def try_SGD(lr):
    weights = nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())

        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
        print(weights)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                lr_t = lr * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
                p.data -= lr * weight_decay * p.data
                grad = p.grad.data
                m = state.get("m", 0)
                m = beta1 * m + (1 - beta1) * grad
                v = state.get("v", 0)
                v = beta2 * v + (1 - beta2) * grad ** 2
                p.data -= lr_t * m / (v ** 0.5 + eps)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return t / T_w * alpha_max
    elif t > T_c:
        return alpha_min
    else:
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi))


def gradient_clipping(params, M, eps=1e-6):
    params = list(params)
    with torch.no_grad():
        total_norm = torch.sqrt(sum(
            torch.linalg.norm(p.grad) ** 2
            for p in params if p.grad is not None
        ))
        if total_norm >= M:
            scale = M / (total_norm + eps)
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(scale)


def data_loading(data, batch_size, context_length, device):
    # NOTE: should use np.memmap when loading from file, also should specify dtype
    ix = np.random.randint(0, len(data) - context_length, size=(batch_size,))
    x = np.stack([data[i : i + context_length] for i in ix])
    y = np.stack([data[i + 1 : i + context_length + 1] for i in ix])
    
    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)
    return x, y


def save_checkpoint(model, optimizer, iteration, out):
    model_data = model.state_dict()
    optimizer_data = optimizer.state_dict()
    torch.save({"model": model_data, "optimizer": optimizer_data, "iteration": iteration}, out)


def load_checkpoint(src, model, optimizer):
    data = torch.load(src)
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    return data["iteration"]


def compute_param(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    num_params = 0
    # 1. Embedding
    num_params += vocab_size * d_model

    # 2. Transformer Blocks
    # each block takes the following params:
    # d_model * 2 (RMSNorm)
    # d_model * d_model * 4 (Q, K, V, O in attention)
    # d_model * d_ff * 3 (W1, W2, W3 in SwiGLU)
    num_params += num_layers * (d_model * 2 + d_model * d_model * 4 + d_model * d_ff * 3)

    # 3. RMSNorm after Transformer Blocks
    num_params += d_model

    # 4. Output Embedding
    num_params += d_model * vocab_size

    print(f"Total trainable #params: {num_params}")
    print(f"Est. memory (assume fp32): {num_params*4/10**9} GB")
    return num_params


def compute_matrix_multiply_flops(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    # 1. Embedding
    flops_emb = 0

    # 2. Transformer Blocks
    # each block takes the following params:
    # 0 (RMSNorm, only Hadamard)
    # 3 * 2 * context_length * d_model * d_model (Q, K, V computation) + num_heads * (2 * context_length * d_model/num_heads * context_length + 2 * context_length * context_length * d_model/num_heads) + 2 * context_length * d_model * d_model (Attention)
    # 3 * 2 * context_length * d_model *d_ff (W1, W2, W3 in SwiGLU)
    flops_att = 2 * num_layers * (context_length * d_model * context_length + context_length * context_length * d_model)
    flops_linear = 2 * num_layers * (3 * context_length * d_model * d_model + 3 * context_length * d_model * d_ff + context_length * d_model * d_model)

    # 3. RMSNorm after Transformer Blocks
    flops_rmsnorm = 0

    # 4. Output Embedding
    flops_linear += 2 * context_length * d_model * vocab_size

    num_flops = flops_att + flops_linear

    print(f"Total FLOPs: {num_flops}")
    print(f"Attention FLOPs percentage: {flops_att/num_flops}, #params: {flops_att}")
    print(f"Linear PLOPs: {flops_linear/num_flops}, #params: {flops_linear}")
    return num_flops


if __name__ == "__main__":
    # output: 1640452800 params, 6.5618112 GB Memory
    # compute_param(50257, 1024, 48, 1600, 25, 4288)

    # compute_matrix_multiply_flops(50257, 1024, 12, 768, 12, round(8/3*768/64)*64)
    # compute_matrix_multiply_flops(50257, 1024, 24, 1024, 16, round(8/3*1024/64)*64)
    # compute_matrix_multiply_flops(50257, 1024, 36, 1280, 20, round(8/3*1280/64)*64)
    # # output: 1494425600000 total, 1329743462400 transformer, 164682137600 output_emb
    # compute_matrix_multiply_flops(50257, 1024, 48, 1600, 25, 4288)
    # compute_matrix_multiply_flops(50257, 16384, 48, 1600, 25, 4288)
    
    # try_SGD(1000)

    pass