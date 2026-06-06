from cs336_basics.model import *
from cs336_basics.bpe import *
from cs336_basics.tokenizer import *
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
import torch
import numpy as np 
import os
os.environ['WANDB_API_KEY'] = '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'


def get_model(mcfg: DictConfig):
    if mcfg.name == "transformer":
        model = Transformer(**{k: v for k, v in mcfg.items() if k != "name"})
    else:
        raise NotImplementedError
    return model


def get_optimizer(model, ocfg: DictConfig):
    if ocfg.name == "adamw":
        optimizer = AdamW(model.parameters(), **{k: v for k, v in ocfg.items() if k != "name"})
    else:
        raise NotImplementedError
    return optimizer


@torch.no_grad()
def evaluate(model, valid_data, batch_size, context_length, device):
    model.eval()
    valid_losses = []
    
    # 1. Get starting indices for non-overlapping sequences
    valid_seq_starts = np.arange(0, len(valid_data) - context_length, context_length)
    
    # 2. Iterate through the validation set in batches
    for i in range(0, len(valid_seq_starts), batch_size):
        batch_starts = valid_seq_starts[i : i + batch_size]
        
        # 3. Construct the batch
        x_batch = np.stack([valid_data[start : start + context_length] for start in batch_starts])
        y_batch = np.stack([valid_data[start + 1 : start + context_length + 1] for start in batch_starts])
        
        x_tensor = torch.from_numpy(x_batch).long().to(device)
        y_tensor = torch.from_numpy(y_batch).long().to(device)
        
        # 4. Forward pass and accumulate loss
        logits = model(x_tensor)
        loss = cross_entropy(logits, y_tensor)
        valid_losses.append(loss.item())
        
    # 5. Average the losses over all validation batches
    avg_valid_loss = sum(valid_losses) / len(valid_losses) if valid_losses else 0.0
    
    # Switch model back to training mode before returning
    model.train()
    
    return avg_valid_loss


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    wandb.init(project=cfg.logging.project, name=f"{cfg.model.name}-{cfg.optimizer.name}-SiLU", config=OmegaConf.to_container(cfg))

    train_data = np.memmap(cfg.dataset.train_path, dtype=np.uint16, mode='r')
    valid_data = np.memmap(cfg.dataset.valid_path, dtype=np.uint16, mode='r')
    model = get_model(cfg.model).to(cfg.training.device)
    optimizer = get_optimizer(model, cfg.optimizer)
    eval_interval = cfg.training.eval_interval
    grad_clip = cfg.training.grad_clip
    save_dir = os.path.join(cfg.logging.save_dir, "chechpoints")
    os.makedirs(save_dir, exist_ok=True)

    for step in range(cfg.training.steps):
        # ============ learning rate scheduler ============
        current_lr = cosine_schedule(step, cfg.scheduler.alpha_max, cfg.scheduler.alpha_min, cfg.scheduler.T_w, cfg.scheduler.T_c)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # ============ training ============
        model.train()
        optimizer.zero_grad()
        train_sample = data_loading(train_data, cfg.training.batch_size, context_length=cfg.model.context_length, device=cfg.training.device)
        logits = model(train_sample[0])
        train_loss = cross_entropy(logits, train_sample[1])
        train_loss.backward()
        if grad_clip > 0.0:
            gradient_clipping(model.parameters(), grad_clip)
        optimizer.step()
        wandb.log({"train/loss": train_loss.item()}, step=step)
        print(f"Step {step}: train loss = {train_loss.item()}")

        # ============ evaluating ============
        if step % eval_interval == 0 or step == cfg.training.steps - 1:
            valid_loss = evaluate(
                model=model, 
                valid_data=valid_data, 
                batch_size=cfg.training.batch_size, 
                context_length=cfg.model.context_length, 
                device=cfg.training.device
            )
            wandb.log({"valid/loss": valid_loss}, step=step)
            print(f"Step {step}: valid loss = {valid_loss}")
            save_checkpoint(model, optimizer, step, os.path.join(save_dir, f"model_step_{step}.pt"))
    wandb.finish()


if __name__ == "__main__":
    train()


