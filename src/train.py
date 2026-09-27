"""Training loop.

Forward pass, cross-entropy loss with padding excluded via ignore_index,
backward pass, optimizer step, periodic validation, and checkpointing.
"""


import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.tokenizer import CharTokenizer
from src.data import ShapeScenesDataset, make_collate_fn
from src.model.vlm import TinyVLM


def get_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA demande mais indisponible, repli sur CPU.")
        return "cpu"
    return requested


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    with torch.set_grad_enabled(train):
        for images, tokens_in, tokens_out in loader:
            images = images.to(device)
            tokens_in = tokens_in.to(device)
            tokens_out = tokens_out.to(device)

            logits = model(images, tokens_in)

            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                tokens_out.reshape(-1),
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n_real_tokens = (tokens_out != criterion.ignore_index).sum().item()
            total_loss += loss.item() * n_real_tokens
            total_tokens += n_real_tokens

    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overfit_one_batch", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    device = get_device(cfg["device"])
    print("Using device:", device)

    tokenizer = CharTokenizer()
    collate = make_collate_fn(tokenizer.pad_id)

    train_ds = ShapeScenesDataset(cfg["processed_dir"], "train", tokenizer, blind=cfg.get("blind", False))
    val_ds = ShapeScenesDataset(cfg["processed_dir"], "val", tokenizer, blind=cfg.get("blind", False))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               collate_fn=collate, num_workers=cfg.get("num_workers", 2))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             collate_fn=collate, num_workers=cfg.get("num_workers", 2))

    model = TinyVLM(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id,
                     d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"]).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.overfit_one_batch:
        one_batch = next(iter(train_loader))
        images, tokens_in, tokens_out = one_batch
        images, tokens_in, tokens_out = images.to(device), tokens_in.to(device), tokens_out.to(device)

        losses = []
        for step in range(cfg.get("overfit_steps", 300)):
            model.train()
            logits = model(images, tokens_in)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tokens_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % 20 == 0:
                print(f"step {step}: loss = {loss.item():.4f}")

        print(f"Loss finale: {losses[-1]:.4f} (doit etre proche de 0)")

        import json
        with open(ckpt_dir / "e0_overfit_losses.json", "w") as f:
            json.dump(losses, f)
        return

    best_val_loss = float("inf")
    for epoch in range(cfg["n_epochs"]):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        print(f"Epoch {epoch+1}/{cfg['n_epochs']}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
