"""
experiments/E1_blind/run_e1.py

Experience E1 : le modele utilise-t-il vraiment l'image ?
Compare un modele entraine normalement a un modele entraine avec
images=zero (configs/blind.yaml), sur le split test.
"""

import argparse
import json
from pathlib import Path

import torch

from src.tokenizer import CharTokenizer
from src.data import ShapeScenesDataset
from src.model.vlm import TinyVLM
from src.generate import generate_word
from src.evaluate import evaluate_predictions
from src.train import load_config, get_device


def load_model_from_checkpoint(cfg: dict, tokenizer: CharTokenizer, device: str) -> TinyVLM:
    model = TinyVLM(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
    ).to(device)

    ckpt_path = Path(cfg["checkpoint_dir"]) / "best.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def run_generation_on_split(model, tokenizer, dataset, device, max_examples=None):
    predictions, targets = [], []
    n = len(dataset) if max_examples is None else min(max_examples, len(dataset))

    for idx in range(n):
        image, token_ids = dataset[idx]
        pred_word = generate_word(model, tokenizer, image, device=device)
        target_word = tokenizer.decode(token_ids)
        predictions.append(pred_word)
        targets.append(target_word)
        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{n} exemples generes")

    return predictions, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--blind_config", type=str, default="configs/blind.yaml")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    baseline_cfg = load_config(args.baseline_config)
    blind_cfg = load_config(args.blind_config)

    device = get_device(baseline_cfg["device"])
    print("Using device:", device)

    tokenizer = CharTokenizer()

    print("Chargement du modele baseline...")
    baseline_model = load_model_from_checkpoint(baseline_cfg, tokenizer, device)
    baseline_test_ds = ShapeScenesDataset(baseline_cfg["processed_dir"], "test", tokenizer, blind=False)
    print("Generation (baseline)...")
    base_preds, base_targets = run_generation_on_split(
        baseline_model, tokenizer, baseline_test_ds, device, args.max_examples
    )
    base_metrics = evaluate_predictions(base_preds, base_targets)

    print("Chargement du modele blind...")
    blind_model = load_model_from_checkpoint(blind_cfg, tokenizer, device)
    blind_test_ds = ShapeScenesDataset(blind_cfg["processed_dir"], "test", tokenizer, blind=True)
    print("Generation (blind)...")
    blind_preds, blind_targets = run_generation_on_split(
        blind_model, tokenizer, blind_test_ds, device, args.max_examples
    )
    blind_metrics = evaluate_predictions(blind_preds, blind_targets)

    print("\n=== Resultats E1 ===")
    print(f"{'metric':<20}{'baseline':>12}{'blind':>12}{'ecart':>12}")
    comparison = {}
    for key in base_metrics:
        b = base_metrics[key]
        bl = blind_metrics.get(key, 0.0)
        comparison[key] = {"baseline": b, "blind": bl, "gap": b - bl}
        print(f"{key:<20}{b:>12.3f}{bl:>12.3f}{b - bl:>12.3f}")

    out_dir = Path(__file__).parent
    with open(out_dir / "results.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nResultats sauvegardes dans {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()