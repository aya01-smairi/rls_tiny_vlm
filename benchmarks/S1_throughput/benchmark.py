"""
benchmarks/S1_throughput/benchmark.py

Mesure le throughput d'entrainement (images/seconde) pour plusieurs
batch sizes, sur CPU et GPU. Gere les 3 pieges cites par le brief :
  1. Warm-up : les premieres iterations sont plus lentes -> on les ignore.
  2. Execution asynchrone du GPU : torch.cuda.synchronize() AVANT de
     demarrer le chrono ET avant de le lire.
  3. Chargement des donnees : ici un batch FACTICE deja en memoire, donc
     le chargement disque n'est PAS inclus dans cette mesure -- choix
     explicite, mentionne dans le rapport.

Usage :
  python -m benchmarks.S1_throughput.benchmark --config configs/baseline.yaml
"""

import argparse
import csv
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.tokenizer import CharTokenizer
from src.model.vlm import TinyVLM
from src.train import load_config


def make_dummy_batch(batch_size: int, seq_len: int, vocab_size: int, device: str):
    images = torch.randn(batch_size, 3, 64, 64, device=device)
    tokens_in = torch.randint(0, vocab_size - 2, (batch_size, seq_len), device=device)
    tokens_out = tokens_in.clone()
    return images, tokens_in, tokens_out


def measure_throughput(model, criterion, optimizer, batch_size, seq_len,
                        vocab_size, device, n_warmup=5, n_measure=20):
    images, tokens_in, tokens_out = make_dummy_batch(batch_size, seq_len, vocab_size, device)

    def one_step():
        optimizer.zero_grad()
        logits = model(images, tokens_in)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tokens_out.reshape(-1))
        loss.backward()
        optimizer.step()

    for _ in range(n_warmup):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(n_measure):
        one_step()

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    images_per_sec = (batch_size * n_measure) / elapsed
    return images_per_sec, elapsed


def get_hardware_info(device: str) -> str:
    lines = [f"Platform: {platform.platform()}", f"Processor: {platform.processor()}"]
    if device == "cuda" and torch.cuda.is_available():
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
        lines.append(f"CUDA version: {torch.version.cuda}")
    else:
        lines.append("GPU: none used for this run")
    lines.append(f"PyTorch version: {torch.__version__}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--seq_len", type=int, default=45)
    parser.add_argument("--n_repeats", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = CharTokenizer()

    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    devices_to_test = ["cpu"]
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
    else:
        print("CUDA indisponible : benchmark CPU seulement.")

    rows = []

    for device in devices_to_test:
        print(f"\n=== Device: {device} ===")

        model = TinyVLM(
            vocab_size=tokenizer.vocab_size,
            d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

        for batch_size in args.batch_sizes:
            n_measure = 20 if device == "cuda" else 5

            speeds = []
            for repeat in range(args.n_repeats):
                try:
                    imgs_per_sec, elapsed = measure_throughput(
                        model, criterion, optimizer, batch_size, args.seq_len,
                        tokenizer.vocab_size, device, n_warmup=3, n_measure=n_measure,
                    )
                    speeds.append(imgs_per_sec)
                    rows.append({"device": device, "batch_size": batch_size,
                                 "repeat": repeat, "images_per_sec": imgs_per_sec})
                    print(f"  batch={batch_size:>4} repeat={repeat}: {imgs_per_sec:.1f} images/s")
                except RuntimeError as e:
                    print(f"  batch={batch_size:>4} repeat={repeat}: ECHEC ({e})")

            if speeds:
                mean = sum(speeds) / len(speeds)
                std = (sum((s - mean) ** 2 for s in speeds) / len(speeds)) ** 0.5
                print(f"  -> moyenne: {mean:.1f} +/- {std:.1f} images/s")

    csv_path = out_dir / "raw_timings.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["device", "batch_size", "repeat", "images_per_sec"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDonnees brutes sauvegardees dans {csv_path}")

    hw_lines = []
    for device in devices_to_test:
        hw_lines.append(get_hardware_info(device))
        hw_lines.append("")
    (out_dir / "hardware.txt").write_text("\n".join(hw_lines))
    print(f"Info hardware sauvegardee dans {out_dir / 'hardware.txt'}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for device in devices_to_test:
            device_rows = [r for r in rows if r["device"] == device]
            batch_sizes_sorted = sorted(set(r["batch_size"] for r in device_rows))
            means = []
            for bs in batch_sizes_sorted:
                vals = [r["images_per_sec"] for r in device_rows if r["batch_size"] == bs]
                means.append(sum(vals) / len(vals))
            ax.plot(batch_sizes_sorted, means, marker="o", label=device)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Images / seconde")
        ax.set_xscale("log", base=2)
        ax.set_title("S1 - Throughput d'entrainement")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = out_dir / "throughput_plot.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot sauvegarde dans {plot_path}")
    except ImportError:
        print("matplotlib non installe : plot ignore.")


if __name__ == "__main__":
    main()