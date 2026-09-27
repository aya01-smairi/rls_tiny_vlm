"""Greedy decoding.

Generates one letter at a time from the visual tokens until <eos> or
45 letters, for inference and for evaluation.
"""


import torch

from src.tokenizer import CharTokenizer
from src.model.vlm import TinyVLM


@torch.no_grad()
def generate_word(model: TinyVLM, tokenizer: CharTokenizer, image: torch.Tensor,
                   max_letters: int = 45, device: str = "cpu") -> str:
    """image : (3, 64, 64), deja normalisee [0,1]."""
    model.eval()
    image = image.unsqueeze(0).to(device)  # (1, 3, 64, 64)

    # `generated` = UNIQUEMENT les vraies lettres deja produites.
    # A chaque etape, on ajoute un placeholder EN DERNIERE position (jamais
    # en premiere !) car c'est la derniere sortie de la sequence qu'on lit,
    # et le masque causal garantit qu'elle ne depend jamais de ce dernier
    # slot lui-meme, seulement de ce qui le precede.
    generated = []

    for _ in range(max_letters):
        tokens_in_list = generated + [tokenizer.pad_id]
        tokens_in = torch.tensor([tokens_in_list], dtype=torch.long, device=device)

        logits = model(image, tokens_in)

        next_logits = logits[0, -1, :]
        next_id = torch.argmax(next_logits).item()

        if next_id == tokenizer.eos_id:
            break

        generated.append(next_id)

    return tokenizer.decode(generated + [tokenizer.eos_id])


@torch.no_grad()
def generate_batch(model, tokenizer, images: torch.Tensor, max_letters: int = 45, device: str = "cpu"):
    words = []
    for i in range(images.shape[0]):
        words.append(generate_word(model, tokenizer, images[i], max_letters, device))
    return words
