"""PyTorch Dataset and collate function for ShapeScenes.

Loads the image + word pairs written by generate_data.py and returns
(image tensor, token id tensor) pairs, batched with a collate function
that pads to the longest word in the batch.
"""


import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class ShapeScenesDataset(Dataset):
    def __init__(self, processed_dir: str, split: str, tokenizer, blind: bool = False):
        pt_path = f"{processed_dir}/{split}.pt"
        data = torch.load(pt_path)

        self.images = data["images"]   # (N, 3, 64, 64) uint8
        self.words = data["words"]     # liste de N strings
        self.tokenizer = tokenizer
        self.blind = blind             # True => experience E1 (baseline aveugle)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx: int):
        image = self.images[idx].float() / 255.0   # uint8 [0,255] -> float [0,1]

        if self.blind:
            image = torch.zeros_like(image)          # E1 : le modele ne voit rien

        word = self.words[idx]
        token_ids = self.tokenizer.encode(word)       # ajoute <eos> automatiquement

        return image, token_ids   # longueur variable : collate_fn s'en charge


def make_collate_fn(pad_id: int):
    def collate_fn(batch):
        images, ids_list = zip(*batch)                # "transpose" la liste de tuples

        images = torch.stack(images)                  # (B, 3, 64, 64)

        id_tensors = [torch.tensor(ids, dtype=torch.long) for ids in ids_list]
        tokens = pad_sequence(id_tensors, batch_first=True, padding_value=pad_id)

        tokens_in = tokens
        tokens_out = tokens.clone()
        return images, tokens_in, tokens_out

    return collate_fn


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.tokenizer import CharTokenizer

    tokenizer = CharTokenizer()
    ds = ShapeScenesDataset("data", "train", tokenizer)
    print(f"Dataset charge : {len(ds)} exemples")

    loader = DataLoader(ds, batch_size=4, shuffle=True,
                         collate_fn=make_collate_fn(tokenizer.pad_id))

    images, tokens_in, tokens_out = next(iter(loader))
    print("images:", images.shape)
    print("tokens_in:", tokens_in.shape)
    for i in range(4):
        print(f"  mot {i} decode :", tokenizer.decode(tokens_in[i]))