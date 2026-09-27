import string


class CharTokenizer:
    def __init__(self):
        self.letters = list(string.ascii_lowercase)
        self.stoi = {ch: i for i, ch in enumerate(self.letters)}

        self.eos_id = 26
        self.pad_id = 27
        self.stoi["<eos>"] = self.eos_id
        self.stoi["<pad>"] = self.pad_id

        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)  # 28

    def encode(self, word: str, add_eos: bool = True) -> list:
        ids = [self.stoi[ch] for ch in word]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, stop_at_eos: bool = True) -> str:
        chars = []
        for raw_id in ids:
            i = int(raw_id)
            if i == self.eos_id:
                if stop_at_eos:
                    break
                else:
                    continue
            if i == self.pad_id:
                continue
            chars.append(self.itos[i])
        return "".join(chars)


if __name__ == "__main__":
    tok = CharTokenizer()
    for word in ["largeredcircle", "smallbluesquareleftoflargeyellowtriangle"]:
        ids = tok.encode(word)
        back = tok.decode(ids)
        assert back == word, f"Tokenizer casse sur '{word}'"
        print(f"[OK] {word} -> {ids[:5]}... -> {back}")
    print("pad_id =", tok.pad_id, "| eos_id =", tok.eos_id, "| vocab_size =", tok.vocab_size)