from __future__ import annotations
import os, re, json, random, argparse
from typing import List
import numpy as np

def _generate_logic_puzzles(n_puzzles: int, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    puzzles = []
    var_names = ["x", "y", "z", "a", "b", "count", "total", "result", "flag", "value"]
    ops = ["+", "-", "*"]
    for _ in range(n_puzzles):
        t = rng.randint(0, 4)
        if t == 0:
            v = rng.choice(var_names); val = rng.randint(0, 20); steps = [f"Let {v} = {val}."]
            for _ in range(rng.randint(2, 5)):
                op = rng.choice(ops); o = rng.randint(1, 10); steps.append(f"Set {v} = {v} {op} {o}.")
                val = val + o if op == "+" else (val - o if op == "-" else val * o)
            steps.append(f"Question: What is the value of {v}? Answer: {v} = {val}")
            puzzles.append(" ".join(steps))
        elif t == 1:
            v = rng.choice(var_names); val = rng.randint(0, 100); th = rng.randint(0, 100)
            yw, nw = rng.choice(["true", "yes", "positive"]), rng.choice(["false", "no", "negative"])
            puzzles.append(f"Given {v} = {val}. If {v} > {th}, the result is {yw}. Otherwise, the result is {nw}. Question: What is the result? Answer: {yw if val > th else nw}")
        elif t == 2:
            items = rng.sample(range(1, 50), k=rng.randint(3, 8))
            q = rng.choice(items) if rng.random() > 0.4 else rng.randint(1, 50)
            puzzles.append(f"The set S contains: {{{', '.join(map(str, sorted(items)))}}}. Question: Is {q} in S? Answer: {'yes' if q in items else 'no'}")
        elif t == 3:
            st = rng.randint(0, 20); step = rng.randint(1, 7); seq = [st + step * j for j in range(5)]
            puzzles.append(f"Complete the sequence: {', '.join(map(str, seq))}, ? Answer: {st + step * 5}")
        else:
            vu = rng.sample(var_names, k=rng.randint(2, 4)); vals = {v: rng.randint(0, 10) for v in vu}
            steps = [f"Let {v} = {vals[v]}" for v in vu]
            for _ in range(rng.randint(2, 4)):
                v1, v2 = rng.sample(vu, 2); op = rng.choice(ops); steps.append(f"Set {v1} = {v1} {op} {v2}")
                vals[v1] = vals[v1] + vals[v2] if op == "+" else (vals[v1] - vals[v2] if op == "-" else vals[v1] * vals[v2])
            qv = rng.choice(vu); steps.append(f"Question: What is {qv}? Answer: {qv} = {vals[qv]}")
            puzzles.append(". ".join(steps) + ".")
    return puzzles

def _generate_code_snippets(n_snippets: int, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    templates = [
        lambda: f"def process_items(items):\n    result = []\n    for item in items:\n        if item > {rng.randint(0, 10)}:\n            result.append(item * {rng.randint(2, 5)})\n    return result",
        lambda: f"class Buffer:\n    def __init__(self, limit={rng.randint(8, 64)}):\n        self.limit = limit\n        self.data = []\n    def add(self, item):\n        if len(self.data) < self.limit:\n            self.data.append(item)\n            return True\n        return False",
        lambda: "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    ]
    return [rng.choice(templates)() for _ in range(n_snippets)]

_TRIVIA = re.compile(r"\b(footballer|midfielder|goalkeeper|striker|batting|innings|discography|album|episode|season \d+|municipality|commune in|village in|census|born \d{1,2} \w+ \d{4})\b", re.I)

def _fetch_real_simple_wiki(n_articles: int) -> List[str]:
    from datasets import load_dataset
    print(f"Streaming up to {n_articles:,} filtered Simple English Wikipedia articles...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
    texts, scanned = [], 0
    for row in ds:
        scanned += 1
        title, body = row.get("title", "").strip(), row.get("text", "").strip()
        if len(body) < 350 or _TRIVIA.search(body[:500]):
            continue
        texts.append(f"{title}\n{re.sub(r'\n{3,}', '\n\n', body[:3500]).strip()}")
        if len(texts) % 5000 == 0:
            print(f"  Kept {len(texts):,}/{n_articles:,} articles (scanned {scanned:,})...")
        if len(texts) >= n_articles:
            break
    print(f"  ✓ Collected {len(texts):,} clean articles.")
    return texts

class TokenizerWrapper:
    def __init__(self, tokenizer, vocab_size: int, eos_id: int, pad_id: int):
        self.tokenizer, self.vocab_size, self.eos_id, self.pad_id = tokenizer, vocab_size, eos_id, pad_id
        self._is_tiktoken = hasattr(tokenizer, "encode_ordinary")
        self._is_tokenizers = hasattr(tokenizer, "get_vocab_size")
    def encode(self, text: str) -> List[int]:
        if self._is_tiktoken: return self.tokenizer.encode_ordinary(text)
        if self._is_tokenizers: return self.tokenizer.encode(text).ids
        return self.tokenizer.encode(text, add_special_tokens=False)
    def decode(self, ids: List[int]) -> str:
        if self._is_tiktoken: return self.tokenizer.decode(ids)
        if self._is_tokenizers: return self.tokenizer.decode(ids)
        return self.tokenizer.decode(ids, skip_special_tokens=True)
    @staticmethod
    def for_dev_15m(texts: List[str], save_dir: str) -> "TokenizerWrapper":
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        path = os.path.join(save_dir, "tokenizer_8k.json")
        if os.path.exists(path):
            tok = Tokenizer.from_file(path)
        else:
            tok = Tokenizer(models.BPE()); tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder = decoders.ByteLevel()
            tok.train_from_iterator(texts, trainer=trainers.BpeTrainer(vocab_size=8192, special_tokens=["<|pad|>", "<|eos|>", "<|unk|>"], min_frequency=2))
            tok.save(path)
        v = tok.get_vocab()
        return TokenizerWrapper(tok, tok.get_vocab_size(), v.get("<|eos|>", 1), v.get("<|pad|>", 0))
    @staticmethod
    def for_gpt2() -> "TokenizerWrapper":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        eos_id = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        return TokenizerWrapper(enc, 50257, eos_id, eos_id)

def build_memmap_dataset(texts: List[str], tokenizer: TokenizerWrapper, output_path: str, seq_len: int, split: str):
    dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.uint32
    all_tokens = []
    for i, text in enumerate(texts):
        toks = tokenizer.encode(text)
        if toks:
            all_tokens.extend(toks); all_tokens.append(tokenizer.eos_id)
        if (i + 1) % 10000 == 0:
            print(f"  Tokenized {i+1:,}/{len(texts):,} texts ({len(all_tokens):,} tokens)...")
    chunk_len = seq_len + 1
    n_chunks = max(1, len(all_tokens) // chunk_len)
    if len(all_tokens) < chunk_len:
        all_tokens.extend([tokenizer.pad_id] * (chunk_len - len(all_tokens)))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mm = np.memmap(output_path, dtype=dtype, mode="w+", shape=(n_chunks, chunk_len))
    for i in range(n_chunks):
        mm[i] = np.array(all_tokens[i * chunk_len : (i + 1) * chunk_len], dtype=dtype)
    mm.flush()
    with open(output_path.replace(".bin", "_meta.json"), "w") as f:
        json.dump({"split": split, "n_chunks": n_chunks, "chunk_len": chunk_len, "seq_len": seq_len,
                   "total_tokens": n_chunks * chunk_len, "vocab_size": tokenizer.vocab_size,
                   "dtype": dtype.__name__, "eos_id": tokenizer.eos_id, "pad_id": tokenizer.pad_id}, f, indent=2)
    print(f"  ✓ Wrote {output_path} ({os.path.getsize(output_path)/(1024*1024):.1f} MB, {n_chunks:,} chunks)")

if __name__ == "__main__":
    from configs import get_config
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="dev_15m", choices=["dev_15m", "local_200m", "prod_500m"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--n-wiki", type=int, default=500)
    p.add_argument("--n-logic", type=int, default=300)
    p.add_argument("--n-code", type=int, default=200)
    args = p.parse_args()
    cfg = get_config(args.preset)
    wiki = _fetch_real_simple_wiki(args.n_wiki) if args.download else [f"Water is H2O and covers 71 percent of Earth." for _ in range(args.n_wiki)]
    texts = wiki + _generate_logic_puzzles(args.n_logic) + _generate_code_snippets(args.n_code)
    random.Random(42).shuffle(texts)
    n_val = max(1, int(len(texts) * 0.02))
    os.makedirs(args.data_dir, exist_ok=True)
    tok = TokenizerWrapper.for_dev_15m(texts[n_val:], args.data_dir) if args.preset == "dev_15m" else TokenizerWrapper.for_gpt2()
    build_memmap_dataset(texts[n_val:], tok, os.path.join(args.data_dir, f"{args.preset}_train.bin"), cfg.seq_len, "train")
    build_memmap_dataset(texts[:n_val], tok, os.path.join(args.data_dir, f"{args.preset}_val.bin"), cfg.seq_len, "val")
