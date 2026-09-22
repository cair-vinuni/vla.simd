"""Export the SmolVLM2 (SmolLM2) byte-level BPE tokenizer to the shared flat format
the engine's tokenizer loads.

Run: python3 tools/convert_tokenizer.py
Files -> build/smolvla_tok/: vocab.txt, merges.txt, specials.txt
"""
import os, json
from huggingface_hub import hf_hub_download

REPO = "HuggingFaceTB/SmolVLM2-500M-Instruct"
OUT = "build/smolvla_tok"
os.makedirs(OUT, exist_ok=True)

tk = json.load(open(hf_hub_download(REPO, "tokenizer.json")))
model = tk["model"]
vocab = model["vocab"]
merges = model["merges"]
added = tk.get("added_tokens", [])

with open(os.path.join(OUT, "vocab.txt"), "w", encoding="utf-8") as f:
    for tok, i in vocab.items():
        f.write(f"{i}\t{tok}\n")
with open(os.path.join(OUT, "merges.txt"), "w", encoding="utf-8") as f:
    for m in merges:
        f.write((m if isinstance(m, str) else f"{m[0]} {m[1]}") + "\n")
with open(os.path.join(OUT, "specials.txt"), "w", encoding="utf-8") as f:
    for a in added:
        f.write(f"{a['id']}\t{a['content']}\n")
print(f"vocab={len(vocab)} merges={len(merges)} specials={len(added)}")

print("done")
