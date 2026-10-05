"""Frozen ESM features, overlapping long-protein windows and resumable masked LLRs."""
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .common import AA, save_json
from .corpus import read_corpus


def window_starts(length, window, overlap):
    if not 0 <= overlap < window:
        raise ValueError("Require 0 <= overlap < window")
    if length <= window:
        return [0]
    return sorted(set(list(range(0, length-window+1, window-overlap)) + [length-window]))


def extract(corpus, resources, cache, model_name="facebook/esm2_t33_650M_UR50D", window=768, overlap=128, llr=True):
    from transformers import AutoTokenizer, EsmForMaskedLM
    root = Path(cache)
    root.mkdir(parents=True, exist_ok=True)
    sequences = json.loads((Path(resources) / "sequences.json").read_text())
    d = read_corpus(corpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).to(device).eval()
    model.requires_grad_(False)
    if window > model.config.max_position_embeddings - 2:
        raise ValueError("Window exceeds ESM positional capacity")
    aa_ids = torch.tensor([tokenizer.convert_tokens_to_ids(a) for a in AA], device=device)
    manifest = {"model": model_name, "revision": getattr(model.config, "_commit_hash", None),
                "dim": model.config.hidden_size, "window": window, "overlap": overlap,
                "llr": "masked_marginal" if llr else "disabled", "schema": 1}
    if (root / "manifest.json").exists() and json.loads((root / "manifest.json").read_text()) != manifest:
        raise ValueError("Cache configuration mismatch: choose a separate cache directory")
    save_json(root / "manifest.json", manifest)
    for gene in sorted(d.protein_reference_id.unique()):
        record = sequences[gene]
        seq, key = record["sequence"], record["key"]
        folder = root / key
        folder.mkdir(exist_ok=True)
        if not (folder / "esm.npy").exists():
            total = np.zeros((len(seq), manifest["dim"]), np.float32)
            weights = np.zeros((len(seq), 1), np.float32)
            for start in window_starts(len(seq), window, overlap):
                chunk = seq[start:start+window]
                tokens = tokenizer(chunk, return_tensors="pt").to(device)
                with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                    emb = model.esm(**tokens).last_hidden_state[0, 1:len(chunk)+1].float().cpu().numpy()
                w = np.minimum(np.arange(len(chunk))+1, np.arange(len(chunk), 0, -1)).clip(max=max(overlap, 1))[:, None]
                total[start:start+len(chunk)] += emb*w
                weights[start:start+len(chunk)] += w
            target = folder / "esm.tmp.npy"
            np.save(target, (total/weights).astype(np.float16))
            target.replace(folder / "esm.npy")
            save_json(folder / "sequence.json", record)
        if llr:
            path = folder / "llr.npy"
            scores = np.load(path) if path.exists() else np.full((len(seq), 20), np.nan, np.float32)
            sites = sorted(set(d.loc[d.protein_reference_id == gene, "ref_pos"] - 1))
            pending = [p for p in sites if not np.isfinite(scores[p]).all()]
            for number, p in enumerate(tqdm(pending, desc=f"{gene} masked LLR")):
                start = max(0, min(p-window//2, len(seq)-window))
                chunk = seq[start:start+window]
                tokens = tokenizer(chunk, return_tensors="pt").to(device)
                local = p-start+1
                tokens["input_ids"][0, local] = tokenizer.mask_token_id
                with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                    logits = model(**tokens).logits[0, local, aa_ids].float()
                scores[p] = (logits-logits[AA.index(seq[p])]).cpu().numpy()
                if (number+1) % 50 == 0 or number+1 == len(pending):
                    tmp = folder / "llr.tmp.npy"
                    np.save(tmp, scores)
                    tmp.replace(path)
        print(f"{gene}: feature cache ready", flush=True)
