import os
import re
import csv

import torch
from torch.utils.data import DataLoader
import datasets

from dataset import SmileDataset, SmileCollator
from tokenizer import SmilesTokenizer
from model import GPTConfig, GPT


# 你需要改成真实路径
DATASET_PATHS = {
    "DeepSMILES": "/share/home/tm866079609100000/a866071650/doomx/MolGen/DATA/DeepSMILES-300M",
    "FragSeq":    "/share/home/tm866079609100000/a866071650/doomx/MolGen/DATA/FragSeq-300M",
    "FragSeqV2":  "/share/home/tm866079609100000/a866071650/doomx/MolGen/DATA/FragSeqV2-300M",
    "SAFE":       "/share/home/tm866079609100000/a866071650/doomx/MolGen/DATA/SAFE-300M",
    "SMILES":     "/share/home/tm866079609100000/a866071650/doomx/MolGen/DATA/SMILES-300M",
}

MODEL_SPECS = {
    "1M":   dict(n_layer=4,  n_head=4,  n_embd=160),
    "4M":  dict(n_layer=5,  n_head=4,  n_embd=256),
    "16M":  dict(n_layer=5, n_head=4, n_embd=512),
    "56M":  dict(n_layer=6,  n_head=16, n_embd=768),
    "85M": dict(n_layer=12, n_head=12, n_embd=768),
    "152M": dict(n_layer=12, n_head=16, n_embd=1024),
    "278M": dict(n_layer=22, n_head=16, n_embd=1024),
    "650M": dict(n_layer=13, n_head=32, n_embd=2048),
}


def norm_dash(s: str) -> str:
    return s.replace("—", "-").replace("–", "-").replace("−", "-")


def parse_run_name(pt_name: str):
    # DeepSMILES-16M-1B-Epoch2.pt
    base = os.path.splitext(os.path.basename(pt_name))[0]
    base = norm_dash(base)
    m = re.match(r"^(?P<mod>[^-]+)-(?P<ms>\d+M)-(?P<ds>(?:\d+M|\d+B))-(?:Epoch(?P<ep>\d+))$", base)
    if not m:
        return None

    mod = m.group("mod")
    ms = m.group("ms")
    ds = m.group("ds")
    ep = int(m.group("ep"))

    ms_num = float(ms[:-1])
    ds_num = float(ds[:-1]) if ds.endswith("M") else float(ds[:-1]) * 1000.0  # 1B -> 1000

    return {
        "name": base,
        "modality": mod,
        "model_size": ms,
        "model_size_num": ms_num,
        "data_size_num": ds_num,
        "epochs": float(ep),
    }


def build_tokenizer():
    tok = SmilesTokenizer("./vocabs/vocab.txt")
    tok.bos_token = "[BOS]"
    tok.bos_token_id = tok.convert_tokens_to_ids("[BOS]")
    tok.eos_token = "[EOS]"
    tok.eos_token_id = tok.convert_tokens_to_ids("[EOS]")
    tok.sep_token = "[SEP]"
    tok.sep_token_id = tok.convert_tokens_to_ids("[SEP]")
    return tok


def make_model(model_size: str, vocab_size: int, device: str):
    spec = MODEL_SPECS[model_size]
    mconf = GPTConfig(vocab_size=vocab_size, **spec)
    return GPT(mconf).to(device)


def load_state_dict(path: str, device: str):
    sd = torch.load(path, map_location=device, weights_only=False)
    k0 = next(iter(sd.keys()))
    if k0.startswith("module."):
        sd = {k[7:]: v for k, v in sd.items()}
    return sd


@torch.no_grad()
def eval_like_trainer(model, loader, tokenizer, device: str):
    model.eval()
    losses = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.amp.autocast("cuda"):
            logits, loss, _ = model(x, tokenizer, y)
            loss = loss.mean()

        losses.append(loss.item())

    return sum(losses) / max(len(losses), 1)


def write_csv(out_csv: str, rows: list):
    rows.sort(key=lambda r: r["name"])
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "model size", "data size", "epochs", "loss"])
        w.writeheader()
        w.writerows(rows)


def main(ckpt_dir: str, out_csv: str, batch_size: int):
    device = "cuda:0"
    num_workers = 5

    tok = build_tokenizer()
    collator = SmileCollator(tok)

    pt_files = [fn for fn in os.listdir(ckpt_dir) if fn.endswith(".pt")]
    pt_files.sort()

    parsed = []
    for fn in pt_files:
        info = parse_run_name(fn)
        if info is None:
            continue
        if info["modality"] not in DATASET_PATHS:
            continue
        parsed.append((os.path.join(ckpt_dir, fn), info))

    if not parsed:
        write_csv(out_csv, [])
        print(f"[OK] no matched ckpt in {ckpt_dir}, wrote empty csv -> {out_csv}")
        return

    # 只加载一次该模态的 validation dataloader
    # 一个 ckpt_dir 理论上只包含同一模态，但这里按解析结果取第一个
    modality = parsed[0][1]["modality"]
    raw = datasets.load_from_disk(DATASET_PATHS[modality])
    valid = SmileDataset(raw, data_type="validation", tokenizer=tok)
    loader = DataLoader(
        valid,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    rows = []
    for ckpt_path, info in parsed:
        model = make_model(info["model_size"], tok.vocab_size, device)
        sd = load_state_dict(ckpt_path, device)
        model.load_state_dict(sd, strict=True)

        loss = eval_like_trainer(model, loader, tok, device)

        rows.append({
            "name": info["name"],
            "model size": float(info["model_size_num"]),
            "data size": float(info["data_size_num"]),
            "epochs": float(info["epochs"]),
            "loss": float(loss),
        })

        del model
        torch.cuda.empty_cache()

        print(f"[OK] {info['name']}  loss={loss:.6f}")

    write_csv(out_csv, rows)
    print(f"[OK] wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    main(args.ckpt_dir, args.out_csv, args.batch_size)
