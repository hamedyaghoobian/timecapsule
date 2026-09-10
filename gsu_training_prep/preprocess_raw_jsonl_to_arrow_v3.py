#!/usr/bin/env python3
"""
Preprocess raw TimeCapsuleLLM v3 JSONL.GZ shards into fixed-length Arrow shards.

Pipeline:
raw JSONL.GZ records with {id, text, metadata}
-> tokenize text with local tokenizer
-> add BOS/EOS
-> concatenate token streams
-> pack into fixed 4096-token rows
-> write Arrow shards with input_ids and attention_mask

This is a preparation utility only. It does not launch training.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from datasets import Dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing raw .jsonl.gz shards")
    parser.add_argument("--output_dir", required=True, help="Directory to write packed Arrow shards")
    parser.add_argument("--tokenizer_dir", required=True, help="Directory containing tokenizer.json")
    parser.add_argument("--sequence_length", type=int, default=4096)
    parser.add_argument("--target_sequences_per_shard", type=int, default=8192)
    parser.add_argument("--add_bos", action="store_true", default=True)
    parser.add_argument("--add_eos", action="store_true", default=True)
    parser.add_argument("--no_add_bos", action="store_false", dest="add_bos")
    parser.add_argument("--no_add_eos", action="store_false", dest="add_eos")
    return parser.parse_args()


def flush_arrow(output_dir: Path, shard_index: int, rows: list[dict]) -> Path:
    ds = Dataset.from_list(rows)
    out_dir = output_dir / f"packed_arrow_{shard_index:05d}"
    ds.save_to_disk(str(out_dir))
    return out_dir


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tokenizer_dir = Path(args.tokenizer_dir)
    tokenizer_path = tokenizer_dir / "tokenizer.json"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    vocab = tokenizer.get_vocab()
    bos_id = vocab["<s>"]
    eos_id = vocab["</s>"]

    shard_paths = sorted(input_dir.glob("*.jsonl.gz"))
    if not shard_paths:
        raise RuntimeError(f"No .jsonl.gz files found in {input_dir}")

    buffer: list[int] = []
    packed_rows: list[dict] = []
    output_shard_index = 1

    doc_count = 0
    token_count = 0
    packed_sequence_count = 0
    dropped_remainder_tokens = 0

    stats = {
        "input_shards": len(shard_paths),
        "documents": 0,
        "tokens_total_before_packing": 0,
        "packed_sequences": 0,
        "dropped_remainder_tokens": 0,
        "output_arrow_shards": 0,
    }

    for shard_path in tqdm(shard_paths, desc="raw shards"):
        with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                text = obj.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue

                ids = tokenizer.encode(text).ids
                if args.add_bos:
                    ids = [bos_id] + ids
                if args.add_eos:
                    ids = ids + [eos_id]

                buffer.extend(ids)
                doc_count += 1
                token_count += len(ids)

                while len(buffer) >= args.sequence_length:
                    seq = buffer[: args.sequence_length]
                    del buffer[: args.sequence_length]
                    packed_rows.append(
                        {
                            "input_ids": seq,
                            "attention_mask": [1] * args.sequence_length,
                        }
                    )
                    packed_sequence_count += 1

                    if len(packed_rows) >= args.target_sequences_per_shard:
                        flush_arrow(output_dir, output_shard_index, packed_rows)
                        output_shard_index += 1
                        packed_rows = []

    dropped_remainder_tokens = len(buffer)
    if packed_rows:
        flush_arrow(output_dir, output_shard_index, packed_rows)
        output_shard_index += 1

    stats["documents"] = doc_count
    stats["tokens_total_before_packing"] = token_count
    stats["packed_sequences"] = packed_sequence_count
    stats["dropped_remainder_tokens"] = dropped_remainder_tokens
    stats["output_arrow_shards"] = output_shard_index - 1

    summary_path = output_dir / "preprocessing_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
