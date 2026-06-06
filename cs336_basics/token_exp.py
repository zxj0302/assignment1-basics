from cs336_basics.tokenizer import Tokenizer
import regex as re
from cs336_basics.config import * 
import time
import json
import mmap
import numpy as np


def sample_docs(input_filepath, num_docs, special_tokens=["<|endoftext|>"]):
    with open(input_filepath, "r", encoding="utf-8") as f:
        text = f.read()
    pat_special = "|".join(map(re.escape, special_tokens))
    prev_end = 0
    count = 0
    for m in re.finditer(pat_special, text):
        yield text[prev_end:m.end()]
        prev_end = m.end()
        count += 1
        if num_docs > 0 and count >= num_docs:
            return

def compute_compression_ratio(vocab_filepath, merges_filepath, special_tokens, input_filepath, num_doc=10):
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    text_it = sample_docs(input_filepath, num_doc)
    text = [t for t in text_it]
    bytes_len = [len(t.encode("utf-8")) for t in text]
    start = time.time()
    token_len = [len(tokenizer.encode(t)) for t in text]
    time_elapsed = time.time() - start
    print(f"Time Elapsed for Encoding: {time_elapsed} Seconds")
    throughput = sum(bytes_len)/time_elapsed
    print(f"Estimated Throughput: {throughput} bytes/second")
    print(f"Estimated Time Consumption for 825GB text: {825*10**9/throughput//3600} Hours")
    bytes_token_len = zip(bytes_len, token_len)
    # print(f"Token Length & Bytes Length Pairs for {num_doc} Documents: ")
    # print(bytes_token_len)

    compression_ratio = [p[0]/p[1] for p in bytes_token_len]
    # print("Compression Ratio for Each Document: ")
    # print(compression_ratio)
    print(f"Everage Compression Ratio: {sum(compression_ratio)/len(compression_ratio)}")

# def encode_file(vocab_filepath, merges_filepath, special_tokens, input_filepath, ids_filepath):
#     tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
#     with open(input_filepath, "r", encoding="utf-8") as f:
#         text = f.read()
#     encoded = tokenizer.encode(text)
#     with open(ids_filepath, "w", encoding="utf-8") as f:
#         json.dump(encoded, f, indent=2)

def encode_file(vocab_filepath, merges_filepath, special_tokens, input_filepath, ids_filepath):
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    pat_special = b"|".join(re.escape(t.encode("utf-8")) for t in special_tokens)
    token_buffer = []
    FLUSH_THRESHOLD = 500_000
    total_tokens = 0

    with open(input_filepath, "rb") as fin, open(ids_filepath, "wb") as fout:
        with mmap.mmap(fin.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            prev_end = 0
            for m in re.finditer(pat_special, mm):
                doc_text = mm[prev_end:m.end()].decode("utf-8")
                prev_end = m.end()

                tokens = tokenizer.encode(doc_text)
                token_buffer.extend(tokens)

                if len(token_buffer) >= FLUSH_THRESHOLD:
                    np.array(token_buffer, dtype=np.uint16).tofile(fout)
                    total_tokens += len(token_buffer)
                    token_buffer = []
                    print(f"Flushed, total: {total_tokens:,}")

            if prev_end < len(mm):
                remaining = mm[prev_end:].decode("utf-8")
                if remaining.strip():
                    token_buffer.extend(tokenizer.encode(remaining))

        if token_buffer:
            np.array(token_buffer, dtype=np.uint16).tofile(fout)
            total_tokens += len(token_buffer)

    print(f"Done! Total tokens: {total_tokens:,} → {ids_filepath}")


if __name__ == "__main__":
    # NOTE: only doing experiments on TinyStories, as OpenWebText is too large.
    # num_doc = 10
    
    # (1) compute the compression ratio
    # print("Quantifying on TinyStories Valid Dataset: ")
    # compute_compression_ratio(vocab_ts_10k_filepath, merges_ts_10k_filepath, special_tokens, ts_valid_filepath, num_doc)

    # print("\nQuantifying on OpenWebText Valid Dataset: ")
    # compute_compression_ratio(vocab_ts_10k_filepath, merges_ts_10k_filepath, special_tokens, owt_valid_filepath, num_doc)

    # (2) encode TinyStories and OpenWebText to token IDs
    # encode_file(vocab_ts_10k_filepath, merges_ts_10k_filepath, special_tokens, ts_train_filepath, ts_train_encoded_ids_filepath)
    # tokens = np.fromfile(ts_train_encoded_ids_filepath, dtype=np.uint16)
    # print(tokens.shape, tokens[:10])
    # encode_file(vocab_ts_10k_filepath, merges_ts_10k_filepath, special_tokens, ts_valid_filepath, ts_valid_encoded_ids_filepath)
    # tokens = np.fromfile(ts_valid_encoded_ids_filepath, dtype=np.uint16)
    # print(tokens.shape, tokens[:10])
    encode_file(vocab_owt_32k_filepath, merges_owt_32k_filepath, special_tokens, owt_train_filepath, owt_train_encoded_ids_filepath)
    encode_file(vocab_owt_32k_filepath, merges_owt_32k_filepath, special_tokens, owt_valid_filepath, owt_valid_encoded_ids_filepath)