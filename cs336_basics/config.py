import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

special_tokens = ["<|endoftext|>"]

ts_train_filepath = os.path.join(BASE_DIR, "../data/TinyStoriesV2-GPT4-train.txt")
ts_valid_filepath = os.path.join(BASE_DIR, "../data/TinyStoriesV2-GPT4-valid.txt")
owt_train_filepath = os.path.join(BASE_DIR, "../data/owt_train.txt")
owt_valid_filepath = os.path.join(BASE_DIR, "../data/owt_valid.txt")

vocab_ts_10k_filepath = os.path.join(BASE_DIR, 'vocab_ts_10k.json')
merges_ts_10k_filepath = os.path.join(BASE_DIR, 'merges_ts_10k.json')
vocab_owt_32k_filepath = os.path.join(BASE_DIR, 'vocab_owt_32k.json')
merges_owt_32k_filepath = os.path.join(BASE_DIR, 'merges_owt_32k.json')

ts_train_encoded_ids_filepath = os.path.join(BASE_DIR, 'ts_train_encoded_ids')
ts_valid_encoded_ids_filepath = os.path.join(BASE_DIR, 'ts_valid_encoded_ids')
owt_train_encoded_ids_filepath = os.path.join(BASE_DIR, 'owt_train_encoded_ids')
owt_valid_encoded_ids_filepath = os.path.join(BASE_DIR, 'owt_valid_encoded_ids')