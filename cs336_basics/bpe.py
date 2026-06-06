import os
from typing import BinaryIO
import multiprocessing as mp
import regex as re
from collections import defaultdict
from sortedcontainers import SortedList
import datetime
import json
# from memory_profiler import profile
# import mmap
from cs336_basics.config import * 


# @profile
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = max(
            chunk_boundaries[bi], 
            chunk_boundaries[bi - 1] + len(split_special_token) if bi > 0 else 0
        )
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += len(mini_chunk)

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


# @profile
# # 在模块级别预编译，避免每次调用都重新编译
# _PAT = re.compile(
#     r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# )

# def pretokenization(input_path, start, end, special_tokens):
#     # 按长度降序排列，避免短 token 匹配掉长 token 的前缀
#     special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
#     pat_special = re.compile("|".join(map(re.escape, special_tokens_sorted)))

#     result = defaultdict(int)

#     # 优化1: mmap + memoryview，避免将整个 chunk 复制到 Python 堆
#     with open(input_path, "rb") as f:
#         with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
#             view = memoryview(mm)
#             # bytes(view[start:end]) 只复制所需范围，而非整个文件
#             chunk = bytes(view[start:end]).decode("utf-8", errors="ignore")
#             del view  # 立即释放 memoryview 对象

#     # 优化2: finditer + pos/endpos 参数
#     # 直接在原字符串上划定搜索区间，既不创建片段列表，也不创建子字符串
#     last = 0
#     for m in pat_special.finditer(chunk):
#         if m.start() > last:
#             for tok in _PAT.finditer(chunk, last, m.start()):
#                 # 优化3: 直接用 bytes 作 key，而非 tuple
#                 result[tok.group().encode("utf-8")] += 1
#         last = m.end()

#     # 处理最后一个 special token 之后的尾部
#     for tok in _PAT.finditer(chunk, last):
#         result[tok.group().encode("utf-8")] += 1

#     # 处理完后主动释放，让 GC 可以在函数返回前回收
#     del chunk

#     return result
def pretokenization(input_path, start, end, special_tokens):
    pat_special = "|".join(map(re.escape, special_tokens))
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    # chunk = re.split(pat_special, chunk)
    # print(chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # Note: should use re.finditer to avoid storing the pre-tokenized words
    # chunk = [re.findall(PAT, c) for c in chunk if len(c) > 0]
    result = defaultdict(int)
    def _iter_segments(text, pattern):
        prev_end = 0
        for m in re.finditer(pattern, text):
            yield text[prev_end:m.start()]
            prev_end = m.end()
        yield text[prev_end:]
    for c in _iter_segments(chunk, pat_special):
        if c:
            for match in re.finditer(PAT, c):
                t = match.group()
                result[t.encode("utf-8")] += 1
    return result


# @profile
def bpe(input_path, vocab_size, special_tokens, num_processes=-1):
    if num_processes == -1:
        num_processes = mp.cpu_count()
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], ": BPE Started")

    #============== Vocabulary ==============
    vocab = {}
    id_to_vocab = []
    for s in special_tokens:
        vocab[s.encode("utf-8")] = len(vocab)
        id_to_vocab += [s.encode("utf-8")]
    for i in range(256):
        vocab[bytes([i])] = len(vocab)
        id_to_vocab += [bytes([i])]
    assert len(vocab) <= vocab_size, f"Error param: vocab_size should be at least {len(vocab)}"
    # print(vocab)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], ": Vocab Finished")

    #============== Pre-tokenization in Parallel ==============
    # Chunk the Corpus for Parallelization
    boundaries = find_chunk_boundaries(open(input_path, 'rb'), num_processes, b"<|endoftext|>")
    boundaries = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    # print(boundaries)

    # NOTE: chunks, i.e. len(boundaries) might be smaller than num_processes
    with mp.Pool(num_processes) as pool:
        # TODO: change this to startmap_async
        pre_toks = pool.starmap(pretokenization, boundaries)
    corpus = pre_toks[0]
    for pre_tok in pre_toks[1:]:
        for k, v in pre_tok.items():
            corpus[k] += v
    corpus = [[[vocab[bytes([i])] for i in k], v] for k, v in corpus.items()]
    # print([[b"".join([id_to_vocab[k1] for k1 in k]), v] for (k, v) in corpus])
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], ": Pre-Token Finished")
    
    #============== Merges ==============
    tok_to_freq = {}
    for i, (k, v) in enumerate(corpus):
        for iter in range(len(k)-1):
            pair = (k[iter], k[iter+1])
            if pair in tok_to_freq:
                ori = tok_to_freq[pair]
                ori[0] += v
                if i in ori[1]:
                    ori[1][i] += v
                else:
                    ori[1][i] = v
            else:
                tok_to_freq[pair] = [v, {i: v}]
    # print(tok_to_freq)
    sl = SortedList([(v[0], (id_to_vocab[k[0]], id_to_vocab[k[1]])) for k, v in tok_to_freq.items()])
    # print(sl)

    # Should update: chunks(corpus[:][0]), tok_to_freq, pq, merges, vocab
    merges = []
    while len(vocab) < vocab_size and sl:
        max_freq = sl[-1]
        # print(max_freq)
        # print(sl)
        freq_detail = tok_to_freq[(vocab[max_freq[1][0]], vocab[max_freq[1][1]])]
        assert freq_detail[0] == max_freq[0]
        new_bytes = max_freq[1][0]+max_freq[1][1]
        vocab[new_bytes] = len(vocab)
        id_to_vocab += [new_bytes]
        # print(max_freq[1][0]+max_freq[1][1], vocab[max_freq[1][0]+max_freq[1][1]])
        
        # print(freq_detail[1])
        for k in freq_detail[1].keys():
            c = corpus[k]
            # print(c)
            it = 0
            while it < len(c[0])-1:
                # print(it, c[0])
                if c[0][it] == vocab[max_freq[1][0]] and c[0][it+1] == vocab[max_freq[1][1]]:
                    if it > 0:
                        pre_pair = (c[0][it-1], c[0][it])
                        pre_pair_freq = tok_to_freq[pre_pair]
                        sl.remove((pre_pair_freq[0], (id_to_vocab[pre_pair[0]], id_to_vocab[pre_pair[1]])))
                        new_freq = pre_pair_freq[0] - c[1]
                        if new_freq > 0:
                            sl.add((new_freq, (id_to_vocab[pre_pair[0]], id_to_vocab[pre_pair[1]])))
                        pre_pair_freq[0] -= c[1]
                        if pre_pair_freq[0] == 0:
                            del tok_to_freq[pre_pair]
                        else:
                            pre_pair_freq[1][k] -= c[1]
                            if pre_pair_freq[1][k] == 0:
                                del pre_pair_freq[1][k]

                        new_pair = (c[0][it-1], vocab[new_bytes])
                        if new_pair in tok_to_freq:
                            sl.remove((tok_to_freq[new_pair][0], (id_to_vocab[new_pair[0]], id_to_vocab[new_pair[1]])))
                            sl.add((tok_to_freq[new_pair][0]+c[1], (id_to_vocab[new_pair[0]], id_to_vocab[new_pair[1]])))
                            tok_to_freq[new_pair][0] += c[1]
                            if k in tok_to_freq[new_pair][1]:
                                tok_to_freq[new_pair][1][k] += c[1]
                            else:
                                tok_to_freq[new_pair][1][k] = c[1]
                        else:
                            sl.add((c[1], (id_to_vocab[new_pair[0]], id_to_vocab[new_pair[1]])))
                            tok_to_freq[new_pair] = [c[1], {k: c[1]}]

                    if it < len(c[0])-2:
                        suf_pair = (c[0][it+1], c[0][it+2])
                        suf_pair_freq = tok_to_freq[suf_pair]
                        sl.remove((suf_pair_freq[0], (id_to_vocab[suf_pair[0]], id_to_vocab[suf_pair[1]])))
                        new_freq = suf_pair_freq[0] - c[1]
                        if new_freq > 0:
                            sl.add((new_freq, (id_to_vocab[suf_pair[0]], id_to_vocab[suf_pair[1]])))
                        suf_pair_freq[0] -= c[1]
                        if suf_pair_freq[0] == 0:
                            del tok_to_freq[suf_pair]
                        else: 
                            suf_pair_freq[1][k] -= c[1]
                            if suf_pair_freq[1][k] == 0:
                                del suf_pair_freq[1][k]

                        new_pair = (vocab[new_bytes], c[0][it+2])
                        if new_pair in tok_to_freq:
                            sl.remove((tok_to_freq[new_pair][0], (id_to_vocab[new_pair[0]], id_to_vocab[new_pair[1]])))
                            sl.add((tok_to_freq[new_pair][0]+c[1], (id_to_vocab[new_pair[0]], id_to_vocab[new_pair[1]])))
                            tok_to_freq[new_pair][0] += c[1]
                            if k in tok_to_freq[new_pair][1]:
                                tok_to_freq[new_pair][1][k] += c[1]
                            else:
                                tok_to_freq[new_pair][1][k] = c[1]
                        else:
                            sl.add((c[1], (id_to_vocab[new_pair[0]], id_to_vocab[new_pair[1]])))
                            tok_to_freq[new_pair] = [c[1], {k: c[1]}]

                    c[0] = c[0][:it] + [vocab[new_bytes]] + c[0][it+2:]
                    sl.remove((freq_detail[0], max_freq[1]))
                    new_freq = freq_detail[0] - c[1]
                    if new_freq > 0:
                        sl.add((new_freq, max_freq[1]))
                    freq_detail[0] -= c[1]
                    if freq_detail[0] == 0:
                        del tok_to_freq[(vocab[max_freq[1][0]], vocab[max_freq[1][1]])]
                    
                it += 1

        merges += [(max_freq[1][0], max_freq[1][1])]

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], ": Merge Finished")

    id_to_vocab = {i: b for i, b in enumerate(id_to_vocab)}
    return id_to_vocab, merges


if __name__ == "__main__":
    # file_path = ts_train_filepath
    # vocab_size = 10000
    # vocab_filepath = vocab_ts_10k_filepath
    # merges_filepath = merges_ts_10k_filepath
    file_path = owt_train_filepath
    vocab_size = 32000
    vocab_filepath = vocab_owt_32k_filepath
    merges_filepath = merges_owt_32k_filepath
    
    vocab, merges = bpe(file_path, vocab_size, special_tokens)

    # 1. Save the vocabulary (vocab) to a JSON file
    serializable_vocab = {
        k: v.decode("latin-1") for k, v in vocab.items()
    }
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)

    # 2. Save the merge rules to a JSON file (merges.json)
    serializable_merges = []
    for merge in merges:
        if isinstance(merge, (tuple, list)) and len(merge) == 2:
            p0 = merge[0].decode("latin-1")
            p1 = merge[1].decode("latin-1")
            serializable_merges.append([p0, p1])
        else:
            if isinstance(merge, bytes):
                merge = merge.decode("latin-1")
            serializable_merges.append(merge)

    with open(merges_filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_merges, f, ensure_ascii=False, indent=2)

    print(f"Successfully saved: {vocab_filepath} and {merges_filepath}")

    # Find the longest token by byte length
    longest_token_bytes = max(vocab.values(), key=len)
    
    # Decode it safely to a string so you can print it
    longest_token_str = longest_token_bytes.decode("utf-8")
    max_length = len(longest_token_bytes)

    print(f"The longest token is: '{longest_token_str}'")
    print(f"Length: {max_length} bytes")

    # import cProfile
    # import pstats
    # # Wrap your execution in cProfile.run()
    # print("Starting profiler...")
    # cProfile.run('bpe(file_path, 500, ["<|endoftext|>"], 8)', 'profile_stats')
    
    # # Print the stats
    # p = pstats.Stats('profile_stats')
    # p.sort_stats('cumulative').print_stats(20) # Prints the top 20 time-consuming calls