import os
from typing import BinaryIO
import multiprocessing as mp
import regex as re
from collections import defaultdict
from sortedcontainers import SortedList
import datetime
import json
# from memory_profiler import profile
    

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
def pretokenization(input_path, start, end, special_tokens):
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    pat_special = "|".join(map(re.escape, special_tokens))
    chunk = re.split(pat_special, chunk)
    # print(chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # Note: should use re.finditer to avoid storing the pre-tokenized words
    # chunk = [re.findall(PAT, c) for c in chunk if len(c) > 0]
    result = defaultdict(int)
    for c in chunk:
        if c:
            for match in re.finditer(PAT, c):
                t = match.group()
                result[tuple(t.encode("utf-8"))] += 1
    return result


# @profile
def bpe(input_path, vocab_size, special_tokens, num_processes=1):
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
    # file_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    # file_path = "/Users/zxj//Desktop/AI/cs336/assignment1-basics/data/test"
    # file_path = "/Users/zxj//Desktop/AI/cs336/assignment1-basics/tests/fixtures/corpus.en"
    # file_path = "../data/TinyStoriesV2-GPT4-train.txt"
    file_path = "../data/owt_train.txt"
    vocab, merges = bpe(file_path, 32000, ["<|endoftext|>"], 1)

    # 1. Save the vocabulary (vocab) to a JSON file
    # We must decode the bytes objects to strings because JSON cannot serialize bytes.
    # We use errors="backslashreplace" (or "replace") to handle invalid UTF-8 byte sequences safely.
    serializable_vocab = {
        k: v.decode("utf-8", errors="backslashreplace") for k, v in vocab.items()
    }
    
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)

    # 2. Save the merge rules to a plain text file (merges.txt)
    with open("merges.txt", "w", encoding="utf-8") as f:
        for merge in merges:
            if isinstance(merge, (tuple, list)) and len(merge) == 2:
                # Decode the bytes to strings before writing to avoid b'...' formatting
                p0 = merge[0].decode("utf-8", errors="backslashreplace")
                p1 = merge[1].decode("utf-8", errors="backslashreplace")
                f.write(f"{p0} {p1}\n")
            else:
                # Fallback if it's already a string/bytes
                if isinstance(merge, bytes):
                    merge = merge.decode("utf-8", errors="backslashreplace")
                f.write(f"{merge}\n")

    print("Successfully saved: vocab.json and merges.txt")

    # Find the longest token by byte length
    longest_token_bytes = max(vocab.values(), key=len)
    
    # Decode it safely to a string so you can print it
    longest_token_str = longest_token_bytes.decode("utf-8", errors="backslashreplace")
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