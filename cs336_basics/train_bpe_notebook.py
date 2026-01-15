"""
Helper function for tests.ipynb
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
import regex as re
import os
from typing import BinaryIO
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def find_chunk_boundaries(
    file: str,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file = open(file, "rb")
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
        initial_position = chunk_boundaries[bi]
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
            initial_position += mini_chunk_size

    file.close()
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _process_chunk(file, start, end, special_tokens):
    # Consider streaming the file inputs (However, this makes special token splitting particularly complicated)
    # Must split special tokens BEFORE regex nextiter streaming, 
    # since that would cause splitting conflicts otherwise.
    # Fixed-mini-chunking as done in find_chunk_boundaries risks splitting at exactly special tokens
    # Thus, either adding alpha tunable parameter or repeated calls to find_chunk_boundaries are viable potential solutions.
    f = open(file, "rb")
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="replace")
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
    # Remove all the special tokens before pretokenization
    splitter = re.compile("|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)))
    split_chunk = splitter.split(chunk)

    pretokens = {}
    for segment in split_chunk:
        for i in PAT.finditer(segment):
            pretoken = i.group(0)
            pretokens[pretoken] = pretokens.get(pretoken, 0) + 1
    f.close()
    return pretokens

def TrainBPE(input_path, vocab_size, special_tokens):
    num_processes = 4

    loop_limit = 10000
    memory_limit = 5000000 # technically approximate memory average

    # Arbitrary tunable parameter (make this fixed or tunable based on memory limit)
    alpha = min(loop_limit, (os.path.getsize(input_path) + memory_limit - 1) // memory_limit)
    print("Number of chunks:", alpha)

    # Arbitrarily set the zero-th index of special_tokens to be the end-of-text token (assume special_tokens is nonempty)
    boundaries =  find_chunk_boundaries(input_path, alpha, special_tokens[0].encode("utf-8"))
    
    jobs = []
    chunk_freqs = {}
    with ProcessPoolExecutor(max_workers=num_processes) as ex:
    # with ThreadPoolExecutor(max_workers=num_processes) as ex:
        for i in range(len(boundaries) - 1):
            jobs.append(ex.submit(_process_chunk, input_path, boundaries[i], boundaries[i + 1], special_tokens))

        for j in as_completed(jobs):
            print("test: ", j)
            for k, v in j.result().items():
                chunk_freqs[k] = chunk_freqs.get(k, 0) + v
    
    freqs = {}
    for token, cnt in chunk_freqs.items():
        byte_token = token.encode("utf-8")
        # print("Token:", token)
        # print("Byte token:", byte_token)
        bchars = tuple(bytes([b]) for b in byte_token)
        freqs[bchars] = cnt
    
    # Now that pretoken frequencies are acculumated and tupled,
    # We can start merging most frequent bytes
    vocab = {i : bytes([i]) for i in range(256)}
    cnt = 256
    for token in special_tokens:
        binary_token = token.encode("utf-8")
        if binary_token not in vocab.values():
            vocab[cnt] = binary_token
            cnt += 1

    pair_freqs = {}
    pair_index = {}
    for bchars, freq in freqs.items():
        for i in range(len(bchars) - 1):
            p = (bchars[i], bchars[i + 1])
            pair_freqs[p] = pair_freqs.get(p, 0) + freq
            if p not in pair_index:
                pair_index[p] = set()
            pair_index[p].add(bchars)

    merges = []
    while len(vocab) < vocab_size and pair_freqs:
        byte_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
        # print("Byte pair:", byte_pair)
        merges.append(byte_pair)

        m = byte_pair[0] + byte_pair[1]
        if m not in vocab.values():
            vocab[cnt] = m
            cnt += 1

        affected = pair_index.pop(byte_pair, set())
        if not affected:
            pair_freqs[byte_pair] = 0
            continue

        new_freqs = {}
        for bchars in list(affected):
            freq = freqs.get(bchars, 0)
            if freq == 0:
                continue

            n = len(bchars)
            for i in range(n - 1):
                p = (bchars[i], bchars[i + 1])
                pair_freqs[p] = pair_freqs.get(p, 0) - freq
                s = pair_index.get(p)
                if s is not None:
                    s.discard(bchars)

            out = []
            i = 0
            while i < n:
                if i + 1 < n and bchars[i] == byte_pair[0] and bchars[i + 1] == byte_pair[1]:
                    out.append(m)
                    i += 2
                else:
                    out.append(bchars[i])
                    i += 1
            t = tuple(out)

            freqs.pop(bchars, None)
            freqs[t] = freqs.get(t, 0) + freq
            new_freqs[t] = new_freqs.get(t, 0) + freq

        for t, f in new_freqs.items():
            for i in range(len(t) - 1):
                p = (t[i], t[i + 1])
                pair_freqs[p] = pair_freqs.get(p, 0) + f
                if p not in pair_index:
                    pair_index[p] = set()
                pair_index[p].add(t)

    return vocab, merges