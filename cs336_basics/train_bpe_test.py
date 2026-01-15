# from concurrent.futures import ProcessPoolExecutor, as_completed
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

    # Arbitrary tunable parameter (make this fixed or tunable based on memory limit)
    alpha = 1

    # Set loop limit
    beta = 10000

    while True:
        # Arbitrarily set the zero-th index of special_tokens to be the end-of-text token (assume special_tokens is nonempty)
        boundaries =  find_chunk_boundaries(input_path, num_processes * alpha, special_tokens[0].encode("utf-8"))
        max_gap = 0
        for i in range(len(boundaries) - 1):
            max_gap = max(max_gap, boundaries[i + 1] - boundaries[i])
        if max_gap > 5000000:
            alpha *= 2
        else:
            break
        beta -= 1
        if beta == 0:
            break
    print(alpha)
    print(boundaries)

    # Potential bottleneck: MERGING frequencies
    # I think one can tank the extra time it takes for merges