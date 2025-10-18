from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
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
    f = open(file, "rb")
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="ignore")
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
    # May need to parallel process this somehow later if it takes too long
    # Arbitrarily set the zero-th index of special tokens to be the end-of-text token
    boundaries =  find_chunk_boundaries(input_path, num_processes, special_tokens[0].encode("utf-8"))
    
    jobs = []
    chunk_freqs = {}
    # with ProcessPoolExecutor(max_workers=num_processes) as ex:
    with ThreadPoolExecutor(max_workers=num_processes) as ex:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            jobs.append(ex.submit(_process_chunk, input_path, start, end, special_tokens))

        for j in as_completed(jobs):
            print("test: ", j)
            for k, v in j.result().items():
                chunk_freqs[k] = chunk_freqs.get(k, 0) + v
    
    freqs = {}
    for token, cnt in chunk_freqs.items():
        byte_token = token.encode("utf-8")
        atoms = tuple(bytes([b]) for b in byte_token)
        freqs[atoms] = cnt
    
    # Now that pretoken frequencies are acculumated and tuple-d,
    # We can start merging most frequent bytes
    vocab = {i : bytes([i]) for i in range(256)}
    cnt = 256
    for token in special_tokens:
        binary_token = token.encode("utf-8")
        if binary_token not in vocab.values():
            vocab[cnt] = binary_token
            cnt += 1


    merges = []
    while len(vocab) < vocab_size:
        pair_freqs = {}
        for atoms, freq in freqs.items():
            for i in range(len(atoms) - 1):
                pair_freqs[(atoms[i], atoms[i + 1])] = pair_freqs.get((atoms[i], atoms[i + 1]), 0) + freq

        byte_pair = max(pair_freqs.items(), key = lambda x: (x[1], x[0]))[0]
        merges.append(byte_pair)

        if byte_pair[0] + byte_pair[1] not in vocab.values():
            vocab[cnt] = byte_pair[0] + byte_pair[1]
            cnt += 1
        
        # Looping through freqs again; only constant factor slowdown
        new_freqs = {}
        for atoms, freq in freqs.items():
            new_atoms = []
            i = 0
            while i < len(atoms):
                if i < len(atoms) - 1 and atoms[i] == byte_pair[0] and atoms[i + 1] == byte_pair[1]:
                    new_atoms.append(atoms[i] + atoms[i + 1])
                    i += 2
                else:
                    new_atoms.append(atoms[i])
                    i += 1
            new_atoms_t = tuple(new_atoms)
            # Merging is not necessarily injective (?) Adding just to be safe
            new_freqs[new_atoms_t] = new_freqs.get(new_atoms_t, 0) + freq
        freqs = new_freqs

    return vocab, merges

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    print(os.listdir('.'))
    TrainBPE('./data/TinyStoriesV2-GPT4-valid.txt', 50257, ["<|endoftext|>"])
