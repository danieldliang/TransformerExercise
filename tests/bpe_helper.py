# Enables multiprocessing
import regex as re
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

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