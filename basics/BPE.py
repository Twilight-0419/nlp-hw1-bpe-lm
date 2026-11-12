import regex as re
from collections import defaultdict,Counter
from multiprocessing import Pool
from basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
def pretokenize_chunk(task):
    path, start, end, special_pattern = task
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    counter = Counter()
    for part in re.split(special_pattern, text):
        counter.update(m.group() for m in re.finditer(PAT, part))
    return counter
    
def pretokenize_parallel(path, special_tokens, num_procs):
    special_pattern = "(?:" + "|".join(
        re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)
    ) + ")"
    with open(path, "rb") as f:
        b = find_chunk_boundaries(f, num_procs, special_tokens[0].encode("utf-8"))
    tasks = [(path, s, e, special_pattern) for s, e in zip(b[:-1], b[1:])]

    total = Counter()
    with Pool(num_procs) as pool:
        for c in pool.imap_unordered(pretokenize_chunk, tasks):
            total.update(c)
    
    return {w.encode("utf-8"): n for w, n in total.items()}



def bpe_tokenizer(input_path, vocab_size, special_tokens):
    vocab={}
    for i in range(256):
        vocab[i]=bytes([i])
    next_word_id = 256
    for special_token in special_tokens:
        vocab[next_word_id]=special_token.encode('UTF-8')
        next_word_id+=1

    merges=[]

    word_counts=pretokenize_parallel(input_path,special_tokens,4)
    words=list(word_counts)
    counts=[word_counts[w] for w in words]
    splits=[tuple(bytes([b]) for b in w) for w in words]

    pair_counts=defaultdict(int)
    pair_to_words=defaultdict(set)

    def add_pairs(wid):
        split=splits[wid]
        c=counts[wid]
        for i in range(len(split)-1):
            pair=(split[i],split[i+1])
            pair_counts[pair]+=c
            pair_to_words[pair].add(wid)

    def remove_pairs(wid):
        split=splits[wid]
        c=counts[wid]
        for i in range(len(split)-1):
            pair=(split[i],split[i+1])
            pair_counts[pair]-=c
            if pair_counts[pair]==0:
                del pair_counts[pair]
                del pair_to_words[pair]
            else:
                pair_to_words[pair].discard(wid)

    for wid in range(len(words)):
        add_pairs(wid)

    while len(vocab) < vocab_size and pair_counts:
        merge = max(pair_counts, key=lambda p: (pair_counts[p], p))

        new_merge=merge[0]+merge[1]

        for wid in list(pair_to_words[merge]):
            remove_pairs(wid)

            old_split=splits[wid]
            new_split=[]
            i=0
            while i < len(old_split):
                if i+1 < len(old_split) and old_split[i]==merge[0] and old_split[i+1]==merge[1]:
                    new_split.append(new_merge)
                    i+=2
                else:
                    new_split.append(old_split[i])
                    i+=1
            splits[wid]=new_split

            add_pairs(wid)

        merges.append(merge)
        vocab[next_word_id]=new_merge
        next_word_id+=1

    return vocab , merges