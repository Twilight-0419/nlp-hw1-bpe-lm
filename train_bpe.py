import json
import threading
import time
from pathlib import Path

import psutil

import basics.BPE as B
from tests.common import gpt2_bytes_to_unicode

STATS = {}


def sample_peak_memory(stop):
    me = psutil.Process()
    peak = 0
    while not stop.is_set():
        total = me.memory_info().rss
        for child in me.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        peak = max(peak, total)
        time.sleep(0.5)
    STATS["peak_bytes"] = peak


if __name__ == '__main__':
    original = B.pretokenize_parallel

    def timed(*args, **kwargs):
        t = time.time()
        result = original(*args, **kwargs)
        STATS["pretok_seconds"] = time.time() - t
        print(f"pretok done: {STATS['pretok_seconds']:.1f}s", flush=True)
        return result

    B.pretokenize_parallel = timed

    stop = threading.Event()
    thread = threading.Thread(target=sample_peak_memory, args=(stop,), daemon=True)
    thread.start()

    start_time = time.time()
    vocab, merges = B.bpe_tokenizer(
        'data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>']
    )
    end_time = time.time()
    total_time = end_time - start_time

    stop.set()
    thread.join(timeout=2)

    longest = max(vocab.values(), key=len)
    print(f"vocab={len(vocab)} merges={len(merges)}")
    print(f"total={total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"pretok={STATS['pretok_seconds']:.1f}s "
          f"merge={total_time - STATS['pretok_seconds']:.1f}s")
    print(f"peak={STATS['peak_bytes'] / 2 ** 30:.2f} GB")
    print(f"longest={longest!r} ({len(longest)} bytes)")

    b2u = gpt2_bytes_to_unicode()
    out = Path("output")
    out.mkdir(exist_ok=True)
    with open(out / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"".join(b2u[b] for b in tok): i for i, tok in vocab.items()},
                  f, ensure_ascii=False)
    with open(out / "merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write("".join(b2u[x] for x in a) + " " +
                    "".join(b2u[x] for x in b) + "\n")
