"""End-to-end benchmark for WhisperModel.transcribe_batch.

Usage:
  # Basic run with a fixed batch size:
    python test_run.py --batch-size 8

  # Sweep to find the sweet spot (tries batch sizes 1,2,4,...,max-batch-size):
    python test_run.py --batch-size 64 --search-from-batch-size 1

  # Custom sweep start:
    python test_run.py --batch-size 128 --search-from-batch-size 8
"""

import argparse
import time

from faster_whisper import WhisperModel, decode_audio

parser = argparse.ArgumentParser()
parser.add_argument("--audio-path", default="data/short_heb.m4a")
parser.add_argument("--language", default="he")
parser.add_argument("--model", default="yoad/whisper-tiny-v2-ct2")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument(
    "--search-from-batch-size",
    type=int,
    default=None,
    help="When set, sweep batch sizes from this value up to --batch-size "
    "(doubling each step) and report the sweet spot.",
)
parser.add_argument(
    "--warmup",
    type=int,
    default=1,
    help="Number of warmup runs before timing (default: 1).",
)
parser.add_argument(
    "--repeats",
    type=int,
    default=3,
    help="Number of timed runs to average (default: 3).",
)
args = parser.parse_args()


def time_sequential(model, audio, n, language, repeats):
    """Run model.transcribe n times sequentially, return average wall time."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            segs, _ = model.transcribe(audio, language=language)
            list(segs)  # drain the generator
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


def time_batch(model, audio, n, language, repeats):
    """Run model.transcribe_batch with n copies, return average wall time."""
    batch = [audio] * n
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.transcribe_batch(batch, language=language)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


def sweep_batch_sizes(start, end):
    """Generate batch sizes: start, start*2, start*4, ..., end."""
    sizes = []
    s = start
    while s <= end:
        sizes.append(s)
        s *= 2
    if sizes[-1] != end:
        sizes.append(end)
    return sizes


def main():
    model = WhisperModel(args.model)
    audio = decode_audio(args.audio_path)
    duration = audio.shape[0] / model.feature_extractor.sampling_rate
    print(f"Audio: {args.audio_path} ({duration:.2f}s)")
    print(f"Model: {args.model}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats} (best-of)")

    # -- warmup --
    for _ in range(args.warmup):
        list(model.transcribe(audio, language=args.language)[0])
        model.transcribe_batch([audio], language=args.language)

    if args.search_from_batch_size is not None:
        # -- sweep mode --
        sizes = sweep_batch_sizes(args.search_from_batch_size, args.batch_size)
        print(f"\nSweeping batch sizes: {sizes}")
        print(f"{'batch':>6}  {'sequential':>11}  {'batched':>11}  {'speedup':>8}  {'per-item':>10}")
        print(f"{'size':>6}  {'(s)':>11}  {'(s)':>11}  {'':>8}  {'batch (ms)':>10}")
        print("-" * 60)

        best_speedup = 0.0
        best_size = sizes[0]
        results = []

        for n in sizes:
            t_seq = time_sequential(model, audio, n, args.language, args.repeats)
            t_bat = time_batch(model, audio, n, args.language, args.repeats)
            speedup = t_seq / t_bat if t_bat > 0 else float("inf")
            per_item_ms = (t_bat / n) * 1000

            print(f"{n:>6}  {t_seq:>11.3f}  {t_bat:>11.3f}  {speedup:>7.2f}x  {per_item_ms:>10.1f}")
            results.append((n, t_seq, t_bat, speedup, per_item_ms))

            if speedup > best_speedup:
                best_speedup = speedup
                best_size = n

        print("-" * 60)
        print(f"Sweet spot: batch_size={best_size} ({best_speedup:.2f}x speedup)")

        # find where batching becomes slower than sequential
        crossover = None
        for n, t_seq, t_bat, speedup, _ in results:
            if speedup < 1.0:
                crossover = n
                break
        if crossover:
            print(f"Crossover (batch slower than sequential): batch_size={crossover}")
        else:
            print("Batching was faster than sequential at all tested sizes.")

    else:
        # -- single batch size mode --
        n = args.batch_size

        # sequential baseline (actually run it)
        t_seq = time_sequential(model, audio, n, args.language, args.repeats)

        # show single-transcribe output once for reference
        segs, info = model.transcribe(audio, language=args.language)
        segs = list(segs)
        print(f"\n--- transcription (language={info.language}, p={info.language_probability:.2f}) ---")
        for seg in segs:
            print(f"  [{seg.start:.2f} -> {seg.end:.2f}] {seg.text}")

        # batch
        t_bat = time_batch(model, audio, n, args.language, args.repeats)
        speedup = t_seq / t_bat if t_bat > 0 else float("inf")

        print(f"\n--- benchmark (n={n}) ---")
        print(f"  sequential ({n}x transcribe): {t_seq:.3f}s")
        print(f"  batched    (transcribe_batch): {t_bat:.3f}s")
        print(f"  speedup: {speedup:.2f}x")

        # sanity check
        results = model.transcribe_batch([audio] * n, language=args.language)
        texts = [" ".join(s.text for s in segs) for segs, _ in results]
        assert len(set(texts)) == 1, "batch results differ across identical inputs"
        print("  all batch outputs identical: OK")


if __name__ == "__main__":
    main()
