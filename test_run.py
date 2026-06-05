"""End-to-end smoke test for WhisperModel.transcribe_batch."""

import argparse
import time

from faster_whisper import WhisperModel, decode_audio

parser = argparse.ArgumentParser()
parser.add_argument("--audio-path", default="data/short_heb.m4a")
parser.add_argument("--language", default="he")
parser.add_argument("--model", default="yoad/whisper-tiny-v2-ct2")
parser.add_argument("--batch-size", type=int, default=4)
args = parser.parse_args()


def main():
    model = WhisperModel(args.model)
    audio = decode_audio(args.audio_path)
    duration = audio.shape[0] / model.feature_extractor.sampling_rate
    print(f"Audio duration: {duration:.2f}s")

    # -- single transcribe (baseline) --
    t0 = time.perf_counter()
    segments_gen, info = model.transcribe(audio, language=args.language)
    segments_single = list(segments_gen)
    t_single = time.perf_counter() - t0

    print(f"\n--- single transcribe ---")
    print(f"  language: {info.language} (p={info.language_probability:.2f})")
    for seg in segments_single:
        print(f"  [{seg.start:.2f} -> {seg.end:.2f}] {seg.text}")
    print(f"  time: {t_single:.3f}s")

    # -- batch transcribe (4x same audio) --
    batch = [audio] * args.batch_size

    t0 = time.perf_counter()
    results = model.transcribe_batch(batch, language=args.language)
    t_batch = time.perf_counter() - t0

    print(f"\n--- batch transcribe ({args.batch_size}x) ---")
    for i, (segments, info) in enumerate(results):
        texts = " | ".join(seg.text.strip() for seg in segments)
        print(f"  [{i}] ({info.duration:.2f}s) {texts}")
    print(f"  time: {t_batch:.3f}s")

    # -- compare --
    print(f"\n--- summary ---")
    print(f"   single x{args.batch_size}: {t_single * args.batch_size:.3f}s (projected)")
    print(f"   batch  x{args.batch_size}: {t_batch:.3f}s")
    if t_single * args.batch_size > 0:
        print(f"   speedup: {t_single * args.batch_size / t_batch:.1f}x")

    # sanity: all batch results should produce the same text
    batch_texts = [
        " ".join(seg.text for seg in segments) for segments, _ in results
    ]
    assert len(set(batch_texts)) == 1, "batch results differ across identical inputs"
    print("\n  all batch outputs identical: OK")


if __name__ == "__main__":
    main()
