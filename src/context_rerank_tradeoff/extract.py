"""Extract E1, E2 and context-profile features for the MTG-Jamendo tracks used in the study.

DRAFT for src/context_rerank_tradeoff/extract.py (review before committing).

Track sets (fixed before extraction; see the decision log, 2026-11-04):
  dev_sample  SAMPLE_SIZE dev tracks drawn with random.Random(SEED)          -> full pipeline
              (percentile reference for the features, mean vector for E2 centring)
  dev_tagged  per context, up to TAG_CAP dev tracks carrying >= 1 of its tags,
              drawn with random.Random(f"{SEED}-{context}")                 -> features + tempo
              (context profiles are the mean over exactly these tracks)
  eval        all split-0 test tracks                                        -> full pipeline
full pipeline = features + tempo + E1 + E2, all on the whole track.
E2 on the middle 30 s was rejected (top-10 overlap 6.2 < 7, pre-registered check).

One .npz per track is written atomically (tmp + rename), so an interrupted run
resumes where it stopped. Every run writes a manifest (versions, model hashes, settings).

Usage (from the repository root, after committing):
  uv run python -m context_rerank_tradeoff.extract \
      --data /Volumes/ResearchData/mtg-jamendo --models /Volumes/ResearchData/models \
      --out /Volumes/ResearchData/features
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence TensorFlow's per-model INFO lines

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import random
import subprocess
import sys
import time
from importlib import metadata, util
from pathlib import Path

import numpy as np

SEED = 20261104
SAMPLE_SIZE = 500
TAG_CAP = 300
# Draft mapping from the research plan; must be fixed on dev data before eval is looked at.
CONTEXT_TAGS = {
    "workout": {"sport", "energetic", "upbeat"},
    "relax": {"relaxing", "calm", "meditative"},
    "focus": {"background"},
}
MODEL_FILES = {
    "musicnn": "msd-musicnn-1.pb",
    "deam": "deam-msd-musicnn-2.pb",
    "acoustic": "mood_acoustic-msd-musicnn-1.pb",
    "voice": "voice_instrumental-msd-musicnn-1.pb",
    "effnet": "discogs-effnet-bs64-1.pb",
    "openl3": "openl3-music-mel128-emb512-3.pb",
}
# palonso's OpenL3 front-end has no licence notice, so it is not vendored into this
# repository; it is read from the models directory and pinned by its hash instead.
OPENL3_GIST = "openl3_gist.py"


# ---------------------------------------------------------------- track sets


def read_split(data: Path, part: str) -> set[str]:
    f = data / f"repo/data/splits/split-0/autotagging_moodtheme-{part}.tsv"
    with open(f) as fh:
        next(fh)
        return {line.split("\t")[0] for line in fh}


def read_meta(data: Path) -> dict[str, tuple[str, set[str]]]:
    """track_id -> (relative mp3 path, set of mood/theme tag names)."""
    meta = {}
    with open(data / "repo/data/autotagging_moodtheme.tsv") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\r\n").split("\t")
            meta[p[0]] = (p[3], {t.split("---", 1)[1] for t in p[5:]})
    return meta


def track_sets(data: Path) -> dict[str, list[str]]:
    meta = read_meta(data)
    dev = sorted(read_split(data, "train") | read_split(data, "validation"))
    sets = {"dev_sample": sorted(random.Random(SEED).sample(dev, SAMPLE_SIZE))}
    for c, tags in CONTEXT_TAGS.items():
        tagged = [t for t in dev if meta[t][1] & tags]
        k = min(TAG_CAP, len(tagged))
        sets[f"dev_tagged_{c}"] = sorted(random.Random(f"{SEED}-{c}").sample(tagged, k))
    sets["eval"] = sorted(read_split(data, "test"))
    return sets


def track_plan(data: Path) -> list[tuple[str, str, bool]]:
    """[(track_id, rel_path, full_pipeline)]: dev first (needed to build profiles), eval last."""
    meta = read_meta(data)
    sets = track_sets(data)
    full = set(sets["dev_sample"]) | set(sets["eval"])
    tagged = sorted({t for k, v in sets.items() if k.startswith("dev_tagged_") for t in v} - full)
    plan = [(t, meta[t][0], False) for t in tagged]
    plan += [(t, meta[t][0], True) for t in sets["dev_sample"]]
    plan += [(t, meta[t][0], True) for t in sets["eval"]]
    return plan


def audio_path(data: Path, rel: str) -> Path:
    # audio-low files are "<id>.low.mp3" while the metadata PATH says "<id>.mp3"
    return data / "audio-low" / (rel[: -len(".mp3")] + ".low.mp3")


# ---------------------------------------------------------------- models


def load_models(models: Path) -> dict:
    import essentia
    import essentia.standard as es

    essentia.log.infoActive = False
    spec = util.spec_from_file_location("openl3_gist", models / OPENL3_GIST)
    gist = util.module_from_spec(spec)
    spec.loader.exec_module(gist)
    m = {k: str(models / v) for k, v in MODEL_FILES.items()}
    return {
        "musicnn": es.TensorflowPredictMusiCNN(graphFilename=m["musicnn"], output="model/dense/BiasAdd"),
        "deam": es.TensorflowPredict2D(graphFilename=m["deam"], output="model/Identity"),
        "acoustic": es.TensorflowPredict2D(graphFilename=m["acoustic"], output="model/Softmax"),
        "voice": es.TensorflowPredict2D(graphFilename=m["voice"], output="model/Softmax"),
        "effnet": es.TensorflowPredictEffnetDiscogs(graphFilename=m["effnet"], output="PartitionedCall:1"),
        "openl3": gist.EmbeddingsOpenL3(m["openl3"]),
    }


def extract(models: dict, path: Path, full: bool) -> dict:
    import essentia.standard as es
    import librosa

    t = {}
    s = time.perf_counter()
    a16 = es.MonoLoader(filename=str(path), sampleRate=16000, resampleQuality=4)()
    mc = models["musicnn"](a16)
    av = models["deam"](mc).mean(axis=0)  # [valence, arousal]
    ac = models["acoustic"](mc).mean(axis=0)  # [acoustic, non_acoustic]
    vo = models["voice"](mc).mean(axis=0)  # [instrumental, voice]
    t["features"] = time.perf_counter() - s

    s = time.perf_counter()
    a22 = es.MonoLoader(filename=str(path), sampleRate=22050, resampleQuality=4)()
    tempo, _ = librosa.beat.beat_track(y=a22, sr=22050)
    t["tempo"] = time.perf_counter() - s

    out = {
        "valence": np.float32(av[0]),
        "arousal": np.float32(av[1]),
        "acoustic": np.float32(ac[0]),
        "voice": np.float32(vo[1]),
        "tempo": np.float32(np.atleast_1d(tempo)[0]),
        "duration_s": np.float32(len(a16) / 16000),
    }
    if full:
        s = time.perf_counter()
        out["e1"] = models["effnet"](a16).mean(axis=0).astype(np.float32)
        t["e1"] = time.perf_counter() - s
        s = time.perf_counter()
        out["e2"] = models["openl3"].compute(str(path)).mean(axis=0).astype(np.float32)
        t["e2"] = time.perf_counter() - s
    out["time"] = json.dumps(t)
    return out


# ---------------------------------------------------------------- run


def worker(args, jobs):
    models = load_models(args.models)
    total = len(jobs)
    for i, (tid, rel, full) in enumerate(jobs, 1):
        dst = args.out / "tracks" / f"{tid}.npz"
        if dst.exists():
            continue
        try:
            r = extract(models, audio_path(args.data, rel), full)
        except Exception as e:  # keep going; failures are listed and retried on rerun
            with open(args.out / "failed.tsv", "a") as fh:
                fh.write(f"{time.strftime('%F %T')}\t{tid}\t{type(e).__name__}: {e}\n")
            print(f"{time.strftime('%T')} FAILED {tid}: {e}", flush=True)
            continue
        tmp = dst.with_name(f"{tid}.tmp.npz")
        np.savez(tmp, **r)
        tmp.rename(dst)
        secs = sum(json.loads(r["time"]).values())
        print(f"{time.strftime('%T')} {i}/{total} {tid} full={int(full)} {secs:.1f}s", flush=True)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_manifest(args, plan):
    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    man = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {p: metadata.version(p) for p in ("essentia-tensorflow", "librosa", "numpy")},
        "models_sha256": {f: sha256(args.models / f) for f in [*MODEL_FILES.values(), OPENL3_GIST]},
        "settings": {"seed": SEED, "sample_size": SAMPLE_SIZE, "tag_cap": TAG_CAP,
                     "context_tags": {k: sorted(v) for k, v in CONTEXT_TAGS.items()}},
        "n_tracks": {"total": len(plan), "full": sum(f for *_, f in plan)},
        "workers": args.workers,
        "limit": args.limit,
    }
    name = f"manifest-{time.strftime('%Y%m%dT%H%M%S')}.json"
    (args.out / name).write_text(json.dumps(man, indent=2))
    (args.out / "track_sets.json").write_text(json.dumps(track_sets(args.data), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: only the first N tracks of each kind (features-only / full)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    plan = track_plan(args.data)
    if args.limit:
        plan = [j for j in plan if not j[2]][: args.limit] + [j for j in plan if j[2]][: args.limit]
    todo = [j for j in plan if not (args.out / "tracks" / f"{j[0]}.npz").exists()]
    n_full = sum(f for *_, f in plan)
    print(f"plan {len(plan)} tracks ({len(plan) - n_full} features-only, {n_full} full), "
          f"todo {len(todo)}", flush=True)
    if args.dry_run:
        return
    (args.out / "tracks").mkdir(parents=True, exist_ok=True)
    write_manifest(args, plan)
    if args.workers == 1:
        worker(args, todo)
    else:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=worker, args=(args, todo[i :: args.workers]))
                 for i in range(args.workers)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    left = sum(not (args.out / "tracks" / f"{j[0]}.npz").exists() for j in plan)
    print(f"{time.strftime('%T')} finished: {len(plan) - left}/{len(plan)} done, {left} left", flush=True)


if __name__ == "__main__":
    main()
