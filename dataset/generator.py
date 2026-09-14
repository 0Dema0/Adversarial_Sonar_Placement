import time
import datetime
import traceback
import sys

import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor
import threading
import os
from pathlib import Path
import json

from dataset.sample import sample_one
from dataset.utils import ETA, save_dataset, stack_samples
import dataset.static_cache as sc

_STATIC = None


def worker_init(cache_subdir: str, slow_threshold: float | None = None):
    """Worker initializer that mmap-loads precomputed arrays and builds a light StaticFeatures.

    Expects `cache_subdir` to contain `visibility_map.npy`, `base_contributions.npy`, and `meta.json`.
    """
    global _STATIC
    start = time.time()
    try:
        from environment.map import RawMap, ObstacleMap
        from environment.features import StaticFeatures
        sub = Path(cache_subdir)
        # record slow-sample threshold in worker process
        global _SLOW_SAMPLE_THRESHOLD
        _SLOW_SAMPLE_THRESHOLD = float(slow_threshold) if slow_threshold is not None else None
        print(f"{datetime.datetime.now().isoformat()} worker_init(pid={os.getpid()}): loading memmaps from {sub} slow_threshold={_SLOW_SAMPLE_THRESHOLD}", flush=True)
        vis = np.load(sub / "visibility_map.npy", mmap_mode="r")
        base = np.load(sub / "base_contributions.npy", mmap_mode="r")
        meta = json.load(open(sub / "meta.json"))

        height = int(meta.get("height"))
        width = int(meta.get("width"))
        obs = meta.get("obstacle_indices")

        raw_map = RawMap(width=width, height=height)
        obstacle_map = ObstacleMap(raw_map, obstacle_indices=obs)

        # pass precomputed arrays into StaticFeatures to avoid recomputation
        _STATIC = StaticFeatures(raw_map=raw_map, obstacle_map=obstacle_map, visibility_map=vis, base_contributions=base)
        dt = time.time() - start
        print(f"{datetime.datetime.now().isoformat()} worker_init(pid={os.getpid()}): loaded _STATIC in {dt:.3f}s", flush=True)
    except Exception:
        dt = time.time() - start
        print(f"{datetime.datetime.now().isoformat()} worker_init(pid={os.getpid()}): FAILED after {dt:.3f}s", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        # fallback: leave _STATIC None and let worker build from args
        _STATIC = None


def worker(args):
    global _STATIC

    slow_thresh = globals().get("_SLOW_SAMPLE_THRESHOLD", None)

    # Batch of samples
    if isinstance(args, (list, tuple)) and args and isinstance(args[0], tuple):
        res = []

        for a in args:
            (
                sonar_number,
                unknown_sonar_number,
                goal_index,
                single_value,
                start_index,
                forbidden_radius,
                best_sonar_split,
                sonar_placement_policy,
                starting_sonar_placement,
                first_time,
            ) = a

            t0 = time.time()

            sample = sample_one(
                _STATIC,
                sonar_number=sonar_number,
                sonar_indices=None,
                unknown_sonar_number=unknown_sonar_number,
                goal_index=goal_index,
                single_value=single_value,
                start_index=start_index,
                forbidden_radius=forbidden_radius,
                best_sonar_split=best_sonar_split,
                sonar_placement_policy=sonar_placement_policy,
                starting_sonar_placement=starting_sonar_placement,
                first_time=first_time,
            )

            dt = time.time() - t0

            if slow_thresh is not None and dt >= float(slow_thresh):
                print(
                    f"{datetime.datetime.now().isoformat()} "
                    f"worker(pid={os.getpid()}): "
                    f"SLOW_SAMPLE dt={dt:.3f}s args={a}",
                    flush=True,
                )

            res.append(sample)

        return res

    # Single sample
    (
        sonar_number,
        unknown_sonar_number,
        goal_index,
        single_value,
        start_index,
        forbidden_radius,
        best_sonar_split,
        sonar_placement_policy,
        starting_sonar_placement,
        first_time,
    ) = args

    t0 = time.time()

    sample = sample_one(
        _STATIC,
        sonar_number=sonar_number,
        sonar_indices=None,
        unknown_sonar_number=unknown_sonar_number,
        goal_index=goal_index,
        single_value=single_value,
        start_index=start_index,
        forbidden_radius=forbidden_radius,
        best_sonar_split=best_sonar_split,
        sonar_placement_policy=sonar_placement_policy,
        starting_sonar_placement=starting_sonar_placement,
        first_time=first_time,
    )

    dt = time.time() - t0

    if slow_thresh is not None and dt >= float(slow_thresh):
        print(
            f"{datetime.datetime.now().isoformat()} "
            f"worker(pid={os.getpid()}): "
            f"SLOW_SAMPLE dt={dt:.3f}s args={args}",
            flush=True,
        )

    return sample

def sampler(args):
    for i in range(args.samples):
        yield (
            args.sonar_number,
            args.unknown_sonar_number,
            args.goal_index,
            args.single_value,
            args.start_index,
            args.forbidden_radius,
            args.best_sonar_split,
            args.sonar_placement_policy,
            args.starting_sonar_placement,
            i == 0,
        )

def generate_dataset(
    static,
    sampler,
    total,
    out,
    show_eta=True,
    workers: int | None = None,
    batch_size: int | None = None,
    slow_threshold: float | None = None,
    stream_output: bool = False,
):
    eta = ETA() if show_eta else None
    # If streaming, we won't accumulate samples in memory; we'll write memmaps to disk
    results = [] if not stream_output else None
    memmaps = None
    stream_dir = None

    # Ensure a cached precompute exists and pass its path to workers so they mmap-load it
    # Expect `static` to be a StaticFeatures instance or None; if it's an instance we derive cache params
    try:
        height = int(static.raw_map.height)
        width = int(static.raw_map.width)
        obstacle_indices = None if getattr(static, "obstacles", None) is None else list(static.obstacles)
        detection_fn = getattr(static, "detection_probability", None)
    except Exception:
        # fallback to defaults: 20x20 grid
        height, width, obstacle_indices, detection_fn = 20, 20, None, None

    # build or ensure cache exists (will return a StaticFeatures but we only need the cache files)
    det_name = getattr(detection_fn, "__name__", repr(detection_fn))
    print(f"{datetime.datetime.now().isoformat()} main(pid={os.getpid()}): ensuring cache height={height} width={width} obs_count={(0 if obstacle_indices is None else len(obstacle_indices))} det={det_name}", flush=True)
    t0 = time.time()
    sc.build_static(height=height, width=width, obstacle_indices=obstacle_indices, detection_probability=(detection_fn if detection_fn is not None else None))
    dt_cache = time.time() - t0
    print(f"{datetime.datetime.now().isoformat()} main: build_static took {dt_cache:.3f}s", flush=True)

    cache_root = Path("data/precomp")
    cache_subdir = sc._cache_subdir(cache_root, height, width, obstacle_indices, detection_fn if detection_fn is not None else (lambda x: None))

    # Batch multiple samples per worker task to amortize scheduling/pickling costs
    if batch_size is None:
        batch_size = 8

    def _batched(iterable, bsize):
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) >= bsize:
                yield batch
                batch = []
        if batch:
            yield batch

    # Tune worker count and chunksize to avoid excessive prefetch and long
    # startup latency on Windows when `total` is large.
    cpu_workers = max(1, os.cpu_count() // 2)
    if workers is None:
        # don't spawn more workers than there are batches
        num_workers = min(cpu_workers, max(1, total // batch_size))
    else:
        num_workers = int(max(1, min(int(workers), cpu_workers)))

    main_start = time.time()
    print(f"{datetime.datetime.now().isoformat()} main: about to start pool workers={num_workers} batch_size={batch_size}", flush=True)
    # If streaming and an existing parts directory exists, try to resume
    memmaps = None
    stream_dir = None
    processed = 0
    META_WRITE_INTERVAL = 60.0
    last_meta_write = time.time()

    if stream_output:
        stream_dir = Path(out).parent / (Path(out).stem + "_parts")
        meta_path = stream_dir / "meta.json"
        if stream_dir.exists() and meta_path.exists():
            try:
                m = json.load(open(meta_path, "r"))
                # only resume if the total matches; otherwise start fresh
                if int(m.get("total", -1)) == int(total):
                    processed = int(m.get("processed", 0))
                    memmaps = {}
                    for key, kmeta in m.get("keys", {}).items():
                        dtype = np.dtype(kmeta["dtype"])
                        shape = tuple(kmeta["shape"])
                        # open existing memmap for read+write
                        mp = np.lib.format.open_memmap(stream_dir / f"{key}.npy", mode="r+", dtype=dtype, shape=shape)
                        memmaps[key] = mp
                    last_meta_write = time.time()
                    print(f"{datetime.datetime.now().isoformat()} main: resuming stream from {stream_dir} processed={processed}", flush=True)
                else:
                    print(f"{datetime.datetime.now().isoformat()} main: found existing parts at {stream_dir} but total mismatch (meta.total={m.get('total')} requested={total}); will overwrite when first sample arrives", flush=True)
            except Exception:
                print(f"{datetime.datetime.now().isoformat()} main: failed reading existing meta.json at {meta_path}; will recreate parts on first sample", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=worker_init,
        initargs=(str(cache_subdir), slow_threshold),
    ) as pool:

        # reasonable chunksize per worker, clamped to avoid huge prefetch
        chunksize = max(1, total // (num_workers * 4))
        # cap chunksize to keep time-to-first-result small even for large `total`
        CHUNKSIZE_CAP = 2
        chunksize = min(chunksize, CHUNKSIZE_CAP)
        print(f"{datetime.datetime.now().isoformat()} main: using chunksize={chunksize}", flush=True)
        # `processed` may have been set from an existing parts meta; don't overwrite it
        task_iter = _batched(itertools.islice(sampler(), processed, None), batch_size)
        start_time = time.time()
        last_time = start_time
        # background monitor to print status even when main loop is waiting on workers
        MONITOR_INTERVAL = 30.0
        stop_monitor = threading.Event()

        def _monitor():
            while not stop_monitor.wait(MONITOR_INTERVAL):
                try:
                    elapsed = time.time() - main_start
                    if eta:
                        eta_str = eta.update(processed, total)
                    else:
                        eta_str = "?"
                    print(f"{datetime.datetime.now().isoformat()} monitor: {processed}/{total} | ETA {eta_str} | elapsed {elapsed:.1f}s", flush=True)
                except Exception:
                    pass

        monitor_thread = threading.Thread(target=_monitor, daemon=True)
        monitor_thread.start()

        last_time = start_time
        # ETA printing: ensure we print at least every N seconds even if processed%100 != 0
        ETA_PRINT_INTERVAL = 30.0
        last_eta_print = start_time
        first_batch = True
        try:
            for batch_result in pool.map(worker, task_iter, chunksize=chunksize):
                # batch_result may be a list of samples or a single sample
                n = 1
                if isinstance(batch_result, list):
                    n = len(batch_result)

                # initialize streaming memmaps on first sample when requested
                if stream_output and memmaps is None:
                    # derive parts directory from output path
                    stream_dir = Path(out).parent / (Path(out).stem + "_parts")
                    stream_dir.mkdir(parents=True, exist_ok=True)

                    # pick representative sample
                    first_sample = batch_result[0] if isinstance(batch_result, list) else batch_result

                    # create memmaps for each key
                    memmaps = {}
                    meta = {}
                    for key, arr in first_sample.items():
                        a = np.asarray(arr)
                        mem_shape = (total,) + a.shape
                        mp = np.lib.format.open_memmap(stream_dir / f"{key}.npy", mode="w+", dtype=a.dtype, shape=mem_shape)
                        memmaps[key] = mp
                        meta[key] = {"shape": list(mem_shape), "dtype": str(a.dtype)}

                    # write meta (include processed counter)
                    import json as _json

                    meta_path = stream_dir / "meta.json"
                    meta_payload = {"total": int(total), "keys": meta, "processed": 0, "created": datetime.datetime.now().isoformat()}
                    with open(meta_path, "w") as _f:
                        _json.dump(meta_payload, _f)

                    # track meta write time so we can periodically update processed count
                    last_meta_write = time.time()
                    META_WRITE_INTERVAL = 60.0

                if stream_output:
                    # write samples directly into memmaps
                    if isinstance(batch_result, list):
                        for i, sample in enumerate(batch_result):
                            idx = processed + i
                            for key, mp in memmaps.items():
                                mp[idx] = sample[key]
                    else:
                        idx = processed
                        for key, mp in memmaps.items():
                            mp[idx] = batch_result[key]

                    processed += n
                    # periodically update meta.json with latest processed count
                    try:
                        now_meta = time.time()
                        if (processed % 100 == 0) or (now_meta - last_meta_write >= META_WRITE_INTERVAL):
                            m = json.load(open(meta_path, "r"))
                            m["processed"] = int(processed)
                            m["updated"] = datetime.datetime.now().isoformat()
                            json.dump(m, open(meta_path, "w"))
                            last_meta_write = now_meta
                    except Exception:
                        pass

                else:
                    # collect in-memory
                    if isinstance(batch_result, list):
                        results.extend(batch_result)
                        processed += len(batch_result)
                    else:
                        results.append(batch_result)
                        processed += 1

                now = time.time()
                batch_dt = now - last_time
                total_dt = now - start_time
                last_time = now

                if first_batch and processed > 0:
                    print(f"{datetime.datetime.now().isoformat()} main: first result after {total_dt:.3f}s", flush=True)
                    first_batch = False

                if eta:
                    nowt = time.time()
                    if processed % 100 == 0 or (nowt - last_eta_print) >= ETA_PRINT_INTERVAL:
                        print(f"{datetime.datetime.now().isoformat()} {processed}/{total} | ETA {eta.update(processed, total)} | batch {batch_dt:.3f}s | total {total_dt:.1f}s", flush=True)
                        last_eta_print = nowt

        finally:
            # stop background monitor
            try:
                stop_monitor.set()
                monitor_thread.join(timeout=5)
            except Exception:
                pass

        total_run = time.time() - main_start
        print(f"{datetime.datetime.now().isoformat()} main: completed processing {processed} samples in {total_run:.3f}s", flush=True)

    # finalize streaming output (if used) or build/save dataset from memory
    if stream_output:
        if memmaps is not None:
            for mp in memmaps.values():
                try:
                    mp.flush()
                except Exception:
                    pass

            # update meta.json with final processed count
            try:
                meta_path = stream_dir / "meta.json"
                m = json.load(open(meta_path, "r"))
                m["processed"] = int(processed)
                json.dump(m, open(meta_path, "w"))
            except Exception:
                pass

        print(
            f"{datetime.datetime.now().isoformat()} main: streaming parts saved in {stream_dir} (processed={processed}).",
            flush=True,
        )

        print(
            "To merge parts into a single .npz: use dataset/check_tmp_parts.py or run a small merge script.",
            flush=True,
        )

        return None

    dataset = stack_samples(results)
    save_dataset(out, dataset)

    return dataset