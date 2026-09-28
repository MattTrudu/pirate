#!/usr/bin/env python3
"""Measure the candidate-clustering workloads for presentation Figures 6 and 7."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[2]
COUNTS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
FIG6_BEAMS = [1, 2, 10, 30, 60]
FIG7_BATCHES = [60, 30, 20, 10, 2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(output):
    for key, relative in dict(TMPDIR="tmp", TMP="tmp", TEMP="tmp",
                              CUPY_CACHE_DIR="cache/cupy", CUDA_CACHE_PATH="cache/cuda",
                              XDG_CACHE_HOME="cache").items():
        path = output / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))


class Workload:
    def __init__(self, plan, geometry, cp, np, table_type):
        self.plan, self.geometry, self.cp, self.np = plan, geometry, cp, np
        self.table_type = table_type
        self.names = [field.name for field in fields(table_type)]
        self.families = cp.asnumpy(geometry.primary_tree_index_by_tree)
        self.dm_step = cp.asnumpy(geometry.dm_step_by_tree)
        self.time_step = cp.asnumpy(geometry.time_step_samples_by_tree)
        self.dm_lo = np.array([float(t.dm_min) for t in plan.trees])
        self.dm_hi = np.array([float(t.dm_max) for t in plan.trees])
        self.pair_families = [f for f in np.unique(self.families)
                              if np.count_nonzero(self.families == f) >= 2]

    def host(self, n, seed):
        """Original half-paired model, vectorized; beam draws separate from data."""
        np = self.np
        rng = np.random.default_rng(seed)
        tree = np.arange(n, dtype=np.int32) % len(self.plan.trees)
        rng.shuffle(tree)
        dm = rng.uniform(self.dm_lo[tree], self.dm_hi[tree])
        toa = rng.uniform(0, int(self.plan.nt_in), size=n)
        left_rows = np.arange(0, n - 1, 4)
        right_rows = left_rows + 1
        family = rng.choice(self.pair_families, size=len(left_rows))
        for f in self.pair_families:
            rows = np.flatnonzero(family == f)
            bank = np.flatnonzero(self.families == f)
            first = rng.integers(0, len(bank), size=len(rows))
            second = (first + rng.integers(1, len(bank), size=len(rows))) % len(bank)
            tree[left_rows[rows]] = bank[first]
            tree[right_rows[rows]] = bank[second]
        left, right = tree[left_rows], tree[right_rows]
        lo = np.maximum(self.dm_lo[left], self.dm_lo[right])
        hi = np.minimum(self.dm_hi[left], self.dm_hi[right])
        assert np.all(hi > lo)
        dm[left_rows] = rng.uniform(lo + .01 * (hi - lo), hi - .01 * (hi - lo))
        delta = rng.uniform(-.2, .2, size=len(left)) * np.minimum(self.dm_step[left], self.dm_step[right])
        dm[right_rows] = dm[left_rows] + delta
        toa[left_rows] = rng.uniform(64, int(self.plan.nt_in) - 64, size=len(left))
        toa[right_rows] = toa[left_rows] + delta * self.geometry.residual_slope_hi_samples_per_dm / 2
        beam_unit = rng.random(n)
        beam_unit[right_rows] = beam_unit[left_rows]
        idm = ((dm - self.dm_lo[tree]) / self.dm_step[tree]).astype(np.int32)
        itime = (toa / self.time_step[tree]).astype(np.int32)
        cols = dict(
            beam_id=np.zeros(n, np.int32), source_chunk_index=np.zeros(n, np.int64),
            tree=tree, idm=idm, itime=itime, snr=rng.uniform(10, 30, size=n),
            argmax_token=np.zeros(n, np.uint32), edge_flags=np.zeros(n, np.uint8),
            dm=dm, toa_sample_abs=toa, width_samp=np.full(n, 4., np.float64),
            width_ms=np.full(n, 4 * self.geometry.time_sample_ms, np.float64),
            freq_lo_MHz=np.full(n, self.geometry.full_band_freq_lo_MHz, np.float64),
            freq_hi_MHz=np.full(n, self.geometry.full_band_freq_hi_MHz, np.float64),
            primary_tree_index=self.families[tree], dm_step=self.dm_step[tree],
            time_step_samples=self.time_step[tree])
        assert np.all((dm >= self.dm_lo[tree]) & (dm < self.dm_hi[tree]))
        assert np.all((toa >= 0) & (toa < int(self.plan.nt_in)))
        assert np.all(self.families[left] == self.families[right]) and np.all(left != right)
        tolerance = 1.5 * np.maximum(self.time_step[left], self.time_step[right])
        dt = toa[right_rows] - toa[left_rows]
        slope = delta * self.geometry.residual_slope_hi_samples_per_dm
        assert np.all(abs(delta) <= 1.5 * np.maximum(self.dm_step[left], self.dm_step[right]))
        assert np.all((dt >= np.minimum(0, slope) - tolerance) & (dt <= np.maximum(0, slope) + tolerance))
        return cols, beam_unit

    def upload(self, columns):
        return self.table_type(**{k: self.cp.asarray(columns[k]) for k in self.names})

    def full_beams(self, per_beam, seed):
        pieces = []
        for beam in range(60):
            cols, _ = self.host(per_beam, seed + 1009 * beam)
            cols["beam_id"][:] = beam
            pieces.append(cols)
        return {k: self.np.concatenate([part[k] for part in pieces]) for k in self.names}

    def slices(self, full, per_beam, batch):
        return [(start * per_beam, self.table_type(**{
            k: getattr(full, k)[start * per_beam:(start + batch) * per_beam]
            for k in self.names})) for start in range(0, 60, batch)]

    def diagnostics(self, columns):
        np = self.np
        # Match the production kernel: primary family no longer partitions
        # candidates that can associate across a DM boundary.
        _, counts = np.unique(columns["beam_id"], return_counts=True)
        signature = hashlib.sha256()
        for name in self.names:
            signature.update(name.encode())
            signature.update(columns[name].tobytes())
        return dict(input_sha256=signature.hexdigest(), occupied_beams=len(np.unique(columns["beam_id"])),
                    partitions=len(counts), largest_partition=int(counts.max()))


def complete(results, batches):
    for result, (_, table) in zip(results, batches):
        if not result.complete or result.timed_out:
            raise RuntimeError("clustering timed out; incomplete work must not enter the timing figures")
        if len(result.candidates) != len(table) or len(result.members) != len(table):
            raise RuntimeError("clustering did not return every candidate exactly once")


def canonical_assignments(results, batches, n, cp, np):
    assignment = np.full(n, -1, np.int64)
    for result, (offset, _) in zip(results, batches):
        indices = cp.asnumpy(result.input_candidate_index)
        rep = cp.asnumpy(result.events.representative_candidate_index[result.candidate_event_id])
        assignment[offset + indices] = offset + indices[rep]
        members = cp.asnumpy(result.members.candidate_index)
        np.testing.assert_array_equal(np.sort(members), np.arange(len(indices)))
    assert np.all(assignment >= 0)
    return assignment


def summarize(rows, np):
    summaries = []
    keys = sorted(set((r["figure"], r["candidate_count"], r["beams_per_call"], r["calls"]) for r in rows))
    for figure, count, beams, calls in keys:
        vals = [r["wall_ms"] for r in rows if (r["figure"], r["candidate_count"], r["beams_per_call"], r["calls"])
                == (figure, count, beams, calls)]
        summaries.append(dict(figure=figure, candidate_count=count, beams_per_call=beams, calls=calls,
                              median_ms=float(np.median(vals)), p10_ms=float(np.quantile(vals, .1)),
                              p90_ms=float(np.quantile(vals, .9)), samples=len(vals)))
    return summaries


def save(output, metadata, rows, cases, np):
    payload = dict(metadata=metadata, summaries=summarize(rows, np), cases=cases, measurements=rows)
    temporary = output / "measurements.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "measurements.json")
    for name, values in [("timings.csv", rows), ("summary.csv", payload["summaries"])]:
        if values:
            with (output / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=0, help="logical CPU for the benchmark host thread")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--counts", nargs="+", type=int, default=COUNTS)
    parser.add_argument("--output-dir", type=Path, default=REPO / "docs/grouper_timing")
    args = parser.parse_args()
    if min(args.seeds, args.repeats, args.warmup) < 1 or any(n < 4 or n % 4 for n in args.counts):
        parser.error("positive seeds/repeats/warmup and counts divisible by four are required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "measurements.json").exists():
        parser.error("measurements already exist; choose a new --output-dir")
    environment(output)
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("requested logical CPU is unavailable")
    os.sched_setaffinity(0, {args.cpu})
    import cupy as cp
    import numpy as np
    from pirate_frb import DedispersionConfig, DedispersionPlan
    from pirate_frb.Clustering import ClusteringGeometry, ClusteringTolerances, GpuDecodedCandidates, cluster_candidates

    cp.cuda.Device(args.gpu).use()
    config_path = REPO / "configs/dedispersion/chord_sb2_et.yml"
    plan = DedispersionPlan(DedispersionConfig.from_yaml_string(config_path.read_text()),
                            mega_ringbuf=False, gpu_kernels=False)
    assert plan.ntrees == 10 and plan.nt_in == 2048 and plan.config.time_sample_ms == 1.
    geometry = ClusteringGeometry.from_plan(plan)
    tolerance = ClusteringTolerances(dm_tolerance_bins=1.5, time_padding_bins=1.5)
    workload = Workload(plan, geometry, cp, np, GpuDecodedCandidates)
    stream = cp.cuda.get_current_stream()

    def timed(batches):
        stream.synchronize()
        results = []
        start = time.perf_counter_ns()
        for _, table in batches:
            results.append(cluster_candidates(table, geometry, config=tolerance, timeout_ms=400))
            stream.synchronize()
        milliseconds = (time.perf_counter_ns() - start) / 1.e6
        # Validation, CPU copies, and destruction of outputs are outside timing.
        complete(results, batches)
        return milliseconds, results

    properties = cp.cuda.runtime.getDeviceProperties(args.gpu)
    seeds = [20260924 + i for i in range(args.seeds)]
    gpu_name = properties["name"]
    metadata = dict(
        date_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        gpu=gpu_name.decode() if isinstance(gpu_name, bytes) else str(gpu_name), gpu_index=args.gpu,
        cpu_affinity=sorted(os.sched_getaffinity(0)), python=platform.python_version(),
        numpy=np.__version__, cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        cuda_driver=cp.cuda.runtime.driverGetVersion(), seeds=seeds, repeats=args.repeats,
        warmup=args.warmup, global_warmup=30, timeout_ms=400,
        candidate_counts=args.counts, figure6_beams=FIG6_BEAMS, figure7_batches=FIG7_BATCHES,
        chunk_budget_ms=float(plan.nt_in) * float(plan.config.time_sample_ms),
        config_sha256=digest(config_path), script_sha256=digest(__file__),
        clustering_sha256=digest(REPO / "pirate_frb/Clustering.py"),
        partition_keys=["beam_id"],
        plan_yaml=plan.to_yaml_string(), config_yaml=plan.config.to_yaml_string(),
        timing_scope="Synchronized wall time of public cluster_candidates API, including validation, sorting, partitioning and GPU result construction. One timer surrounds each entire schedule; each sequential call is synchronized. Excludes input generation/upload/slicing, compilation, CPU validation, peakfinding, decoding and catalog output.",
        input_model="Synthetic normalized decoded table: half rows in compatible distinct-tree, same-family pairs; remainder uniform in native DM and 2048-sample time ranges. S/N uniform 10–30. Tokens and frequency fields are reporting placeholders, not a physical native decoder simulation.",
        figure6_distribution="The same non-beam candidate columns at a given seed/N; indivisible pairs and independent singles assigned uniformly to available beams. Some beams can be empty, especially at small N; actual occupancy is recorded per case.",
        figure7_distribution="Exactly N candidates in each of 60 independently seeded beams. All batching schedules slice the same GPU table before timing. Exact per-input representative assignments compared across all schedules.",
        legacy_method_sources={str(path): digest(path) for path in [
            Path('/home/mtrudu/memo/iteration2/profiling/paired_association.py'),
            Path('/home/mtrudu/memo/iteration3/profiling/loaded_association.py')] if path.exists()},
    )
    columns, units = workload.host(4800, 20260912)
    columns["beam_id"] = (units * 2).astype(np.int32)
    warm = [(0, workload.upload(columns))]
    for _ in range(30):
        _, results = timed(warm)
        del results
    del warm
    print(f"Warm-up complete on {metadata['gpu']}; starting Figure 6", flush=True)
    rows, cases = [], []
    for figure in (6, 7):
        for count in args.counts:
            for seed in seeds:
                batches_by_size, diagnostics = {}, {}
                if figure == 6:
                    columns, units = workload.host(count, seed)
                    sizes = FIG6_BEAMS
                    total = count
                    for beams in sizes:
                        columns["beam_id"] = (units * beams).astype(np.int32)
                        batches_by_size[beams] = [(0, workload.upload(columns))]
                        diagnostics[beams] = workload.diagnostics(columns)
                else:
                    columns = workload.full_beams(count, seed)
                    assert np.bincount(columns["beam_id"]).tolist() == [count] * 60
                    full = workload.upload(columns)
                    sizes = FIG7_BATCHES
                    total = 60 * count
                    diag = workload.diagnostics(columns)
                    for beams in sizes:
                        batches_by_size[beams] = workload.slices(full, count, beams)
                        diagnostics[beams] = diag
                reference = None
                expected_events = {}
                for beams in sizes:
                    batches = batches_by_size[beams]
                    for _ in range(args.warmup):
                        _, results = timed(batches)
                    assignment = canonical_assignments(results, batches, total, cp, np)
                    if figure == 7:
                        if reference is None:
                            reference = assignment
                        else:
                            np.testing.assert_array_equal(assignment, reference)
                    expected_events[beams] = sum(len(r.events) for r in results)
                    cases.append(dict(figure=figure, candidate_count=count, total_candidates=total,
                        beams_per_call=beams, calls=len(batches), seed=seed,
                        events=expected_events[beams], members=total, complete=True,
                        batching_assignment_equal=True if figure == 7 else None, **diagnostics[beams]))
                    del results
                order_rng = np.random.default_rng(seed + count + figure)
                for repeat in range(args.repeats):
                    # Interleave schedules to limit bias from temperature/clock drift.
                    for beams in order_rng.permutation(sizes).tolist():
                        batches = batches_by_size[beams]
                        ms, results = timed(batches)
                        assert sum(len(r.events) for r in results) == expected_events[beams]
                        if repeat == args.repeats - 1:
                            assignment = canonical_assignments(results, batches, total, cp, np)
                            if figure == 7:
                                np.testing.assert_array_equal(assignment, reference)
                        rows.append(dict(figure=figure, candidate_count=count, total_candidates=total,
                            beams_per_call=beams, calls=len(batches), seed=seed, repeat=repeat,
                            wall_ms=ms, events=expected_events[beams], members=total, complete=True))
                        del results
                save(output, metadata, rows, cases, np)
                print(f"Figure {figure}: N={count}, seed={seed} complete", flush=True)
                del batches_by_size, columns
                if figure == 7:
                    del full
            summary = [s for s in summarize(rows, np) if s['figure'] == figure and s['candidate_count'] == count]
            print("Medians (ms): " + ", ".join(f"{s['beams_per_call']} beams: {s['median_ms']:.3f}" for s in summary), flush=True)
    metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["complete"] = True
    save(output, metadata, rows, cases, np)
    print(f"Saved {len(rows)} complete schedule timings to {output}", flush=True)


if __name__ == "__main__":
    main()
