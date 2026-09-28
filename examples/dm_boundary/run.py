#!/usr/bin/env python3
"""Inject one native FRB at a primary-tree boundary and verify its grouping."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

# The same command works from a source checkout and an installed examples tree.
REPO = Path(__file__).resolve().parents[2]
if (REPO / "pirate_frb").is_dir():
    sys.path.insert(0, str(REPO))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def candidate_key(row):
    return tuple(row[name] for name in
                 ("beam_id", "tree", "source_chunk_index", "idm", "itime"))


def prepare(recipe_path, output):
    """Resolve the standard recipe and check its boundary without a GPU."""
    import yaml
    import pirate_frb
    from pirate_frb import DedispersionConfig, DedispersionPlan
    from pirate_frb.LivePipeline import load_live_recipe

    recipe, bundle = load_live_recipe(recipe_path)
    if len(bundle["beam_ids"]) != 1 or len(recipe["bursts"]) != 1:
        raise ValueError("this example requires one beam and one injected burst")
    config = DedispersionConfig.from_yaml_string(bundle["dedispersion_config_yaml"])
    plan = DedispersionPlan(config, mega_ringbuf=False, gpu_kernels=False)
    primary = [(i, tree) for i, tree in enumerate(plan.trees)
               if tree.early_trigger_level == 0]
    primary.sort(key=lambda pair: pair[1].primary_tree_index)
    if len(primary) < 2:
        raise ValueError("the search plan needs at least two primary trees")
    (lower_id, lower), (upper_id, upper) = primary[:2]
    boundary = float(lower.dm_max)
    if boundary != float(upper.dm_min):
        raise ValueError("the first two primary trees do not share a DM boundary")
    truth = recipe["bursts"][0]
    if not math.isclose(truth["dm"], boundary, rel_tol=0, abs_tol=1.e-9):
        raise ValueError(f"recipe DM={truth['dm']:.12f}, but the resolved boundary is "
                         f"{boundary:.12f}; update the recipe after changing the search plan")
    if truth["reference_frequency_MHz"] != float(config.zone_freq_edges[0]):
        raise ValueError("the recipe TOA must be referenced to the band's lowest frequency")
    geometry = [dict(tree=i, primary_tree_index=int(tree.primary_tree_index),
                     early_trigger_level=int(tree.early_trigger_level),
                     dm_min=float(tree.dm_min), dm_max=float(tree.dm_max),
                     ndm_out=int(tree.ndm_out)) for i, tree in enumerate(plan.trees)]
    package = Path(pirate_frb.__file__).resolve().parent
    sources = {name: package / name for name in
               ("Clustering.py", "BowtiePeakfinding.py", "GpuArgmaxDecoder.py",
                "GrouperPipeline.py", "TriggerCatalog.py", "OfflineDedisperser.py", "LivePipeline.py", "Observation.py")}
    sources.update(example=Path(__file__), recipe=recipe_path,
                   metadata=recipe_path.parent / recipe["metadata"],
                   dedispersion=recipe_path.parent / recipe["dedispersion"])
    manifest = dict(recipe_source=str(recipe_path.resolve()), truth=truth,
                    boundary_dm=boundary, boundary_tree_ids=[lower_id, upper_id],
                    boundary_idm=[int(lower.ndm_out) - 1, 0],
                    time_sample_ms=bundle["time_sample_ms"],
                    samples_per_chunk=bundle["samples_per_chunk"],
                    nchunks=bundle["nchunks"], noise_seed=recipe["observation"]["noise_seed"],
                    tree_geometry=geometry, plan_yaml=plan.to_yaml_string(),
                    python=sys.version, python_executable=sys.executable,
                    source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for name, path in sources.items()})
    # Keep a portable resolved recipe alongside the exact config and CPU plan.
    resolved = dict(recipe, metadata="metadata.yml", dedispersion="dedispersion.yml")
    for name, value in (("metadata.yml", bundle["metadata_yaml"]),
                        ("dedispersion.yml", bundle["dedispersion_config_yaml"]),
                        ("grouper.yml", bundle["grouper_config_yaml"]),
                        ("plan.yml", manifest["plan_yaml"]),
                        ("observation.yml", yaml.safe_dump(resolved, sort_keys=False))):
        (output / name).write_text(value)
    write_json(output / "manifest.json", manifest)
    print(f"Effective cadence: {bundle['time_sample_ms']:.4f} ms", flush=True)
    for index, tree in ((lower_id, lower), (upper_id, upper)):
        print(f"Full-band tree {index} (primary {tree.primary_tree_index}): "
              f"DM {tree.dm_min:.9f} to {tree.dm_max:.9f}", flush=True)
    print(f"Injected DM: {boundary:.12f} pc cm^-3; "
          f"one burst, {bundle['nchunks']} native chunks.", flush=True)
    return recipe, bundle, manifest


def run(output, recipe, bundle, manifest, gpu):
    import cupy as cp
    import ksgpu
    import numpy as np
    import yaml
    from pirate_frb import DedispersionConfig
    from pirate_frb.ArgmaxMetadata import ARGMAX_ENCODING
    from pirate_frb.Clustering import GpuDecodedCandidates, cluster_candidates
    from pirate_frb.GrouperConfig import GrouperConfig
    from pirate_frb.GrouperPipeline import CatalogRecorder, GrouperSetup
    from pirate_frb.Observation import make_pulse
    from pirate_frb.OfflineDedisperser import OfflineDedisperser
    from pirate_frb.core import AssembledFrame, XEngineMetadata

    started = last_update = time.monotonic()
    cp.cuda.Device(gpu).use()
    ksgpu.seed_default_rng(manifest["noise_seed"])
    metadata = XEngineMetadata.from_yaml_string(bundle["metadata_yaml"])
    config = DedispersionConfig.from_yaml_string(bundle["dedispersion_config_yaml"])
    dedisperser = OfflineDedisperser(config, cuda_device_id=gpu)
    pulse = make_pulse(manifest["truth"], metadata)
    nt, dt_ms = bundle["samples_per_chunk"], bundle["time_sample_ms"]
    if pulse.it_start < 0 or pulse.it_end > bundle["nchunks"] * nt:
        raise ValueError("burst is clipped: extend the duration or move its arrival time")
    observed_inputs = []

    def record_group_inputs(candidates, geometry, **kwargs):
        # The reporting wrapper never inserts, removes, or edits a candidate.
        candidates = GpuDecodedCandidates.from_decoder_result(candidates, geometry)
        columns = {field.name: cp.asnumpy(getattr(candidates, field.name))
                   for field in fields(candidates)}
        rows = [{name: values[i].item() for name, values in columns.items()}
                for i in range(len(candidates))]
        observed_inputs.append(rows)
        return cluster_candidates(candidates, geometry, **kwargs)

    processor = recorder = None
    for chunk in range(bundle["nchunks"]):
        frame = AssembledFrame.make_uninitialized(
            metadata, ntime=nt, beam_id=bundle["beam_ids"][0], time_chunk_index=chunk)
        active = [pulse] if pulse.it_start < (chunk + 1) * nt and pulse.it_end > chunk * nt else []
        frame.randomize_many(normalize=True, gaussian=True, pulses=active, dt_sp=chunk * nt)
        with dedisperser.dedisperse(frame) as maps:
            if processor is None:
                runtime_plan = yaml.safe_load(dedisperser.plan.to_yaml_string())
                runtime_plan.pop("mega_ringbuf", None)
                if runtime_plan != yaml.safe_load(manifest["plan_yaml"]):
                    raise RuntimeError("the native producer plan differs from the resolved CPU plan")
                setup = GrouperSetup(dedisperser.plan, list(dedisperser.dd.Dcores),
                                    GrouperConfig.from_mapping(recipe["grouper"]),
                                    cuda_device_id=gpu)
                recorder = CatalogRecorder(setup, config_yaml=dedisperser.config.to_yaml_string(),
                                           plan_yaml=dedisperser.plan.to_yaml_string(),
                                           argmax_encoding=ARGMAX_ENCODING)
                beams = tuple(bundle["beam_ids"])
                processor = setup.processor(beams, 0, lambda window: recorder.consume(beams, window))
                processor.group_function = record_group_inputs
            processor.process_chunk([cp.asarray(value) for value in maps.out_max],
                                    [cp.asarray(value) for value in maps.out_argmax], chunk)
            cp.cuda.get_current_stream().synchronize()
        if time.monotonic() - last_update >= 15:
            print(f"Processed chunk {chunk + 1}/{bundle['nchunks']}", flush=True)
            last_update = time.monotonic()
    processor.finish(physical_end=True)
    recorder.write(output / "events.asdf", [processor], complete=True)

    events, members = [], []
    for batch in recorder.batches:
        for table, destination in ((batch.events, events), (batch.members, members)):
            for i in range(len(table["event_id"])):
                row = {name: values[i].item() for name, values in table.items()}
                row["toa_seconds"] = row["toa_sample_abs"] * dt_ms * .001
                destination.append(row)
    unique_inputs = {candidate_key(row): row for rows in observed_inputs for row in rows}
    member_keys = [candidate_key(row) for row in members]
    conservation = len(set(member_keys)) == len(member_keys) and set(unique_inputs) == set(member_keys)
    no_timeouts = not any(window["timed_out"] for window in recorder.windows)
    decode_parity = True
    for row in members:
        integers = dedisperser.plan.decode_argmax(
            row["argmax_token"], row["tree"], dedisperser.dd.Dcores[row["tree"]],
            row["idm"], row["itime"])
        lo, hi, dm, sample, width = dedisperser.plan.decode_argmax2(row["tree"], *integers)
        expected = [lo, hi, dm, sample + row["source_chunk_index"] * nt, width * dt_ms]
        actual = [row[name] for name in
                  ("freq_lo_MHz", "freq_hi_MHz", "dm", "toa_sample_abs", "width_ms")]
        decode_parity &= bool(np.allclose(expected, actual, rtol=1.e-10, atol=1.e-7))

    truth = manifest["truth"]
    wanted_cells = set(zip(manifest["boundary_tree_ids"], manifest["boundary_idm"]))
    boundary_groups, representative_ok = [], True
    for event in events:
        rows = [row for row in members if row["event_id"] == event["event_id"]]
        representatives = [row for row in rows
                           if row["candidate_id"] == event["representative_candidate_id"]]
        representative_ok &= len(rows) == event["member_count"] and len(representatives) == 1
        if representatives:
            representative = representatives[0]
            representative_ok &= event["snr"] == max(row["snr"] for row in rows)
            representative_ok &= all(event[name] == representative[name] for name in
                                     ("beam_id", "tree", "primary_tree_index", "idm", "itime",
                                      "source_chunk_index", "snr", "dm", "toa_sample_abs",
                                      "width_samp", "width_ms", "freq_lo_MHz", "freq_hi_MHz",
                                      "argmax_token", "edge_flags"))
        cells = {(row["tree"], row["idm"]) for row in rows}
        near_pulse = all(
            row["beam_id"] == truth["beam_id"]
            and abs(row["dm"] - truth["dm"]) <= 1.5 * row["dm_step"]
            and abs(row["toa_seconds"] - truth["toa_seconds"]) * 1000
            <= 4 * truth["width_ms"] + row["width_ms"] for row in rows)
        if wanted_cells <= cells and near_pulse:
            boundary_groups.append(dict(event=event, members=rows))
    tree_info = {row["tree"]: row for row in manifest["tree_geometry"]}
    early_events = [row for row in events if tree_info[row["tree"]]["early_trigger_level"] > 0]
    early_delivery = any(
        early["primary_tree_index"] == tree_info[manifest["boundary_tree_ids"][1]]["primary_tree_index"]
        and early["source_chunk_index"] < min(row["source_chunk_index"] for row in group["members"])
        and early["grouping_window_id"] < group["event"]["grouping_window_id"]
        for early in early_events for group in boundary_groups)
    checks = dict(boundary_association_exercised=bool(boundary_groups),
                  all_candidates_emitted_once=conservation, no_timeouts=no_timeouts,
                  native_decode_parity=decode_parity, representative_fields_preserved=representative_ok,
                  early_report_emitted_before_boundary_event=early_delivery)
    gpu_name = cp.cuda.runtime.getDeviceProperties(gpu)["name"]
    summary = dict(boundary_dm=manifest["boundary_dm"], time_sample_ms=dt_ms,
                   gpu_name=gpu_name.decode() if isinstance(gpu_name, bytes) else str(gpu_name),
                   cuda_device_id=gpu, elapsed_seconds=time.monotonic() - started,
                   unique_peak_candidates=len(unique_inputs), events=len(events), members=len(members),
                   peak_candidates_by_tree=dict(Counter(row["tree"] for row in unique_inputs.values())),
                   boundary_groups=boundary_groups, early_events=early_events, checks=checks,
                   success=all(checks.values()))
    write_json(output / "clustering_inputs.json", observed_inputs)
    write_json(output / "catalog_rows.json", dict(events=events, members=members, windows=recorder.windows))
    write_json(output / "result.json", summary)
    print(f"\n{len(unique_inputs)} native peak candidates -> {len(events)} events.", flush=True)
    for group in boundary_groups:
        event = group["event"]
        print(f"Event {event['event_id']}: {event['member_count']} members "
              f"from both sides of the primary DM boundary:", flush=True)
        for row in group["members"]:
            print(f"  tree={row['tree']} primary={row['primary_tree_index']} idm={row['idm']} "
                  f"DM={row['dm']:.9f} S/N={row['snr']:.4f} "
                  f"TOA={row['toa_seconds']:.9f} s", flush=True)
    print(f"Separate early-trigger events: {len(early_events)}; "
          f"early delivery verified: {early_delivery}.", flush=True)
    print(f"Checks: {checks}\nResults: {output}", flush=True)
    if not summary["success"]:
        raise RuntimeError("native boundary example failed; inspect result.json and catalog_rows.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dm-boundary-output"),
                        help="new output directory; an existing directory is never overwritten")
    parser.add_argument("--recipe", type=Path, default=Path(__file__).with_name("observation.yml"))
    parser.add_argument("--gpu", type=int, default=0, help="logical CUDA device (default: 0)")
    parser.add_argument("--prepare-only", action="store_true", help="resolve and check the CPU plan only")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Avoid persistent compilation caches alongside the compact example outputs.
    os.environ["CUPY_CACHE_IN_MEMORY"] = "1"
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    recipe, bundle, manifest = prepare(args.recipe, output)
    if args.prepare_only:
        print(f"CPU preparation passed. Resolved files: {output}", flush=True)
    else:
        run(output, recipe, bundle, manifest, args.gpu)


if __name__ == "__main__":
    main()
