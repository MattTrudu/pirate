"""Synthetic scientific checks for representative-based GPU grouping."""

import ast
from dataclasses import fields, replace
import inspect
import json
from pathlib import Path
import textwrap

import numpy as np

from ..Clustering import (
    GpuDecodedCandidates,
    ClusteringTolerances,
    ClusteringGeometry,
    compatible_with_representative,
    cluster_candidates,
    _cluster_candidates_serial_oracle,
)
from ..BowtiePeakfinding import EdgeFlag


def _geometry(cp):
    """Return four trees spanning two primary-tree families."""

    return ClusteringGeometry(
        ntrees=4,
        primary_tree_index_by_tree=cp.asarray([0, 0, 1, 0], cp.int32),
        dm_step_by_tree=cp.asarray([1.0, 2.0, 1.0, 4.0], cp.float64),
        time_step_samples_by_tree=cp.asarray(
            [1.0, 2.0, 1.0, 4.0], cp.float64
        ),
        reference_freq_MHz=300.0,
        full_band_freq_lo_MHz=300.0,
        full_band_freq_hi_MHz=1500.0,
        time_sample_ms=1.0,
        residual_slope_lo_samples_per_dm=0.0,
        residual_slope_hi_samples_per_dm=2.0,
    )


def _candidates(cp, geometry, rows):
    """Build the production GPU table from compact test dictionaries."""

    n = len(rows)

    def values(name, default):
        return [row.get(name, default(i, row)) for i, row in enumerate(rows)]

    tree_host = np.asarray(
        values("tree", lambda i, row: i % geometry.ntrees), dtype=np.int32
    )
    families = cp.asnumpy(geometry.primary_tree_index_by_tree)[tree_host]
    dm_steps = cp.asnumpy(geometry.dm_step_by_tree)[tree_host]
    time_steps = cp.asnumpy(geometry.time_step_samples_by_tree)[tree_host]
    return GpuDecodedCandidates(
        beam_id=cp.asarray(values("beam_id", lambda i, row: 7), cp.int32),
        source_chunk_index=cp.asarray(
            values("source_chunk_index", lambda i, row: 0), cp.int64
        ),
        tree=cp.asarray(tree_host, cp.int32),
        idm=cp.asarray(values("idm", lambda i, row: i), cp.int32),
        itime=cp.asarray(values("itime", lambda i, row: i), cp.int32),
        snr=cp.asarray(values("snr", lambda i, row: 20.0 - i), cp.float64),
        argmax_token=cp.asarray(
            values("argmax_token", lambda i, row: i), cp.uint32
        ),
        edge_flags=cp.asarray(
            values("edge_flags", lambda i, row: 0), cp.uint8
        ),
        dm=cp.asarray(values("dm", lambda i, row: 100.0), cp.float64),
        toa_sample_abs=cp.asarray(
            values("toa", lambda i, row: 1000.0), cp.float64
        ),
        width_samp=cp.asarray(
            values("width_samp", lambda i, row: 2.0), cp.float64
        ),
        width_ms=cp.asarray(
            values("width_ms", lambda i, row: 2.0), cp.float64
        ),
        freq_lo_MHz=cp.asarray(
            values("freq_lo_MHz", lambda i, row: 300.0), cp.float64
        ),
        freq_hi_MHz=cp.asarray(
            values("freq_hi_MHz", lambda i, row: 1500.0), cp.float64
        ),
        primary_tree_index=cp.asarray(families, cp.int32),
        dm_step=cp.asarray(dm_steps, cp.float64),
        time_step_samples=cp.asarray(time_steps, cp.float64),
    )


def test_pair_compatibility_physical_boundaries(cuda_device_id=0):
    """Only beam and physical distances restrict tree-independent association."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        candidates = _candidates(cp, geometry, [
            {"tree": 0, "dm": 100.0, "toa": 1000.0},
            {"tree": 1, "dm": 101.0, "toa": 1001.0},
            {"tree": 0, "dm": 100.0, "toa": 1000.0},
            {"tree": 1, "beam_id": 8, "dm": 100.0, "toa": 1000.0},
            {"tree": 2, "dm": 100.0, "toa": 1000.0},
            {"tree": 3, "dm": 110.1, "toa": 1000.0},
            {"tree": 3, "dm": 100.0, "toa": 1010.1},
        ])
        match = compatible_with_representative(
            candidates,
            0,
            cp.arange(1, len(candidates), dtype=cp.int64),
            geometry,
            ClusteringTolerances(dm_tolerance_bins=1.5, time_padding_bins=1.0),
        )
        assert np.array_equal(
            cp.asnumpy(match),
            np.asarray([True, True, False, True, False, False]),
        )


def test_greedy_grouping_all_compatible_rows_and_beam_isolation(cuda_device_id=0):
    """All compatible rows join across trees/families, while beams stay apart."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        candidates = _candidates(cp, geometry, [
            {"tree": 0, "snr": 30.0, "dm": 100.0, "toa": 1000.0},
            {"tree": 1, "snr": 25.0, "dm": 100.5, "toa": 1001.0},
            {"tree": 1, "snr": 24.0, "dm": 100.4, "toa": 1000.5},
            {"tree": 3, "snr": 23.0, "dm": 100.2, "toa": 999.5},
            {
                "tree": 1, "snr": 22.0, "beam_id": 8,
                "dm": 100.0, "toa": 1000.0,
            },
            {"tree": 2, "snr": 21.0, "dm": 100.0, "toa": 1000.0},
            {"tree": 3, "snr": 20.0, "dm": 100.0, "toa": 1100.0},
        ])
        grouped = cluster_candidates(candidates, geometry)
        event_id = cp.asnumpy(grouped.candidate_event_id)
        assert event_id[0] == event_id[1] == event_id[2] == event_id[3]
        assert event_id[4] != event_id[0]
        assert event_id[5] == event_id[0]
        assert event_id[6] != event_id[0]
        first_members = cp.asnumpy(
            grouped.members.candidate_index[
                grouped.members.event_id == event_id[0]
            ]
        )
        assert set(first_members.tolist()) == {0, 1, 2, 3, 5}
        assert len(grouped.events) == 3
        assert int(cp.asnumpy(grouped.events.representative_candidate_index)[0]) == 0
        _assert_grouping_tables_equal(
            cp, grouped, _cluster_candidates_serial_oracle(candidates, geometry)
        )


def test_grouping_across_chunks_uses_absolute_toa(cuda_device_id=0):
    """Chunk labels do not block association when absolute arrivals agree."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        candidates = _candidates(cp, geometry, [
            {
                "tree": 0, "source_chunk_index": 0, "itime": 1023,
                "snr": 20.0, "dm": 42.0, "toa": 1023.75,
            },
            {
                "tree": 1, "source_chunk_index": 1, "itime": 0,
                "snr": 18.0, "dm": 42.1, "toa": 1024.0,
            },
        ])
        grouped = cluster_candidates(candidates, geometry)
        assert len(grouped.events) == 1
        assert len(grouped.members) == 2
        assert np.array_equal(
            cp.asnumpy(grouped.candidate_event_id), np.asarray([0, 0])
        )


def test_grouping_preserves_representative_and_member_startup_flags(
        cuda_device_id=0):
    """Event flags come from the seed; member flags remain per candidate."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        startup = int(EdgeFlag.STARTUP_INCOMPLETE)
        candidates = _candidates(cp, geometry, [
            {
                "tree": 0, "snr": 30.0, "dm": 100.0, "toa": 1000.0,
                "edge_flags": 0,
            },
            {
                "tree": 1, "snr": 25.0, "dm": 100.2, "toa": 1000.2,
                "edge_flags": startup,
            },
            {
                "tree": 0, "beam_id": 8, "snr": 20.0,
                "dm": 100.0, "toa": 1000.0, "edge_flags": startup,
            },
        ])
        grouped = cluster_candidates(candidates, geometry)

        assert cp.asnumpy(grouped.events.edge_flags).tolist() == [0, startup]
        first_members = cp.asnumpy(
            grouped.members.candidate_index[
                grouped.members.event_id == cp.int64(0)
            ]
        )
        assert set(first_members.tolist()) == {0, 1}
        assert cp.asnumpy(
            grouped.candidates.edge_flags[first_members]
        ).tolist() == [0, startup]
        assert grouped.events.edge_flags.dtype == cp.uint8
        assert grouped.candidates.edge_flags.dtype == cp.uint8


def test_grouping_empty_and_gpu_residency(cuda_device_id=0):
    """Empty input is valid and production grouping has no array host copy."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        empty = _candidates(cp, geometry, [])
        grouped = cluster_candidates(empty, geometry)
        assert len(grouped.events) == 0
        assert len(grouped.members) == 0
        assert int(grouped.candidate_event_id.size) == 0
        assert grouped.complete is True
        assert grouped.timed_out is False
        assert grouped.input_candidate_index.dtype == cp.int64
        assert int(grouped.input_candidate_index.size) == 0

        source = textwrap.dedent(inspect.getsource(cluster_candidates))
        assert "asnumpy" not in source and ".get(" not in source
        assert "N * N" not in source and "N, N" not in source
        tree = ast.parse(source)
        assert not any(
            isinstance(node, (ast.For, ast.AsyncFor, ast.While))
            for node in ast.walk(tree)
        )
        assert source.count("_raw_grouping_kernel(") == 1
        module_source = inspect.getsource(inspect.getmodule(cluster_candidates))
        assert "%globaltimer" in module_source
        assert "atomicCAS(\n                launch_deadline_ns" in module_source
        assert "test_timeout_after_work_tiles" in module_source
        assert "provisional_representative" in module_source


def _timeout_geometry(cp):
    """Return two compatible trees in one persistent-kernel partition."""

    return ClusteringGeometry(
        ntrees=2,
        primary_tree_index_by_tree=cp.asarray([0, 0], cp.int32),
        dm_step_by_tree=cp.asarray([1.0, 1.0], cp.float64),
        time_step_samples_by_tree=cp.asarray([1.0, 1.0], cp.float64),
        reference_freq_MHz=300.0,
        full_band_freq_lo_MHz=300.0,
        full_band_freq_hi_MHz=1500.0,
        time_sample_ms=1.0,
        residual_slope_lo_samples_per_dm=0.0,
        residual_slope_hi_samples_per_dm=0.0,
    )


def _assert_grouping_tables_equal(cp, left, right):
    """Assert bitwise equality of all GPU-resident grouping table columns."""

    assert cp.array_equal(
        left.candidate_event_id, right.candidate_event_id
    ).item()
    assert cp.array_equal(
        left.input_candidate_index, right.input_candidate_index
    ).item()
    for table_name in ("candidates", "events", "members"):
        left_table = getattr(left, table_name)
        right_table = getattr(right, table_name)
        for field in fields(left_table):
            assert cp.array_equal(
                getattr(left_table, field.name),
                getattr(right_table, field.name),
                equal_nan=True,
            ).item(), (table_name, field.name)


def test_timeout_discards_in_progress_event_and_rebases_indices(
        cuda_device_id=0):
    """Only the first complete event survives cancellation during event two."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _timeout_geometry(cp)
        # Event zero scans two 256-thread tiles and commits input rows 0/299.
        # The deterministic GPU hook expires on tile three, midway through the
        # next representative event. Middle rows are outside the seed's DM
        # tolerance, rather than depending on the obsolete same-tree veto.
        rows = [
            {
                "tree": 0,
                "snr": 1000.0 - i,
                "dm": float(10 * i),
                "toa": 0.0,
            }
            for i in range(299)
        ] + [{"tree": 1, "snr": 1.0, "dm": 0.0, "toa": 0.0}]
        # The canceled second group actually stages a compatible member before
        # cancellation. Neither its seed nor that member may leak to output.
        rows[2]["dm"] = rows[1]["dm"]
        candidates = _candidates(cp, geometry, rows)
        partial = cluster_candidates(
            candidates,
            geometry,
            _test_timeout_after_work_tiles=3,
        )

        assert partial.timed_out is True
        assert partial.complete is False
        assert cp.asnumpy(partial.input_candidate_index).tolist() == [0, 299]
        assert cp.asnumpy(partial.candidate_event_id).tolist() == [0, 0]
        assert cp.asnumpy(
            partial.events.representative_candidate_index
        ).tolist() == [0]
        assert cp.asnumpy(partial.events.member_count).tolist() == [2]
        assert cp.asnumpy(partial.members.event_id).tolist() == [0, 0]
        assert cp.asnumpy(partial.members.candidate_index).tolist() == [0, 1]
        assert cp.asnumpy(
            partial.members.is_representative
        ).tolist() == [True, False]
        assert len(partial.candidates) == len(partial.members) == 2
        assert int(cp.sum(partial.events.member_count).item()) == 2

        # The deterministic hook and seed ordering produce the same
        # clean partial tables on every execution.
        repeated = cluster_candidates(
            candidates,
            geometry,
            _test_timeout_after_work_tiles=3,
        )
        assert repeated.timed_out is True
        _assert_grouping_tables_equal(cp, partial, repeated)


def test_timeout_before_first_commit_returns_typed_empty_partial(
        cuda_device_id=0):
    """Cancellation during event zero never leaks its provisional seed."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _timeout_geometry(cp)
        rows = [
            {"tree": i % 2, "snr": 500.0 - i, "dm": 0.0, "toa": 0.0}
            for i in range(300)
        ]
        partial = cluster_candidates(
            _candidates(cp, geometry, rows),
            geometry,
            _test_timeout_after_work_tiles=1,
        )
        assert partial.timed_out is True
        assert partial.complete is False
        assert len(partial.candidates) == 0
        assert len(partial.events) == 0
        assert len(partial.members) == 0
        assert partial.candidate_event_id.dtype == cp.int64
        assert partial.input_candidate_index.dtype == cp.int64
        assert partial.events.member_count.dtype == cp.int32
        assert partial.members.is_representative.dtype == cp.bool_


def test_timeout_after_final_commit_remains_explicit(cuda_device_id=0):
    """A deadline observed after an indivisible final commit stays a timeout."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _timeout_geometry(cp)
        candidates = _candidates(cp, geometry, [
            {"tree": 0, "snr": 20.0, "dm": 0.0, "toa": 0.0},
            {"tree": 1, "snr": 10.0, "dm": 0.0, "toa": 0.0},
        ])
        partial = cluster_candidates(
            candidates,
            geometry,
            _test_timeout_after_committed_events=1,
        )
        assert partial.timed_out is True
        assert partial.complete is False
        assert cp.asnumpy(partial.input_candidate_index).tolist() == [0, 1]
        assert cp.asnumpy(partial.candidate_event_id).tolist() == [0, 0]
        assert cp.asnumpy(partial.events.member_count).tolist() == [2]
        assert cp.asnumpy(partial.members.candidate_index).tolist() == [0, 1]


def test_timeout_defaults_validation_and_complete_input_mapping(
        cuda_device_id=0):
    """Zero disables timeout; statuses and original-row mapping are explicit."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        assert ClusteringTolerances().dm_tolerance_bins == 1.5
        assert ClusteringTolerances().time_padding_bins == 1.5
        geometry = _timeout_geometry(cp)
        candidates = _candidates(cp, geometry, [
            {"tree": 0, "snr": 20.0},
            {"tree": 1, "snr": 10.0},
        ])
        complete = cluster_candidates(candidates, geometry, timeout_ms=0)
        assert complete.complete is True
        assert complete.timed_out is False
        assert cp.asnumpy(complete.input_candidate_index).tolist() == [0, 1]

        invalid = (
            (True, TypeError),
            (-1, ValueError),
            (np.nan, ValueError),
            (np.inf, ValueError),
            (1.0e308, ValueError),
            ("1", TypeError),
        )
        for value, error_type in invalid:
            try:
                cluster_candidates(candidates, geometry, timeout_ms=value)
            except error_type:
                pass
            else:
                raise AssertionError(
                    f"timeout_ms={value!r} did not raise {error_type.__name__}"
                )


def test_grouping_rejects_noncurrent_owning_device(cuda_device_id=0):
    """Kernel/workspace allocation cannot silently mix two CUDA devices."""

    import cupy as cp

    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count < 2:
        return
    owner = int(cuda_device_id) % device_count
    other = (owner + 1) % device_count
    with cp.cuda.Device(owner):
        geometry = _timeout_geometry(cp)
        candidates = _candidates(cp, geometry, [
            {"tree": 0, "snr": 20.0},
            {"tree": 1, "snr": 10.0},
        ])
    with cp.cuda.Device(other):
        try:
            cluster_candidates(candidates, geometry)
        except ValueError as exc:
            assert "current CUDA device" in str(exc)
        else:
            raise AssertionError("cross-device grouping did not fail explicitly")


def test_saved_native_trial_000317_merges_same_tree_plateau(cuda_device_id=0):
    """Unmodified real singleton reports become two members of one event."""

    import cupy as cp

    from ..pirate_pybind11 import DedispersionConfig, DedispersionPlan

    fixture = json.loads((
        Path(__file__).parent / "data" / "clustering_trial_000317.json"
    ).read_text())
    config = DedispersionConfig.from_yaml_string(fixture["config_yaml"])
    plan = DedispersionPlan.from_yaml_string(config, fixture["plan_yaml"])
    tolerance = ClusteringTolerances(**fixture["tolerances"])
    original = fixture["events"]
    assert len(original) == 2
    assert [event["member_count"] for event in original] == [1, 1]
    assert [event["idm"] for event in original] == [1383, 1384]
    assert original[0]["snr"] == original[1]["snr"] == 59.65625
    assert float(plan.config.time_sample_ms) == 0.9984

    with cp.cuda.Device(cuda_device_id):
        geometry = ClusteringGeometry.from_plan(plan)
        # Reverse input order as well: equal-S/N choice must follow provenance,
        # not whichever row happens to arrive first in the test table.
        for records in (original, list(reversed(original))):
            rows = [dict(row, toa=row["toa_sample_abs"]) for row in records]
            candidates = _candidates(cp, geometry, rows)
            representative = next(
                i for i, record in enumerate(records) if record["idm"] == 1383
            )
            other = 1 - representative
            assert compatible_with_representative(
                candidates, representative, cp.asarray([other]), geometry,
                tolerance,
            ).item()
            grouped = cluster_candidates(
                candidates, geometry, config=tolerance
            )
            reference = _cluster_candidates_serial_oracle(
                candidates, geometry, config=tolerance
            )
            _assert_grouping_tables_equal(cp, grouped, reference)
            assert grouped.complete and not grouped.timed_out
            assert len(grouped.events) == 1
            assert len(grouped.candidates) == len(grouped.members) == 2
            assert grouped.events.member_count.tolist() == [2]
            assert grouped.candidate_event_id.tolist() == [0, 0]
            assert grouped.events.representative_candidate_index.tolist() == [
                representative
            ]
            assert grouped.members.candidate_index.tolist() == [
                representative, other
            ]
            assert grouped.members.is_representative.tolist() == [True, False]
            for field in fields(grouped.events):
                if hasattr(candidates, field.name):
                    assert getattr(grouped.events, field.name).item() == (
                        getattr(candidates, field.name)[representative].item()
                    ), field.name
            # Every saved decoded input field survives unchanged, including
            # full-precision DM/TOA, width, subband, flags and argmax token.
            for field in fields(candidates):
                if field.name in records[0]:
                    assert getattr(grouped.candidates, field.name).tolist() == [
                        row[field.name] for row in records
                    ], field.name


def test_chord_primary_dm_boundary_and_sample_override(cuda_device_id=0):
    """Adjacent full-band primary trees can merge at their actual DM border."""

    import cupy as cp

    from ..pirate_pybind11 import DedispersionConfig, DedispersionPlan

    filename = str(Path(__file__).parent / "data" / "chord_sb2_et.yml")
    boundaries = []
    with cp.cuda.Device(cuda_device_id):
        for sample_ms in (1.0, 0.9984):
            config = DedispersionConfig.from_yaml(filename)
            config.time_sample_ms = sample_ms
            plan = DedispersionPlan(config)
            geometry = ClusteringGeometry.from_plan(plan)
            full_band = {
                int(tree.primary_tree_index): index
                for index, tree in enumerate(plan.trees)
                if int(tree.early_trigger_level) == 0
            }
            low_index, high_index = full_band[0], full_band[1]
            low, high = plan.trees[low_index], plan.trees[high_index]
            boundary = float(low.dm_max)
            assert boundary == float(high.dm_min)
            boundaries.append(boundary)
            low_step = float(geometry.dm_step_by_tree[low_index].item())
            high_step = float(geometry.dm_step_by_tree[high_index].item())
            padding = 1.5 * max(
                float(geometry.time_step_samples_by_tree[low_index].item()),
                float(geometry.time_step_samples_by_tree[high_index].item()),
            )
            # Coarse-bin centres on either side; actual producer grid spacing,
            # not guessed physical DM or an altered tolerance.
            lower_dm = boundary - 0.5 * low_step
            upper_dm = boundary + 0.5 * high_step
            dm_difference = upper_dm - lower_dm
            residual_midpoint = 0.5 * (
                geometry.residual_slope_lo_samples_per_dm
                + geometry.residual_slope_hi_samples_per_dm
            ) * dm_difference
            common = {
                "source_chunk_index": 64,
                "itime": 20,
                "beam_id": 0,
            }
            candidates = _candidates(cp, geometry, [
                dict(
                    common, tree=low_index, snr=40.0,
                    idm=int(low.ndm_out) - 1, dm=lower_dm, toa=100000.0,
                ),
                dict(
                    common, tree=high_index, snr=30.0,
                    idm=0, dm=upper_dm, toa=100000.0 + residual_midpoint,
                ),
                dict(
                    common, tree=high_index, snr=20.0,
                    idm=0, dm=upper_dm,
                    toa=100000.0
                    + geometry.residual_slope_hi_samples_per_dm * dm_difference
                    + padding + 1.0,
                ),
            ])
            assert candidates.primary_tree_index.tolist() == [0, 1, 1]
            assert compatible_with_representative(
                candidates, 0, cp.asarray([1, 2]), geometry
            ).tolist() == [True, False]
            grouped = cluster_candidates(candidates, geometry)
            _assert_grouping_tables_equal(
                cp, grouped,
                _cluster_candidates_serial_oracle(candidates, geometry),
            )
            assert grouped.candidate_event_id.tolist() == [0, 0, 1]
            assert grouped.events.member_count.tolist() == [2, 1]
            assert grouped.events.representative_candidate_index.tolist() == [
                0, 2
            ]
        # The effective 0.9984 ms sampling changes the physical DM border;
        # this guards against silently using the nominal YAML value.
        assert np.isclose(boundaries[1] / boundaries[0], 0.9984, rtol=1e-14)


def test_same_tree_many_members_and_distinct_nearby_bursts(cuda_device_id=0):
    """Deduplicate each nearby burst without fusing distinct temporal peaks."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _timeout_geometry(cp)
        candidates = _candidates(cp, geometry, [
            {"tree": 0, "snr": 30.0, "dm": 100.0, "toa": 1000.0},
            {"tree": 0, "snr": 28.0, "dm": 100.2, "toa": 1000.2},
            {"tree": 0, "snr": 26.0, "dm": 99.9, "toa": 999.8},
            {"tree": 0, "snr": 24.0, "dm": 100.1, "toa": 1000.5},
            {"tree": 0, "snr": 22.0, "dm": 100.0, "toa": 1003.1},
            {"tree": 1, "snr": 20.0, "dm": 100.2, "toa": 1003.3},
            {"tree": 0, "snr": 18.0, "dm": 103.0, "toa": 1000.0},
        ])
        grouped = cluster_candidates(candidates, geometry)
        _assert_grouping_tables_equal(
            cp, grouped, _cluster_candidates_serial_oracle(candidates, geometry)
        )
        assert grouped.candidate_event_id.tolist() == [0, 0, 0, 0, 1, 1, 2]
        assert grouped.events.member_count.tolist() == [4, 2, 1]
        assert grouped.events.representative_candidate_index.tolist() == [0, 4, 6]


def test_grouping_does_not_grow_through_nonrepresentative_members(
        cuda_device_id=0):
    """B matches A and C, but C stays separate when it cannot match seed A."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _timeout_geometry(cp)
        candidates = _candidates(cp, geometry, [
            {"tree": 0, "snr": 30.0, "dm": 0.0, "toa": 0.0},
            {"tree": 0, "snr": 20.0, "dm": 0.0, "toa": 1.0},
            {"tree": 1, "snr": 10.0, "dm": 0.0, "toa": 2.0},
        ])
        assert compatible_with_representative(
            candidates, 1, cp.asarray([2]), geometry
        ).item()
        grouped = cluster_candidates(candidates, geometry)
        _assert_grouping_tables_equal(
            cp, grouped, _cluster_candidates_serial_oracle(candidates, geometry)
        )
        assert grouped.candidate_event_id.tolist() == [0, 0, 1]
        assert grouped.events.member_count.tolist() == [2, 1]


def test_grouping_inclusive_dm_and_time_limits(cuda_device_id=0):
    """Exact limits join; the nearest representable value outside does not."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _timeout_geometry(cp)
        outside = np.nextafter(1.5, np.inf)
        rows = [
            {"tree": 0, "snr": 100.0, "dm": 0.0, "toa": 0.0},
            {"tree": 0, "dm": 1.5, "toa": 0.0},
            {"tree": 1, "dm": -1.5, "toa": 0.0},
            {"tree": 0, "dm": outside, "toa": 0.0},
            {"tree": 1, "dm": -outside, "toa": 0.0},
            {"tree": 0, "dm": 0.0, "toa": 1.5},
            {"tree": 1, "dm": 0.0, "toa": -1.5},
            {"tree": 0, "dm": 0.0, "toa": outside},
            {"tree": 1, "dm": 0.0, "toa": -outside},
        ]
        candidates = _candidates(cp, geometry, rows)
        assert compatible_with_representative(
            candidates, 0, cp.arange(1, len(rows)), geometry
        ).tolist() == [True, True, False, False, True, True, False, False]
        grouped = cluster_candidates(candidates, geometry)
        _assert_grouping_tables_equal(
            cp, grouped, _cluster_candidates_serial_oracle(candidates, geometry)
        )
        first = grouped.members.candidate_index[grouped.members.event_id == 0]
        assert set(first.tolist()) == {0, 1, 2, 5, 6}


def test_large_same_tree_group_commits_all_tiles(cuda_device_id=0):
    """A same-tree group spanning multiple CUDA tiles retains every member."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _timeout_geometry(cp)
        candidates = _candidates(cp, geometry, [
            {"tree": 0, "snr": 1000.0 - i, "dm": 100.0, "toa": 1000.0}
            for i in range(600)
        ])
        grouped = cluster_candidates(candidates, geometry)
        _assert_grouping_tables_equal(
            cp, grouped, _cluster_candidates_serial_oracle(candidates, geometry)
        )
        assert grouped.complete and not grouped.timed_out
        assert grouped.events.member_count.tolist() == [600]
        assert grouped.members.candidate_index.tolist() == list(range(600))
        assert grouped.candidate_event_id.tolist() == [0] * 600


def test_seeded_multibeam_grouping_matches_oracle_for_hot_dtypes(
        cuda_device_id=0):
    """Planted same-/cross-tree groups agree for float32 and mixed predicates."""

    import cupy as cp

    rng = np.random.default_rng(20260928)
    rows = []
    for group_index in range(8):
        # Coincident copies on separate beams must stay separate. Distinct
        # physical groups are far enough apart in DM to avoid accidental links.
        physical_group = group_index // 2
        dm_center = 100.0 + 20.0 * physical_group
        toa_center = 1000.0 + 100.0 * physical_group
        trees = rng.permutation([0, 0, 0, 1, 1, 2, 2, 2, 3, 3])
        for member in range(10):
            # Binary fractions stay exactly representable in each dtype;
            # margins remain away from tolerance boundaries, whose inclusive
            # behavior is tested separately.
            dm_jitter = float(rng.integers(-2, 3)) / 8.0 if member else 0.0
            time_jitter = float(rng.integers(-2, 3)) / 8.0 if member else 0.0
            rows.append({
                "beam_id": 7 + group_index % 2,
                "tree": int(trees[member]),
                "snr": 1000.0 - group_index if member == 0 else float(
                    rng.integers(20, 40)
                ),
                "dm": dm_center + dm_jitter,
                "toa": toa_center + dm_jitter + time_jitter,
                "source_chunk_index": 4 + int(rng.integers(0, 2)),
                "idm": group_index * 100 + member,
                "itime": int(rng.integers(0, 64)),
                "argmax_token": group_index * 10 + member,
                "edge_flags": int(rng.integers(0, 2)),
            })
    permutation = rng.permutation(len(rows))
    shuffled_rows = [rows[int(index)] for index in permutation]
    # The leading member of each planted group is unambiguously the loudest.
    expected_representatives = [
        int(np.flatnonzero(permutation == group * 10)[0])
        for group in range(8)
    ]
    hot_fields = ("dm", "toa_sample_abs", "dm_step", "time_step_samples")
    variants = (
        (np.float32, np.float32, np.float32, np.float32),
        (np.float32, np.float64, np.float32, np.float64),
        (np.float64, np.float32, np.float64, np.float32),
    )
    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        original = _candidates(cp, geometry, shuffled_rows)
        for dtypes in variants:
            candidates = replace(original, **{
                name: getattr(original, name).astype(dtype)
                for name, dtype in zip(hot_fields, dtypes)
            })
            grouped = cluster_candidates(candidates, geometry)
            oracle = _cluster_candidates_serial_oracle(candidates, geometry)
            _assert_grouping_tables_equal(cp, grouped, oracle)
            assert grouped.complete and not grouped.timed_out
            assert len(grouped.events) == 8
            assert grouped.events.member_count.tolist() == [10] * 8
            assert grouped.events.representative_candidate_index.tolist() == (
                expected_representatives
            )
            assert len(grouped.candidates) == len(grouped.members) == 80
            assert sorted(grouped.members.candidate_index.tolist()) == list(
                range(80)
            )
            assert int(cp.sum(grouped.members.is_representative).item()) == 8
            assert grouped.candidate_event_id.tolist() == [
                int(index) // 10 for index in permutation
            ]
