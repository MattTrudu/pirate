# One FRB at the boundary between two primary trees

This example injects one Gaussian FRB into native intensity data, runs PIRATE
dedispersion and bowtie peakfinding, and checks that peaks from two neighboring
primary trees become members of the same event. No candidates are fabricated
or inserted after peakfinding.

The search uses `configs/dedispersion/chord_sb2_et.yml` and the X-engine metadata.
The effective cadence is **0.9984 ms**, so the first primary DM boundary is
**1478.537835445747 pc cm^-3**. The script derives the boundary from the resolved
CPU plan and checks the recipe against it before generating data. The nominal
1 ms value in the search YAML is overridden by the metadata, as in the live
pipeline.

| Parameter | Value |
| --- | --- |
| Beam | 0 |
| Injected DM | First primary-tree boundary, 1478.537835445747 pc cm^-3 |
| Injected S/N | 100 |
| Intrinsic Gaussian sigma | 5 ms |
| Emission band | 300–1500 MHz |
| Arrival time | 74.406582784 s at 300 MHz, from observation start |
| Observation | 41 chunks / 83.8336512 s |
| Noise seed | 20260928 |
| Peakfinding | S/N threshold 10, `dm_reach=8`, `waist_bins=1` |
| Grouping | Halo 2, DM/time tolerances 1.5, no timeout |

## Run and inspect the evidence

Use the environment described in the [installation instructions](../../notes/install.md),
with native CHORD kernels and an NVIDIA GPU. Run from the repository root:

```bash
python -B -m examples.dm_boundary.run --output /tmp/pirate-dm-boundary --gpu 0
```

The output directory must be new. This streams the native frames through the
offline dedisperser and the shared `GrouperPipeline` processor, keeping the
intensity data and maps in memory. Allow about one to two minutes on an A40;
compilation and the installed environment can change that time. This is a
correctness example, not a performance benchmark.

Expected result with the supplied recipe:

| Post-peakfinding candidate | Primary family | DM cell | Grouping result |
| --- | --- | --- | --- |
| Full-band tree 0 | 0 | Last cell, `idm=4095` | One event with both full-band candidates |
| Full-band tree 2 | 1 | First cell, `idm=0` | Member of the same event as tree 0 |
| Early-trigger tree 1 | 1 | First cell, `idm=0` | Separate event sent in an earlier window |

Bowtie peakfinding runs separately in each tree. A pulse at the shared DM
boundary can therefore leave one surviving maximum on each side. Clustering
compares their decoded DM and time and associates them in the same window.

Thus the reference result is **three peak candidates and two events**, with
**two members in the event at the DM boundary**. Tree 1 appears between trees
0 and 2 because its early-trigger tree is part of primary family 1. Tree
indices and primary-family indices are different quantities.

The early report is intentionally delivered before the full-band report. The
streaming grouper does not retain it for long-term association; the next
pipeline stage can therefore receive both reports of this one injected FRB.

The script prints the actual member coordinates, recovered DM, S/N and TOA.
It fails with a nonzero exit status if the expected native boundary cells do
not join the same event. It also checks candidate conservation, absence of
timeouts, CPU/GPU decoding agreement, the loudest representative, preserved
representative fields, and emission of the early report in an earlier window. Exact recovered values can differ across software and
hardware; use the recorded evidence from your run.

The output contains:

- `result.json`: pass/fail checks and the boundary event with all its members.
- `catalog_rows.json`: readable event/member tables and processing windows.
- `clustering_inputs.json`: actual decoded inputs to each grouping call;
  halo candidates can appear in more than one call.
- `events.asdf`: the standard trigger catalog, including membership.
- `observation.yml`, `metadata.yml`, `dedispersion.yml`, `grouper.yml`,
  `plan.yml` and `manifest.json`: resolved configuration and injection truth.

To inspect the plan without running GPU processing, choose another new directory:

```bash
python -B -m examples.dm_boundary.run --prepare-only --output /tmp/pirate-dm-boundary-plan
```

## Use the same recipe with the standard commands

The [observation recipe](observation.yml) also works with the ordinary four live
commands. Use one terminal per command, with the same environment and GPU
visibility in every terminal:

```bash
# Terminal 1
python -B -m pirate_frb live grouper examples/dm_boundary/observation.yml
# Terminal 2
python -B -m pirate_frb live dedisperser examples/dm_boundary/observation.yml --gpu 0
# Terminal 3
python -B -m pirate_frb live event_monitor examples/dm_boundary/observation.yml
# Terminal 4: wait for terminals 2 and 3 to print Listening.
python -B -m pirate_frb live observation examples/dm_boundary/observation.yml
```

Wait for the grouper's `Processing complete` message, then stop the dedisperser
first and the remaining services. See the [one-beam example](../simple_frb/README.md#online-stream-the-same-observation)
for ports and process shutdown. The live event monitor prints representatives;
use `run.py` and its member tables to demonstrate the cross-tree association.

To save intensity frames instead, reuse the standard generator:

```bash
python -B -m examples.simple_frb.generate /tmp/pirate-dm-boundary-frames \
    --recipe examples/dm_boundary/observation.yml
```

Then follow the [offline dedispersion and grouping steps](../simple_frb/README.md#offline-generate-dedisperse-group)
with that output directory. This route saves full frames/maps and uses more
disk than `run.py`. If changing metadata or the search plan, rerun preparation
and update the recipe's DM to the newly reported boundary before using the
standard commands.
