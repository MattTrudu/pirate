# Figures 6 and 7: clustering timings

Run from the repository root with the compiled PIRATE environment active:

```bash
python -B examples/grouper_timing/benchmark.py
python -B examples/grouper_timing/plot.py --input docs/grouper_timing/measurements.json
```

On this machine the working interpreter is
`/home/mtrudu/software/miniforge3/envs/pirate-15-validation/bin/python`.
The benchmark uses GPU 0 and pins its host thread to logical CPU 0 by default;
`--gpu` and `--cpu` select alternatives. It refuses to replace existing timings;
choose another `--output-dir` for a new campaign. Plotting saved timings does not
run a benchmark.

The figures are saved directly in `docs/`:

- `figure6_clustering.png` and `.pdf`
- `figure7_batching.png` and `.pdf`

To match the peakfinding heatmap's percentage-of-real-time scale:

```bash
python -B examples/grouper_timing/plot.py --input docs/grouper_timing/measurements.json --time-unit realtime-percent
```

This writes `figure6_clustering_realtime_percent` and
`figure7_batching_realtime_percent`, each as PNG and PDF, alongside the millisecond
versions. Medians and percentile bands are multiplied by
`100 / metadata.chunk_budget_ms` (currently `100 / 2048`). Figure 6 remains the
cost of **one call** over the stated available beams; Figure 7 measures the
**full 60-beam schedule**, as does the peakfinding heatmap. The percentage is
not divided by the number of beams. Figure 6 keeps a linear y-axis and Figure 7
keeps a logarithmic y-axis.

The current clustering partitions by beam and accepts all directly compatible
rows, including same-tree and cross-primary-family candidates. Saved timings
from the previous implementation remain historical; inspect the recorded
clustering source hash before comparing runs. A small execution check is:

```bash
python -B examples/grouper_timing/benchmark.py --counts 64 128 \
    --seeds 1 --repeats 1 --warmup 1 --output-dir /tmp/pirate-clustering-smoke
```

This checks completion and agreement across batching schedules. One repeat is
not enough for performance conclusions, and no presentation plots are generated.

Raw measurements, summary CSVs, input fingerprints and hardware/software/source
metadata are saved in `docs/grouper_timing/`. No production algorithm is changed.
This experiment is separate from `examples/frb_recovery/run.py`.

## Workloads

Both plots use 64, 128, 256, 512, 1024, 2048, 4096 and 8192, with a log₂ x-axis.
Figure 6 counts **total candidates in one call**, distributed randomly over
1, 2, 10, 30 or 60 available beams. Pairs always stay together. Some beams can
be empty when N is small; actual occupancy and partition sizes are recorded.
Non-beam candidate fields are identical across beam counts for each N/seed.

Figure 7 counts **candidates per beam**, with exactly that many candidates in
each of 60 beams. It measures 1 × 60, 2 × 30, 3 × 20, 6 × 10 and 30 × 2 beam
schedules. All schedules use views of the same input table, made before timing.
The largest case has 491,520 candidates total. Calls execute sequentially on
one GPU. The y-axis is logarithmic so every schedule is visible.

The original PDF's controlled candidate model is preserved: half the rows are
in compatible cross-tree pairs in the same beam and primary-tree family; the
remainder are independent, uniform points in their trees' native DM/time ranges.
Pairs have small DM offsets with compatible residual-dispersion time shifts.
Family zero has only one tree and cannot contribute a cross-tree pair. S/N is
uniform from 10 to 30, and the window is 2048 input samples at 1 ms. The CHORD
configuration supplies ten trees over 300–1500 MHz, with DM/time tolerances 1.5.
These are synthetic normalized candidate tables, not simulated intensity maps.
Their argmax tokens and reporting frequencies are placeholders. Incidental
associations between independently generated rows are allowed.

## Timing and checks

Each point contains 30 measurements: ten repeats for each of three input seeds.
Lines show the median; shaded intervals show the 10th–90th percentiles across
those measurements, not confidence intervals for an astrophysical population.
There are 30 initial warm-up calls and three warm-up schedules per distinct input.
Measured schedule order is shuffled between repeats to reduce clock/thermal bias.

The timer surrounds the full public clustering API, including validation,
sorting, partition construction, association and construction of the GPU result
tables. Synchronization occurs before timing and after every sequential call.
Input generation, upload, slicing, initial compilation, CPU correctness checks,
peakfinding, argmax decoding and catalog output are excluded. Figure 7 uses one
timer around all calls needed to process the full 60-beam workload.

Every timed result must be complete, must not have timed out, and must contain
all input candidates and members. The original 400 ms kernel timeout is retained;
a truncated result stops the campaign rather than being plotted as a timing.
Member indices must cover every candidate exactly once in validation calls.
Figure 7 additionally checks exact candidate-to-representative assignments across
all schedules for every input seed/count, accounting for per-call index changes.
Event counts are checked on every measured repeat. No CPU implementation timings
are included in these figures.

The 2.048 s chunk duration provides a reference budget: percentage spent in
clustering is `100 × total_clustering_ms / 2048`. This does not include the other
pipeline stages and is not a measurement of end-to-end throughput.
