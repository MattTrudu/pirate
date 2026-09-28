# Eight-beam live example revalidation

**PASS — 28 September 2026.** The unchanged `examples/chord_8beams/observation.yml` recipe completed all 108 science chunks (220.8301056 seconds, eight beams) through the actual four-process loopback pipeline: observation → dedisperser → Grouper → event monitor. Physical GPU 1 was an NVIDIA A40. No production code or example configuration was changed for this validation.

The monitor received the expected four reports, covering all three injected bursts. The high-DM burst produced an early report and a later full-band report, as documented in the example.

| Injected burst | Beam | Tree | Recovered DM (pc cm^-3) | TOA at 300 MHz (s) | S/N |
| --- | --- | --- | --- | --- | --- |
| Narrow band | 5 | 0 | 550.1536 | 50.004982 | 41.656 |
| Low DM | 1 | 0 | 99.9905 | 89.999805 | 47.156 |
| High DM, early | 8 | 1 | 2000.0217 | 200.001516 | 35.219 |
| High DM, full band | 8 | 2 | 1999.9912 | 200.000164 | 50.344 |

The truth association check required the same beam, absolute DM error below 5 pc cm^-3, and absolute arrival-time error below 0.1 s. All reports pass. No errors, exceptions, drops, or timeouts were reported in any role's log. All four processes exited with code 0. The Grouper printed `Processing complete: 4 detection(s).`

## Commands and shutdown

Run these in separate terminals from the repository root, in the order shown. Each terminal used `CUDA_VISIBLE_DEVICES=1` and `PYTHONUNBUFFERED=1`; the CLI thus sees physical GPU 1 as logical GPU 0. Ports 21800–21803 were used to avoid the other validation runs.

```bash
PY=/home/mtrudu/software/miniforge3/envs/pirate-15-validation/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
# Terminal 1
"$PY" -B -m pirate_frb live grouper examples/chord_8beams/observation.yml --base-port 21800
# Terminal 2
"$PY" -B -m pirate_frb live dedisperser examples/chord_8beams/observation.yml --base-port 21800 --gpu 0
# Terminal 3
"$PY" -B -m pirate_frb live event_monitor examples/chord_8beams/observation.yml --base-port 21800
# Terminal 4, after dedisperser and monitor report Listening
"$PY" -B -m pirate_frb live observation examples/chord_8beams/observation.yml --base-port 21800
```

After the Grouper reported completion, the supervisor sent SIGINT to the dedisperser first. The Grouper then exited normally when its client disconnected. Finally the monitor received SIGINT. The observation sender had already exited normally. No process required SIGTERM or SIGKILL.

Total wall time including startup and shutdown was 248.768 s; this is a functional example rerun, not a throughput benchmark. Science input, maps, and catalogs were not saved, matching the live example's behavior.

## Evidence

- [Machine-readable result, commands, matching checks and exit codes](result.json)
- [Grouper log](grouper.log)
- [Dedisperser log](dedisperser.log)
- [Event monitor log](event_monitor.log)
- [Observation log](observation.log)
