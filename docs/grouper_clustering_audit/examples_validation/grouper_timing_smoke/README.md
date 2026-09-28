# Esecuzione limitata dell'esempio grouper_timing

28 settembre 2026: **PASS**, codice di uscita 0, GPU fisica 0 (NVIDIA A40), CPU 0.

Eseguiti i casi con 64 e 128 candidati, un seed, un warm-up per caso e una ripetizione. Sono completati tutti i **20 casi** e sono conservati tutti i membri. Per ogni dimensione, le cinque suddivisioni in batch della figura 7 producono esattamente le stesse assegnazioni candidato–rappresentante. I metadati registrano le partizioni correnti per beam.

Questo è un controllo di esecuzione: le poche misure non sostituiscono una campagna di performance. Nessuna figura è stata generata o aggiornata.

[Esito e comando](summary.json), [log](run.log), [misure e verifiche complete](measurements.json).

Comando eseguito dalla root del repository:

```bash
CUDA_VISIBLE_DEVICES=0 /home/mtrudu/software/miniforge3/envs/pirate-15-validation/bin/python -B examples/grouper_timing/benchmark.py --gpu 0 --counts 64 128 --seeds 1 --repeats 1 --warmup 1 --output-dir /home/mtrudu/src/pirate-1.5-integration/docs/grouper_clustering_audit/examples_validation/grouper_timing_smoke
```
