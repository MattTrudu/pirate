# Validazione simple_frb — 28 settembre 2026

Esito: **passed**. Eseguiti i tre comandi offline del README e i quattro processi online reali, usando senza modifiche `examples/simple_frb/observation.yml`, sulla GPU fisica 0 (NVIDIA A40). Nessun mock.

## Risultati

- Generazione: 32 chunk, durata 65.4311424 s, seed 137; 32 mappe S/N e argmax salvate.
- Catalogo offline riaperto e validato: **1 evento, 1 membro**, copertura di tutti i 32 chunk, nessun timeout.
- Rappresentante: beam 1, tree 0, DM 499.99779521093535 pc cm^-3, S/N 46.0625, TOA 40.00006641468528 s, width 2.9952 ms, chunk 19, idm 1385, itime 72.
- Monitor online: **1 detection**, DM 499.9978, S/N 46.062, banda 300–1500 MHz, TOA 40.000067 s. I valori coincidono con il catalogo alla precisione del monitor, dopo la conversione del TOA in tick FPGA.
- Tick FPGA = 5.12 microsecondi: il catalogo conserva TOA frazionario; il trasporto usa il tick intero più vicino (7812513). Lo scarto dalla misura offline è 0.145315 microsecondi, inferiore a mezzo tick.
- Grouper online: `Processed 32/32 chunks`, poi `Processing complete: 1 detection(s)`; sender `Sending complete`.
- Tutti i sette comandi terminati con codice **0**. Arrestato prima il dedisperser, poi grouper e monitor, con SIGINT soltanto ai PID avviati dal runner.
- Nessun errore/warning di perdita nei log. Il CLI live non espone un contatore totale dei pacchetti persi: non viene quindi dichiarato un conteggio non misurato.

## Artefatti e riproduzione

Dati generati: `/tmp/pirate-examples-20260928/simple-offline`. [Risultati strutturati](summary.json) con comandi, PID, codici di uscita, catalogo e confronto. Log `offline-*.log` e `live-*.log` in questa directory.

Comandi eseguiti dalla root del repository:

```bash
CUDA_VISIBLE_DEVICES=0 /home/mtrudu/software/miniforge3/envs/pirate-15-validation/bin/python -B -m examples.simple_frb.generate /tmp/pirate-examples-20260928/simple-offline
CUDA_VISIBLE_DEVICES=0 /home/mtrudu/software/miniforge3/envs/pirate-15-validation/bin/python -B -m pirate_frb run offline_dedisperser /tmp/pirate-examples-20260928/simple-offline/frames /tmp/pirate-examples-20260928/simple-offline/dedispersion.yml --save
CUDA_VISIBLE_DEVICES=0 /home/mtrudu/software/miniforge3/envs/pirate-15-validation/bin/python -B -m pirate_frb run offline_grouper /tmp/pirate-examples-20260928/simple-offline/frames /tmp/pirate-examples-20260928/simple-offline/grouper.yml --output /tmp/pirate-examples-20260928/simple-offline/events.asdf
CUDA_VISIBLE_DEVICES=0 /home/mtrudu/software/miniforge3/envs/pirate-15-validation/bin/python -B -m pirate_frb live grouper examples/simple_frb/observation.yml --base-port 21700
CUDA_VISIBLE_DEVICES=0 /home/mtrudu/software/miniforge3/envs/pirate-15-validation/bin/python -B -m pirate_frb live dedisperser examples/simple_frb/observation.yml --base-port 21700
CUDA_VISIBLE_DEVICES=0 /home/mtrudu/software/miniforge3/envs/pirate-15-validation/bin/python -B -m pirate_frb live event_monitor examples/simple_frb/observation.yml --base-port 21700
CUDA_VISIBLE_DEVICES=0 /home/mtrudu/software/miniforge3/envs/pirate-15-validation/bin/python -B -m pirate_frb live observation examples/simple_frb/observation.yml --base-port 21700
```

Il runner di supervisione era `/tmp/pirate_simple_validate.py`; attende le due notifiche Listening prima di avviare l'osservazione e Processing complete prima dello shutdown. Il confronto iniziale del runner usava erroneamente il TOA frazionario formattato direttamente; la verifica è stata corretta applicando la quantizzazione del trasporto, senza ripetere o modificare i processi della pipeline.
