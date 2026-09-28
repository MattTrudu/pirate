# Esempi rieseguiti con il clustering aggiornato

Data: 28 settembre 2026. Esecuzioni reali con il Python dell'ambiente
pirate-15-validation e GPU NVIDIA A40. Le ricette dei due esempi esistenti
sono rimaste invariate.

## Risultato

| Esempio / verifica | Esito |
| --- | --- |
| simple_frb offline: generazione, dedispersione, grouping, catalogo | PASS: 32 chunk, un evento con un membro |
| simple_frb online: quattro processi con gRPC locale | PASS: stesso evento dell'offline, tutti i processi terminati con codice 0 |
| chord_8beams online: ricetta completa da 220 s | PASS: 108 chunk, quattro report per i tre FRB iniettati |
| Nuovo dm_boundary: producer nativo, peakfinder, clustering, catalogo ASDF | PASS: i candidati dei due primary tree sono nello stesso evento |
| Esempio timing, esecuzione ridotta a 64/128 candidati | PASS: 20 casi completi e associazioni identiche fra i batching |
| Controlli CPU dell'esempio frb_recovery | PASS: 11 test |
| Suite pertinente del Grouper dopo la correzione del catalogo | PASS: 61 test, nessun fallimento o skip |

## Esempi esistenti

Il caso simple_frb recupera DM 499.99779521093535, S/N 46.0625,
TOA offline 40.00006641468528 s e width 2.9952 ms. Il monitor online stampa
TOA 40.000067 s: il trasporto quantizza il timestamp su tick FPGA di 5.12 µs
e poi il monitor arrotonda la visualizzazione. La quantizzazione effettiva
rispetto all'offline è 0.145315 µs; non è una differenza di rilevamento.
Il controllo iniziale del report è stato corretto per rispettare questa
quantizzazione, senza cambiare o rieseguire la pipeline.

Il caso chord_8beams produce i report attesi:

| Beam | Tree | DM | TOA (s) | Origine |
| --- | --- | --- | --- | --- |
| 5 | 0 | 550.1536 | 50.004982 | Burst a banda limitata |
| 1 | 0 | 99.9905 | 89.999805 | Burst broadband a DM basso |
| 8 | 1 | 2000.0217 | 200.001516 | Early trigger del burst ad alto DM |
| 8 | 2 | 1999.9912 | 200.000164 | Report full-band dello stesso burst |

Tutti i processi online hanno terminato con codice 0 dopo il completamento.
I servizi sono stati arrestati nell'ordine documentato. Nessun errore, drop
o timeout è stato segnalato nei log; questo non equivale a un contatore
indipendente delle perdite di rete, che il CLI non espone.

Log, comandi, risultati e controlli:

- [simple_frb/README.md](simple_frb/README.md),
  [summary.json](simple_frb/summary.json).
- [chord_8beams/README.md](chord_8beams/README.md),
  [result.json](chord_8beams/result.json).

## Nuovo esempio al bordo DM

L'esempio è in [examples/dm_boundary](../../../examples/dm_boundary/README.md).
La ricetta usa un impulso con DM **1478.537835445747 pc cm⁻³**, S/N 100,
sigma gaussiana 5 ms e banda 300–1500 MHz. Il confine viene verificato contro
il piano effettivo a 0.9984 ms prima della simulazione.

```bash
python -B -m examples.dm_boundary.run --output /tmp/pirate-dm-boundary --gpu 0
```

Usare l'ambiente PIRATE compilato e una directory nuova. Il comando conserva
la ricetta risolta, il piano, gli input del clustering, eventi/membri in JSON
e il catalogo standard events.asdf. Termina con errore se il caso di bordo
non viene effettivamente esercitato.

Nel run verificato, il peakfinder produce tre candidati:

| Tree | Famiglia | Cella DM | DM recuperata | S/N | Risultato |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 4095, ultima | 1478.4927133633348 | 98.625 | Rappresentante dell'evento 1 |
| 2 | 1 | 0, prima | 1478.6506406517778 | 96.5625 | Secondo membro dell'evento 1 |
| 1, early | 1 | 0 | 1478.7634493004555 | 70.125 | Evento 0, finestra precedente |

**Tre picchi → due eventi; l'evento di bordo contiene due membri.**
L'early trigger viene emesso nella finestra 20; il gruppo full-band nella
finestra 36. La successiva pipeline continua quindi a poter ricevere più
report dello stesso impulso.

Sono passati i controlli su conservazione dei candidati, assenza di timeout,
decoding CPU/GPU, rappresentante più forte e relativi campi, oltre alla
presenza del report early nella finestra precedente.

Evidenze: [result.json](dm_boundary/result.json),
[catalog_rows.json](dm_boundary/catalog_rows.json),
[events.asdf](dm_boundary/events.asdf), [log](dm_boundary.log).
Il README dell'esempio include anche la ricetta per i comandi standard
offline/online; la prova documentata qui usa il runner nativo in memoria.

I [controlli della CLI](dm_boundary_prepare_checks.json) verificano anche
preparazione CPU, rifiuto di sovrascrivere un output esistente e rifiuto di una
ricetta con DM 500 quando il piano richiede il vero confine.

## Problema scoperto e corretto

La prima esecuzione del nuovo esempio ha trovato una restrizione residua in
[TriggerCatalog.py](../../../pirate_frb/TriggerCatalog.py): il validatore ASDF
richiedeva ancora che tutti i membri avessero la famiglia del rappresentante.
Il grouping era riuscito, ma il salvataggio del catalogo veniva rifiutato.

È stato rimosso soltanto il vincolo sulla famiglia; rimangono i vincoli su
beam, finestra, proprietà del chunk, collegamenti e rappresentante.
Non è cambiato lo schema del catalogo.

Una nuova regressione in
[test_trigger_catalog.py](../../../pirate_frb/tests/test_trigger_catalog.py)
ha riprodotto l'errore prima della correzione. Dopo la correzione salva e riapre
un evento con due membri dello stesso tree e un rappresentante più forte
nell'halo di una famiglia diversa. I controlli negativi continuano a rifiutare
membri di beam o finestre diverse.

Il tentativo fallito è conservato in
[dm_boundary_before_catalog_fix.log](dm_boundary_before_catalog_fix.log).
Il run finale è stato rieseguito integralmente e ha completato il salvataggio
ASDF con codice 0.

## Controlli aggiuntivi e perimetro

La [suite Grouper](grouper_tests.xml) ha superato 61 test in 8.87 s.
I cinque test catalogo passano anche nella
[verifica mirata](catalog_tests.xml). Packaging e manifest di distribuzione
includono i tre file del nuovo esempio.

Il [timing ridotto](grouper_timing_smoke/README.md) usa un seed, un repeat e
64/128 candidati; ha verificato tutti i 20 casi e l'uguaglianza delle
assegnazioni fra i batching della figura 7. Sono stati aggiornati i suoi
metadati per contare le partizioni per beam. Questi dati non sostituiscono
una campagna di performance.

I [controlli recovery](recovery_cpu_checks.json) sono gli 11 test CPU
dell'esempio. Non è stata rilanciata la campagna di 500 iniezioni.
Slide e plot della presentazione non sono stati rigenerati.
