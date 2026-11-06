# Estimated Plan Quality and Realized Flow Time in Dynamic Scheduling on Uniformly Related Machines

Simulation code and result reports for a study of dynamic scheduling of independent jobs on uniformly
related virtual machines under uncertain runtimes.

The study replays the MetaCentrum 2023 workload trace in an event-driven simulator. At every decision point
all waiting jobs are replanned to minimize the sum of estimated completion times. Five methods are
compared: FCFS, ECT, a weighted slot-assignment rule, local search, and an assignment-based mixed policy
(exact planning within a size limit, heuristic fallback outside it). The study asks whether improvements in
the estimated objective lead to improvements in realized mean flow time, and whether additional computation
pays off.

## Contents

| File | Purpose |
|---|---|
| `tahmin_uret.py` | Regenerates submission-time runtime estimates with the predictor described for the trace. |
| `profile_windows.py` | Profiles the trace and writes per-window statistics (`pencereler_1gun.csv`). |
| `pencere_sec.py` | Selects windows by a fixed stratified rule from the window profile. |
| `test_pencereleri.txt` | The 12 daily test windows. |
| `kurallar.py` | Scheduling rules: weighted slot-assignment rule, local search, tie-breaking policy. |
| `dogrulama.py` | Event-driven simulator, ECT/FCFS/exact planning, closed-loop runs and open-loop decision measurements. |
| `kural_ozet.py` | Open-loop summary functions (closure, exact match, realized gain); also used by `test_ozet.py`. |
| `test_ozet.py` | Summary of the test runs: closed-loop differences, exactness coverage, open-loop measures, cost. |
| `kayip_analizi.py` | Post-hoc analysis of one window (day 177): job-level decomposition, idle capacity, switching experiment, perturbed runs. |
| `durum_anlik.py` | Machine occupancy and arrival-wave summaries at selected hours, from the outputs of `kayip_analizi.py`. |
| `results/` | Result reports of the runs below. |

## Data

The MetaCentrum 2023 trace is not redistributed here. It is described in D. Klusáček and V. Chlumský,
"Real-life HPC Workload Trace Featuring Refined Job Runtime Estimates" (JSSPP 2024), and can be obtained
from: https://jsspp.org/workload/index.php. 

Only single-core, single-node jobs without GPUs are used; no other records are removed.

The generated submission-time estimates (`tahmin_gonderim.csv.gz`, about 30 MB) are not included because
of their size. They can be regenerated from the trace (step 1 below) or obtained from the author on
request.

## Environment

Python >= 3.9 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```
Simulation outputs do not depend on the hardware; the measured
computation times do.

## Reproducing the results

**1. Submission-time estimates**

```
python tahmin_uret.py METACENTRUM-2023.swf --out tahmin_gonderim.csv.gz --gunler '@test_pencereleri.txt'
```

Writes `tahmin_gonderim.csv.gz` and the accuracy report `tahmin_gonderim_rapor.txt`. The computation is deterministic.

**2. Window profile and selection**

```
python profile_windows.py METACENTRUM-2023.swf --out profil2
python pencere_sec.py profil2/pencereler_1gun.csv TODO: options used for the test windows
```

The resulting test windows are listed in `test_pencereleri.txt`.

**3. Test runs with submission-time estimates**

```
python dogrulama.py --trace METACENTRUM-2023.swf --days '@test_pencereleri.txt' --rho 0.8,1.2,2.0 --seeds 1 --tahmin tahmin_gonderim.csv.gz --out test_gonderim --workers 5
python test_ozet.py test_gonderim
python test_ozet.py test_gonderim --haric 4,134
```

The first command also extracts the window files to `test_gonderim/pencereler/`; later runs reuse them via
`--pencere-kok`. The summary is written to `test_gonderim/test_raporu.txt` together with CSV files. The
second `test_ozet.py` call adds a sensitivity analysis without days 4 and 134.

**4. Test runs with the recorded soft walltime**

```
python dogrulama.py --days '@test_pencereleri.txt' --rho 0.8,1.2,2.0 --seeds 1 --pencere-kok test_gonderim --out test_kayit --workers 5
python test_ozet.py test_kayit
```

Without `--tahmin`, the recorded soft walltime of the trace is used as the estimate.

**5. Analysis of day 177**

```
python kayip_analizi.py --pencere-kok test_gonderim --gun 177 --rho 0.8,1.2,2.0 --tahmin tahmin_gonderim.csv.gz --out kayip177_gonderim --workers 5
python durum_anlik.py --kok kayip177_gonderim --gun 177 --rho 0.8,1.2,2.0 --saatler 10,11,12,14,16,17,18,20
```

Seed 0 uses the actual runtimes of the trace; seeds 1 and 2 perturb them multiplicatively
(log-normal, standard deviation 0.01). Estimates, arrivals and machines are unchanged across seeds.

## License

Code: MIT License (see `LICENSE`). Result reports: CC BY 4.0.
