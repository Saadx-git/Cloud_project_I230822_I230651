# Cutoff 2 — Reproduction of Guruge & Priyadarshana (2025), Prophet–LSTM Kubernetes autoscaling

- `reproduce.py` — the authors' hybrid notebook logic for NASA/FIFA, plus audit checks (alignment-corrected evaluation, persistence baseline).
- `results/` — `*_results.json` (metrics + environment) and full run logs.
- `report/*.tex` — Overleaf LaTeX source of the reproduction report; `report/*.pdf` — compiled report.
- `requirements.lock.txt` — exact package versions used.

## Re-run (≈15 min on a 2-core CPU)
```bash
git clone https://github.com/pasanbhanu/time-series-forcasting-with-facebook-prophet-lstm-hybrid-model upstream/repo
uv venv --python 3.11 venv && VIRTUAL_ENV=$PWD/venv uv pip install -r requirements.lock.txt
for d in nasa fifa; do venv/bin/python reproduce.py $d upstream/repo results; done
```
