# Cloud_project_I230822_I236051
Cloud Computing Project Repository


Member 1:
Saad Azhar
23I-0822
Cloud-A

Member 2:
Syed Ali Ramish Abidi
23I-0651
Cloud-A

## Cutoff 2 — Paper reproduction
Reproduction of Guruge & Priyadarshana (2025), Prophet–LSTM Kubernetes autoscaling: see [`cutoff2_reproduction/`](cutoff2_reproduction/) (code, results, LaTeX report).

## Cutoff 3 — Methodology & preliminary results
**Research question:** can a Prophet–LSTM predictive autoscaler fit inside a Kubernetes controller's CPU budget, produce timely and causally valid forecasts, and improve on the reactive HPA?

See [`cutoff3/`](cutoff3/), which contains:
* a budget-aware KEDA external scaler (gRPC; CPU limit 100m–500m, deadline = control period, kill and persistence fallback);
* a trace-driven simulator;
* a live kind + Prometheus + KEDA testbed;
* experiments E1–E4 with raw results, figures, and the LaTeX report.

Main preliminary findings (single runs):
* **E1:** the base paper's evaluation is leaky. Under causal evaluation the hybrid's RMSE is 40–43 % higher, and AR(10) beats it on both traces.
* **E2:** inference fits every budget, with no late forecasts. Cold start (270 s at 100m), memory and re-fitting are the real costs.
* **E3/E4:** request-rate scaling cuts under-provisioning compared with the CPU HPA (live Θ_U 12.1 % → 0.7 %), but persistence achieves the same gain. Prophet alone fails on flash crowds.

## Cutoff 4 — Day-80 complete draft
The final experiments extend `cutoff3/`: E4 repeated with three load-generator seeds plus a persistence control,
E2 down to 25m, and E5 statistics (block-bootstrap CIs, deadline sensitivity). See
[`cutoff3/README.md`](cutoff3/README.md#final-results-day-80-draft). The draft itself (two-column IEEE LaTeX) is in [`cutoff4_draft/`](cutoff4_draft/).

Findings:
* **Validity:** the published accuracy gain is an artefact of a leaky evaluation; AR(10) is the most accurate causal model.
* **Cost:** the hybrid fits even 25m for steady-state inference, but cold start (270 s at 100m), ≈0.7 GB of memory
  and re-training are the real costs, and deadlines ≤ 2 s are missed at 100m.
* **Benefit:** request-rate scaling beats the CPU HPA in every live seed (Θ_U 21.1 % → 0.9 %), but persistence
  achieves the same (0.8 %); the hybrid is not significantly better than persistence on bursty or ramping load.

License: MIT (see `LICENSE`). How to cite: `CITATION.cff`.

