# TS-BOSS: Time Series Best Order Score Search

Time-series adaptation of the BOSS algorithm for causal discovery.


## About

This project extends the BOSS algorithm to handle time-series data with temporal dependencies.

Based on:
> Andrews, B., Ramsey, J., et al. (2023). *Fast Scalable and Accurate Discovery of DAGs Using the Best Order Score Search and Grow-Shrink Trees*. NeurIPS.

Original implementation: [https://github.com/bja43/boss](https://github.com/bja43/boss)

## Structure

```
TS-BOSS/
├── src/           # Algorithm implementation
├── utils/         # Utilities (metrics, plotting, data generation)
├── notebooks/     # Main experimental notebook
└── results/       # Experimental results
```

## Usage

Run the main notebook:
```bash
jupyter notebook notebooks/TS-BOSSY_notebook_experiments.ipynb
```

## Experiments

Four experiments varying:
- Sample size (T)
- Graph density (d)
- Number of nodes (N)
- Autocorrelation (ρ)

Methods compared: TS-BOSS, PCMCI+, TS-BOSS on IID data.
