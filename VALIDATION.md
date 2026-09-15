# Validation of the visual course update

Validated on 15 September 2026 against repository base `3ad981f79d00d70988f46566a521c6400129513f`.

## Executed material

All five notebooks were validated against notebook schema v4 and their **37 code cells** were executed top to bottom without errors or saved stderr warnings. Executed outputs include **43 static plots** across the five notebooks. Both presentation notebooks were visually inspected, and crowded labels were corrected.

| Notebook | Code cells | Static plots |
|---|---:|---:|
| Visual ML introduction | 18 | 18 |
| Data gathering | 2 | 2 |
| Data cleaning and features | 4 | 3 |
| Airbnb visual walkthrough | 8 | 15 |
| Training and evaluation | 5 | 5 |

Each notebook executed in a fresh isolated IPython process. The introduction ran from the repository root; the remaining notebooks ran from their chapter directories. The sandbox blocked Jupyter kernel sockets, so this was cell execution with captured notebook outputs, **not a live Jupyter frontend test**.

The robot and learning-line JavaScript controls passed slider/play/pause handler checks using a small DOM stub. Static summaries are provided for viewers that strip scripts. The controls were not exercised in a live Jupyter frontend here.

## Data and modeling checks

Seven automated checks pass:

- Hosts and listing IDs are disjoint across Airbnb partitions, with complete partition coverage.
- The positive price tail is retained.
- Unknown bathroom descriptions remain missing, while half-baths and zero are interpreted distinctly.
- IDs and answer-revealing fields are excluded from predictive inputs.
- Bike partitions are chronological and cover all 731 days.
- The fitted Airbnb pipeline accepts an unseen category and missing numeric inputs.
- The learned robot reaches the dock without crossing obstacles.

Run them from the repository root with `python -m unittest discover -s tests -v`.

Local links in the course documents and notebook Markdown resolve. The root README, requirements, editor settings, original Airbnb CSV, and both historical prepared CSVs are byte-for-byte unchanged.

## Representative saved results

The bike example selects the linear model using validation. Final MAE is approximately **983 rentals**, compared with **2,367** for the development-median baseline.

The Airbnb example selects the log-price linear model using validation. Final MAE is **€54.88**, compared with **€64.79** for the median baseline; median absolute error is **€27.38**. The unrestricted tree's training MAE is approximately **€0.46**, while validation MAE is **€124.82**, illustrating poor generalization despite a very close training fit.

These are reproducible teaching outputs from the supplied snapshots, not guarantees of future performance. The old filtered Airbnb workflow used a different population, so its error numbers should not be compared directly with these.

## Runtime used for this check

Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, Matplotlib 3.10.8, scikit-learn 1.8.0, and IPython 8.37.0.

The repository's existing environment instructions and pinned requirements were preserved. The exact Python/library versions specified there were not reproduced in this sandbox. No new runtime dependency was added to the course.
