# A 90-minute visual introduction to machine learning

Audience: adults without technical experience. Delivery: instructor-led demos with code collapsed. Goal: understand how examples become predictions and how to judge those predictions.

## Open these notebooks

1. [Machine learning through everyday examples](01_part_Introduction/01_Machine_Learning_Visual_Demos.ipynb) — 65 minutes.
2. [Vienna Airbnb visual walkthrough](04_part_Training_Evaluation_and_Model_Usage/04_Airbnb_Visual_Walkthrough.ipynb) — 20 minutes.

Reserve five minutes for recap/questions. The three supporting notebooks share the Airbnb preparation/evaluation code. All five run independently; no intermediate exports are needed.

## Running order

| Time | Segment | Audience action | Takeaway |
|---|---|---|---|
| 0–12 | What ML does | Guess demand and invent a rule | Learn patterns from examples and apply them to new situations. |
| 12–17 | AI landscape | Discuss automation versus learning | ML is part of AI; deep learning is part of ML. |
| 17–29 | Three learning settings | Predict, group, watch a robot practice | Answers, structure, and rewards provide different signals. |
| 29–41 | Workflow and data | Find sample gaps and table problems | Data must represent the question; cleaning preserves meaning. |
| 41–48 | Features | Select clues and spot leakage | Inputs must be useful and available at prediction time. |
| 48–59 | Learning and model choice | Watch a line improve and inspect unseen points | Good training fit is only part of success. |
| 59–65 | Evaluation | Compare a baseline and different mistakes | Evaluate unfamiliar examples in meaningful units. |
| 65–85 | Airbnb | Follow one listing from source to estimate | Apply the same workflow to a practical question. |
| 85–90 | Recap/questions | Reconstruct the steps without code | Ask what was predicted, from which data, and how evaluated. |

Questions and predictions are included in the timings. If discussion runs long, omit the second shopper grouping, shorten the complexity discussion, and skip the second Airbnb scenario. Preserve the final test and recap.

## Presenter notes

### Start with curiosity

Ask which day needed more rentals; let two people give different reasons. Reveal observations and show why a warm-day rule has exceptions. Rental counts support operations but are not counts of distinct bikes required.

Connect to spam filters, travel-time estimates, and recommendations. Products can combine methods. Distinguish training from prediction; use does not automatically retrain.

### The AI map and learning signals

A fixed spreadsheet formula is automated without necessarily learning. Self-supervised tasks can obtain training signals from examples themselves; this does not need another full demo.

Supervised learning has known answers. Regression predicts numbers; classification predicts categories. The introductory bike prediction uses validation, never the final test.

Ask for shopper groups before revealing colors. Two and three groups are different interpretations, not discovered permanent personalities.

Ask what rewarding robot movement alone would encourage. Reward should match the intended behavior. The world is simulated but the route uses actual Q-learning. The reward chart averages eight runs and shows their spread; episodes do not improve smoothly.

### Gathering and cleaning

The summer-only chart shows missing coverage, not a measured performance comparison. Ask about winter, other cities, and changing habits.

The source bike table is clean; the dirty table is deliberately altered. Missing is not zero, and unusual is not automatically wrong. Do not invent target labels. Explain that imputation values come from training data.

The bike question conditions on observed weather. An advance forecast needs weather forecasts available at prediction time. Do not describe this as an operational tomorrow-morning forecast.

### Features and learning

Registered + casual rentals reveal total rentals: reject those fields even though they look predictive. The feature comparison changes inputs while keeping the model family fixed. Results are measured; do not promise each extra clue improves them.

The line animation averages squared errors. Gradient descent adjusts its parameters; other models learn differently. Later MAE averages absolute errors in familiar units. Mechanism examples are explicitly simulated.

Distinguish missing a pattern from fitting noise. Suitability depends on the data and question; no algorithm is universally best.

### Evaluation

Training is practice, validation helps choose settings, and test is the final assessment. Chronological bike splits expose growth and seasonality. Report the actual baseline comparison, even if the chosen model disappoints.

For spam, hiding useful mail and missing junk have different costs. The illustrative matrix has 90 useful messages and 10 spam messages. Always predicting useful yields 90% accuracy while catching no spam.

### Airbnb

The currency is EUR despite source dollar signs. Predict asking price, not optimal price or revenue. Keep the historical snapshot date visible.

Host grouping evaluates unfamiliar hosts within the snapshot, not future prices. Positive expensive listings remain, so errors may exceed those in the old filtered workflow. Those scores cover different test populations and are not directly comparable.

Log-spaced axes show ratios. Map color capping is a display choice only. Discuss median error alongside MAE and inspect subgroup counts.

The final listing comes from development data for plausible inputs. It illustrates usage, not unseen performance. Toggling air conditioning is a model response, not a causal estimate. MAE is not a price interval for an individual listing.

## Presentation preparation

- Follow the existing environment setup. The root README, requirements, and editor settings are unchanged; no extra runtime requirements are added.
- Run each presentation notebook top to bottom before teaching. Both the repository root and notebook directory work as launch locations.
- Inputs carry collapsed metadata and a `hide-input` tag. Frontends vary; collapse inputs in your chosen viewer before presenting.
- Saved charts/tables are fallbacks. Robot and loss animations also have static summaries.
- Controls use embedded HTML/SVG/JavaScript without widgets or external assets. Run their cells in an interactive notebook to enable controls. GitHub may display only static summaries.
- Use readable zoom and explain axes before interpreting a result.
- Shared implementation lives in `demo_support/`. Revised notebooks do not write datasets.

See [DATA_SOURCES.md](DATA_SOURCES.md) for provenance. Richer tuning, uncertainty estimation, and future-period deployment evaluation belong in later lessons.
