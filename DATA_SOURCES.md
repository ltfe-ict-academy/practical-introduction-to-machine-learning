# Course data sources and interpretation

## Bike rentals

See [bike attribution](01_part_Introduction/data/ATTRIBUTION.md). The small local snapshot includes its original README. It represents historical rental trips, not present-day performance or distinct bicycles required.

## Vienna Airbnb listings

The revised notebooks reuse the original `02_part_Data_Gathering/data/vienna_listings.csv`. It contains 14,123 records with `last_scraped` dates on 14–15 September 2025. Its bytes are preserved by this update.

- Source: [Inside Airbnb](https://insideairbnb.com/get-the-data/).
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), identified by the source download page.
- [Data dictionary](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit): daily prices use local currency; the dollar symbol is an export artifact. Vienna prices are interpreted as EUR, without currency conversion.
- [Data assumptions](https://insideairbnb.com/data-assumptions/) explain collection and interpretation limits.

Preparation keeps IDs for integrity and splitting, parses bathroom descriptions and amenities, and applies fixed repairs for three district-name encoding artifacts. Only positive known target prices enter modeling. Missing inputs are imputed inside the training pipeline. Positive extreme prices remain.

Hosts are kept separate across training, validation, and final test. This assesses unfamiliar hosts within the snapshot, not future market prices. Missing-price records may represent a different population.

The older `vienna_listings_starting.csv` and `airbnb_clean.csv` files remain historical outputs. Revised notebooks do not read or overwrite them, so their historical filtering and encoding choices do not silently enter the new evaluation.
