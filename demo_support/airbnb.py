"""One consistent Airbnb workflow shared by the capstone and chapter demos."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .visuals import ORANGE, TEAL, cards, show, table

NUMERIC = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "latitude",
    "longitude",
    "has_air_conditioning",
    "has_washer",
    "has_tv",
]
CATEGORICAL = ["room_type", "district"]
FEATURES = NUMERIC + CATEGORICAL
AMENITIES = {"Air conditioning": "has_air_conditioning", "Washer": "has_washer", "TV": "has_tv"}


def parse_bathrooms(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if "half-bath" in text:
        return 0.5
    match = re.match(r"^(\d+(?:\.\d+)?)\b", text)
    return float(match.group(1)) if match else np.nan


def parse_amenities(value):
    if not isinstance(value, str):
        return None
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else None
    except (ValueError, TypeError):
        return None


def load(root):
    path = Path(root) / "02_part_Data_Gathering/data/vienna_listings.csv"
    raw = pd.read_csv(path, dtype={"id": "string", "host_id": "string"})
    return {"raw": raw, "source_path": path}


def overview(state):
    raw = state["raw"]
    cards(
        [
            ("Vienna snapshot", f"{len(raw):,}", "Listings in the existing source file"),
            ("Collected", "Sep 2025", "14–15 September • historical data"),
            (
                "Known prices",
                f"{raw.price.notna().sum():,}",
                "Missing answers cannot train a model",
            ),
        ],
        "What examples are we learning from?",
    )
    table(
        raw[["room_type", "accommodates", "bedrooms", "bathrooms_text", "price"]]
        .head(5)
        .rename(
            columns={
                "room_type": "Room type",
                "accommodates": "Guests",
                "bedrooms": "Bedrooms",
                "bathrooms_text": "Bathroom description",
                "price": "Price as exported",
            },
        ),
        "A few source records • $ is an export artifact; the local currency is EUR",
    )
    cols = ["price", "bedrooms", "beds", "bathrooms", "bathrooms_text"]
    missing = raw[cols].isna().mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(
        ["Price" if c == "price" else c.replace("_", " ").title() for c in missing.index],
        missing.values,
        color=ORANGE,
    )
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in missing], padding=5)
    ax.set(
        title="Missing does not mean zero",
        xlabel="Share of source records with a missing value (%)",
        xlim=(0, 40),
    )
    show(fig)


def prepare(state):
    raw = state["raw"]
    # Fixed validity rules, not thresholds estimated from target distributions.
    unique = raw.drop_duplicates("id", keep="first").copy()
    data = pd.DataFrame(index=unique.index)
    data["listing_id"] = unique.id
    # Grouping keeps a host's listings in one partition. IDs never enter X.
    data["host_group"] = ("host:" + unique.host_id).fillna("listing:" + unique.id)
    data["price"] = pd.to_numeric(
        unique.price.astype("string").str.replace(r"[$€,]", "", regex=True),
        errors="coerce",
    )
    for col in ["accommodates", "bedrooms", "beds", "bathrooms", "latitude", "longitude"]:
        data[col] = pd.to_numeric(unique[col], errors="coerce")
    data["bathrooms"] = data.bathrooms.fillna(unique.bathrooms_text.map(parse_bathrooms))
    for col in ["bedrooms", "beds", "bathrooms"]:
        data.loc[data[col] < 0, col] = np.nan
    data.loc[data.accommodates < 1, "accommodates"] = np.nan
    data.loc[~data.latitude.between(-90, 90), "latitude"] = np.nan
    data.loc[~data.longitude.between(-180, 180), "longitude"] = np.nan
    data["room_type"] = unique.room_type.astype(object).where(unique.room_type.notna(), np.nan)
    # Fixed repairs for encoding artifacts present in the supplied snapshot.
    repairs = {
        "W\x8ahring": "Währing",
        "D\x9abling": "Döbling",
        "Rudolfsheim-F\x9fnfhaus": "Rudolfsheim-Fünfhaus",
    }
    data["district"] = unique.neighbourhood_cleansed.replace(repairs).astype(object)
    amenities = unique.amenities.map(parse_amenities)
    for name, col in AMENITIES.items():
        data[col] = amenities.map(lambda x: float(name in x) if x is not None else np.nan)
    usable = data[np.isfinite(data.price) & (data.price > 0)].copy()
    # Keep ALL positive observed prices, including unusual expensive listings.
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dev_idx, test_idx = next(split.split(usable, groups=usable.host_group))
    development, test = usable.iloc[dev_idx], usable.iloc[test_idx]
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
    train_idx, val_idx = next(split2.split(development, groups=development.host_group))
    train, validation = development.iloc[train_idx], development.iloc[val_idx]
    state.update(
        data=usable,
        train=train,
        validation=validation,
        test=test,
        development=development,
        unique=unique,
        audit={
            "Source records": len(raw),
            "Unique listing IDs": len(unique),
            "Positive known prices": len(usable),
            "Missing / invalid prices": len(unique) - len(usable),
        },
    )
    return state


def cleaning(state):
    audit = state["audit"]
    table(
        pd.DataFrame({"Check": list(audit), "Records": list(audit.values())}),
        "Every exclusion is visible",
    )
    examples = pd.DataFrame(
        {
            "Source value": [
                "$140.00",
                "1.5 shared baths",
                "Unknown bathroom",
                "0 bedrooms",
                "Same title, different IDs",
            ],
            "Interpretation": [
                "€140.00; no currency conversion",
                "1.5 bathrooms",
                "Keep missing; learn an imputation rule",
                "May be a studio; retain it",
                "Keep both unless evidence shows duplication",
            ],
        },
    )
    table(examples, "Cleaning decisions • illustrative examples using the source format")
    train = state["train"]
    cards(
        [
            ("Training", f"{len(train):,}", "Learn patterns and missing-value rules"),
            ("Validation", f"{len(state['validation']):,}", "Choose model settings"),
            ("Final test", f"{len(state['test']):,}", "Unfamiliar hosts • open at the end"),
        ],
        "Split by host before learning preprocessing",
    )
    median = train.bedrooms.median()
    table(
        pd.DataFrame(
            {
                "Input": ["Bedrooms", "Bathrooms", "District"],
                "If missing": [
                    f"Training median: {median:g} + missing flag",
                    "Use bathroom text first; otherwise training median + flag",
                    "Use a dedicated unknown category",
                ],
            },
        ),
    )


def explore(state):
    train = state["train"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Log spacing shows the complete price range without deleting its tail.
    bins = np.geomspace(train.price.min(), train.price.max() * 1.001, 35)
    axes[0].hist(train.price, bins=bins, color=TEAL, edgecolor="white")
    axes[0].set_xscale("log")
    axes[0].set(
        title="Keep the full price range\nInvestigate unusual values",
        xlabel="Listed price (€; logarithmic spacing)",
        ylabel="Training listings",
    )
    cap = train.price.quantile(0.95)
    scatter = axes[1].scatter(
        train.longitude,
        train.latitude,
        c=train.price,
        s=7,
        alpha=0.6,
        cmap="viridis",
        norm=LogNorm(vmin=train.price.min(), vmax=cap, clip=True),
    )
    axes[1].set(title="Location is a useful clue", xlabel="Longitude", ylabel="Latitude")
    fig.colorbar(
        scatter,
        ax=axes[1],
        label=f"Price € • log colors, capped at €{cap:.0f}",
        shrink=0.8,
    )
    show(fig)
    table(
        pd.DataFrame(
            {
                "Training price summary": ["Median", "95th percentile", "Maximum"],
                "EUR": [round(train.price.median(), 2), round(cap, 2), round(train.price.max(), 2)],
            },
        ),
        "Charts inspect training data; the final test stays reserved",
    )


def features(state):
    table(
        pd.DataFrame(
            {
                "Clue": [
                    "Space",
                    "Location",
                    "Type of accommodation",
                    "Selected amenities",
                    "Listing / host ID",
                    "Listed price",
                ],
                "Examples": [
                    "Guests, bedrooms, beds, bathrooms",
                    "District and coordinates",
                    "Entire place or a room",
                    "Air conditioning, washer, TV",
                    "Retained only for integrity and splitting",
                    "The answer to predict",
                ],
                "Role": ["Input", "Input", "Input", "Input", "Not a model input", "Target"],
            },
        ),
    )
    train = state["train"]
    groups = train.groupby("room_type").price.agg(["median", "count"]).sort_values("median")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(groups.index, groups["median"], color=TEAL)
    ax.bar_label(
        bars,
        labels=[f"€{v:,.0f}  (n={n:,})" for v, n in zip(groups["median"], groups["count"])],
        padding=5,
    )
    ax.set(
        title="Different room types have different listed prices",
        xlabel="Median training price (€)",
        xlim=(0, groups["median"].max() * 1.55),
    )
    show(fig)
    values = []
    for label, col in AMENITIES.items():
        yes, no = train.loc[train[col] == 1, "price"], train.loc[train[col] == 0, "price"]
        values.append((label, yes.median() - no.median(), len(yes), len(no)))
    table(
        pd.DataFrame(
            values,
            columns=["Amenity", "Median difference (€)", "With amenity", "Without amenity"],
        ).round(1),
        "Observed associations, not causal effects or guaranteed price premiums",
    )


def preprocessing():
    return ColumnTransformer(
        [
            (
                "numbers",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                ),
                NUMERIC,
            ),
            (
                "categories",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="Unknown"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                CATEGORICAL,
            ),
        ],
    )


def models():
    # Log-price models yield positive estimates; scores are always in original euros.
    estimators = {
        "Always training median": DummyRegressor(strategy="median"),
        "Linear model": TransformedTargetRegressor(
            regressor=Ridge(alpha=15),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Small decision tree": DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=25,
            criterion="absolute_error",
            random_state=42,
        ),
        "Unrestricted tree": DecisionTreeRegressor(random_state=42),
        "Forest of trees": TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=5,
                max_features=0.85,
                random_state=42,
                n_jobs=1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }
    return {
        name: make_pipeline(preprocessing(), estimator) for name, estimator in estimators.items()
    }


def train(state):
    candidates = models()
    results = []
    for name, model in candidates.items():
        model.fit(state["train"][FEATURES], state["train"].price)
        training_error = mean_absolute_error(
            state["train"].price,
            model.predict(state["train"][FEATURES]),
        )
        validation_error = mean_absolute_error(
            state["validation"].price,
            model.predict(state["validation"][FEATURES]),
        )
        results.append((name, training_error, validation_error))
    scores = pd.DataFrame(results, columns=["Model", "Training MAE", "Validation MAE"])
    winner = scores.loc[scores["Validation MAE"].idxmin(), "Model"]
    state.update(candidates=candidates, scores=scores, winner=winner)
    fig, ax = plt.subplots(figsize=(12, 5))
    y = np.arange(len(scores))
    ax.barh(y - 0.17, scores["Training MAE"], height=0.32, color=TEAL, label="Training")
    ax.barh(y + 0.17, scores["Validation MAE"], height=0.32, color=ORANGE, label="Validation")
    ax.set(
        yticks=y,
        yticklabels=scores.Model,
        xlabel="Average absolute error (€) — lower is better",
        title="Choose using unfamiliar validation hosts",
    )
    ax.invert_yaxis()
    ax.legend(frameon=False)
    show(fig)
    table(scores.round(2), f"Selected on validation: {winner}")
    return winner


def evaluate(state):
    development, test = state["development"], state["test"]
    model = clone(state["candidates"][state["winner"]]).fit(
        development[FEATURES],
        development.price,
    )
    baseline = make_pipeline(preprocessing(), DummyRegressor(strategy="median")).fit(
        development[FEATURES],
        development.price,
    )
    prediction = model.predict(test[FEATURES])
    errors = np.abs(test.price.to_numpy() - prediction)
    mae = errors.mean()
    baseline_mae = mean_absolute_error(test.price, baseline.predict(test[FEATURES]))
    state.update(final_model=model, prediction=prediction, errors=errors, test_mae=mae)
    cards(
        [
            ("Final test MAE", f"€{mae:,.2f}", "Average absolute error over every test listing"),
            ("Median error", f"€{np.median(errors):,.2f}", "Half the absolute errors are smaller"),
            ("Baseline MAE", f"€{baseline_mae:,.2f}", "Always predict the development median"),
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(test.price, prediction, s=15, alpha=0.4, color=TEAL)
    low = min(test.price.min(), prediction.min())
    high = max(test.price.max(), prediction.max())
    axes[0].plot([low, high], [low, high], "--", color=ORANGE, label="Perfect predictions")
    axes[0].set(
        xscale="log",
        yscale="log",
        title="All held-out prices are included",
        xlabel="Actual listed price (€; log spacing)",
        ylabel="Predicted price (€; log spacing)",
    )
    axes[0].legend(fontsize=10, frameon=False)
    details = test[["room_type", "price"]].copy()
    details["error"] = errors
    groups = details.groupby("room_type").error.agg(["mean", "count"]).sort_values("mean")
    bars = axes[1].barh(groups.index, groups["mean"], color=ORANGE)
    axes[1].bar_label(
        bars,
        labels=[f"€{v:.0f}; n={n}" for v, n in zip(groups["mean"], groups["count"])],
        padding=5,
        fontsize=10,
    )
    axes[1].set(
        title="One average can hide uneven results",
        xlabel="Average absolute error (€)",
        xlim=(0, groups["mean"].max() * 1.5),
    )
    show(fig)
    rows = pd.DataFrame(
        {
            "Actual price (€)": test.price.iloc[:5].to_numpy(),
            "Prediction (€)": prediction[:5],
            "Absolute error (€)": errors[:5],
        },
    )
    table(
        rows.round(2),
        "The first five held-out listings • chosen by position, not prediction quality",
    )
    return mae


def prediction_demo(state):
    # A real, plausible development row preserves coherent original-format fields.
    development = state["development"]
    choices = development[
        (development.room_type == "Entire home/apt")
        & (development.accommodates == 4)
        & (development.bedrooms == 2)
    ]
    if choices.empty:
        choices = development
    listing = choices.sort_values("listing_id").iloc[[0]][FEATURES].copy()
    alternative = listing.copy()
    alternative["has_air_conditioning"] = 1 - listing.has_air_conditioning.fillna(0)
    prediction = state["final_model"].predict(listing)[0]
    alternative_prediction = state["final_model"].predict(alternative)[0]
    table(
        pd.DataFrame(
            {
                "Listing detail": [
                    "Room type",
                    "District",
                    "Guests",
                    "Bedrooms",
                    "Bathrooms",
                    "Air conditioning",
                ],
                "Value": [
                    listing.room_type.iloc[0],
                    listing.district.iloc[0],
                    str(listing.accommodates.iloc[0]),
                    str(listing.bedrooms.iloc[0]),
                    str(listing.bathrooms.iloc[0]),
                    "Yes" if listing.has_air_conditioning.iloc[0] == 1 else "No",
                ],
            },
        ),
        "A plausible listing from development data • illustration of usage, not a new test",
    )
    cards(
        [
            ("Model estimate", f"€{prediction:,.0f}", "Listed nightly price"),
            (
                "Toggle air conditioning",
                f"€{alternative_prediction:,.0f}",
                "All other model inputs kept the same",
            ),
            (
                "Evaluation context",
                f"€{state['test_mae']:,.0f}",
                "Test MAE; not an individual price interval",
            ),
        ],
    )
    return listing
