"""Everyday ML demonstrations with real bike data and labeled simulations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .visuals import COLORS, NAVY, ORANGE, PALE, TEAL, cards, error_bars, player, show, table


def load(root):
    data = pd.read_csv(
        Path(root) / "01_part_Introduction/data/bike_day.csv",
        parse_dates=["dteday"],
    )
    data = data.sort_values("dteday").reset_index(drop=True)
    # The original archive's daily-data README specifies temp / 41.
    data["temperature_c"] = data.temp * 41
    data["humidity_pct"] = data.hum * 100
    data["wind_speed"] = data.windspeed * 67
    data["month_sin"] = np.sin(2 * np.pi * (data.mnth - 1) / 12)
    data["month_cos"] = np.cos(2 * np.pi * (data.mnth - 1) / 12)
    data["day_index"] = (data.dteday - pd.Timestamp("2011-01-01")).dt.days
    # Dates define all splits before feature choices or model fitting.
    return {
        "data": data,
        "train": data[data.dteday < "2012-01-01"].copy(),
        "validation": data[(data.dteday >= "2012-01-01") & (data.dteday < "2012-07-01")].copy(),
        "test": data[data.dteday >= "2012-07-01"].copy(),
    }


CALENDAR = ["month_sin", "month_cos", "workingday", "holiday", "day_index"]
WEATHER = ["temperature_c", "humidity_pct", "wind_speed", "weathersit"]
FEATURES = CALENDAR + WEATHER


def hook(state, reveal=False):
    d = state["train"]
    choices = d.loc[d.dteday.isin(pd.to_datetime(["2011-06-15", "2011-06-18"]))]
    items = []
    for _, r in choices.iterrows():
        day = r.dteday.strftime("%a %d %b")
        info = f"{r.temperature_c:.0f} °C • " + ("working day" if r.workingday else "day off")
        info += (
            "\n" + {1: "Clear / partly cloudy", 2: "Mist / cloudy", 3: "Rain / snow"}[r.weathersit]
        )
        items.append((day, f"{r.cnt:,} rentals" if reveal else "Your guess?", info))
    cards(items, "Which day needed more bikes?")


def rules_and_examples(state):
    d = state["train"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for flag, label, marker in [(1, "Working day", "o"), (0, "Day off", "^")]:
        sample = d[d.workingday == flag]
        ax.scatter(sample.temperature_c, sample.cnt, alpha=0.6, s=40, marker=marker, label=label)
    ax.set(
        xlabel="Daily temperature (°C)",
        ylabel="Daily rentals",
        title="Warm days often bring more rentals — but not always",
    )
    ax.legend(frameon=False)
    show(fig)


def supervised(state):
    train = state["train"]
    sample = state["validation"].iloc[[30]]
    model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10))
    model.fit(train[FEATURES], train.cnt)
    prediction = model.predict(sample[FEATURES])[0]
    cards(
        [
            ("Learn from", f"{len(train)} days", "Inputs + known rental counts"),
            ("An unseen day", f"{prediction:,.0f}", "Predicted rentals"),
            ("Reveal its answer", f"{sample.cnt.iloc[0]:,}", "Observed rentals"),
        ],
        "Supervised learning: examples come with answers",
    )
    # This example belongs to validation, never to the final test block.


def shoppers(groups=3):
    rng = np.random.default_rng(23)
    clusters = [
        rng.normal([2, 18], [0.65, 5], size=(35, 2)),
        rng.normal([9, 24], [1.4, 7], size=(35, 2)),
        rng.normal([4, 90], [1.0, 15], size=(35, 2)),
    ]
    data = np.maximum(np.vstack(clusters), [0.2, 2])
    labels = KMeans(n_clusters=groups, n_init=10, random_state=23).fit_predict(
        StandardScaler().fit_transform(data),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(data[:, 0], data[:, 1], color="#8797a3", s=45, alpha=0.8)
    axes[0].set_title("Before: no group labels")
    for k in range(groups):
        subset = data[labels == k]
        axes[1].scatter(
            subset[:, 0],
            subset[:, 1],
            color=COLORS[k % len(COLORS)],
            marker=["o", "^", "s", "D"][k % 4],
            s=45,
            label=f"Group {k + 1}",
        )
    axes[1].set_title(f"After: ask for {groups} groups")
    axes[1].legend(fontsize=11, frameon=False)
    for ax in axes:
        ax.set(xlabel="Visits per month", ylabel="Typical basket (€)")
    fig.suptitle("Unsupervised learning • simulated shoppers", fontsize=21, weight="bold")
    show(fig)
    return labels


WALLS = {(1, 1), (1, 2), (1, 3), (3, 2), (4, 2), (3, 4)}
GOAL = (5, 5)
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def robot_step(position, action):
    dr, dc = MOVES[action]
    target = (position[0] + dr, position[1] + dc)
    if target in WALLS or not (0 <= target[0] < 6 and 0 <= target[1] < 6):
        return position, -0.3, False
    return target, (1.0 if target == GOAL else -0.04), target == GOAL


def train_robot(seed, episodes=350):
    rng = np.random.default_rng(seed)
    q = np.zeros((6, 6, 4))
    returns = []
    for episode in range(episodes):
        position = (0, 0)
        total = 0.0
        epsilon = max(0.05, 0.9 * np.exp(-episode / 80))
        for _ in range(100):
            scores = q[position]
            action = int(
                rng.integers(4)
                if rng.random() < epsilon
                else rng.choice(np.flatnonzero(scores == scores.max())),
            )
            target, reward, done = robot_step(position, action)
            next_value = 0.0 if done else q[target].max()
            q[position + (action,)] += 0.25 * (reward + 0.95 * next_value - q[position + (action,)])
            total += reward
            position = target
            if done:
                break
        returns.append(total)
    return q, np.array(returns)


def robot_route(q=None, seed=7):
    rng = np.random.default_rng(seed)
    path = [(0, 0)]
    for _ in range(35):
        action = int(rng.integers(4) if q is None else np.argmax(q[path[-1]]))
        pos, _, done = robot_step(path[-1], action)
        path.append(pos)
        if done:
            break
    return path


def _robot_svg(path, label):
    parts = [
        '<svg viewBox="0 0 700 380" role="img" aria-label="Robot navigating to its dock">',
        '<rect width="700" height="380" fill="white"/>',
    ]
    for row in range(6):
        for col in range(6):
            color = NAVY if (row, col) in WALLS else ("#b7e2c8" if (row, col) == GOAL else PALE)
            parts.append(
                f'<rect x="{20 + col * 55}" y="{15 + row * 55}" width="51" height="51" rx="5" fill="{color}"/>',
            )
    points = " ".join(f"{47 + c * 55},{42 + r * 55}" for r, c in path)
    parts.append(f'<polyline points="{points}" fill="none" stroke="{ORANGE}" stroke-width="4"/>')
    r, c = path[-1]
    parts.append(f'<circle cx="{47 + c * 55}" cy="{42 + r * 55}" r="15" fill="{TEAL}"/>')
    parts.append(
        f'<text x="385" y="90" font-family="sans-serif" font-size="21" fill="{NAVY}">{label}</text>',
    )
    parts.append(
        '<text x="385" y="140" font-family="sans-serif" font-size="17">Green square: dock</text>',
    )
    parts.append(
        '<text x="385" y="175" font-family="sans-serif" font-size="17">Dark squares: obstacles</text>',
    )
    parts.append(
        f'<text x="385" y="220" font-family="sans-serif" font-size="17">Actions so far: {len(path) - 1}</text></svg>',
    )
    return "".join(parts)


def robot():
    runs = [train_robot(seed) for seed in range(8)]
    before, after = robot_route(), robot_route(runs[0][0])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    grid = np.zeros((6, 6))
    for w in WALLS:
        grid[w] = 1
    for ax, path, title in zip(
        axes[:2],
        [before, after],
        ["Early: exploring", "Later: learned route"],
    ):
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=1)
        ax.plot([p[1] for p in path], [p[0] for p in path], "-o", color=ORANGE, markersize=4)
        ax.scatter([5], [5], marker="s", s=180, color=TEAL, label="Dock")
        ax.text(0, 0, "S", color=NAVY, ha="center", va="center", weight="bold")
        ax.set(title=title, xticks=[], yticks=[])
        ax.set_xlabel(
            f"{len(path) - 1} actions • "
            + ("dock reached" if path[-1] == GOAL else "still searching"),
        )
    returns = np.array([r for _, r in runs])
    smoothed = np.array([pd.Series(r).rolling(15, min_periods=1).mean().values for r in returns])
    axes[2].plot(smoothed.mean(axis=0), color=TEAL, label="Average of 8 runs")
    axes[2].fill_between(
        np.arange(350),
        np.quantile(smoothed, 0.1, axis=0),
        np.quantile(smoothed, 0.9, axis=0),
        color=TEAL,
        alpha=0.15,
        label="10–90% across runs",
    )
    axes[2].set(
        title="Learning through reward",
        xlabel="Practice episode",
        ylabel="Reward (15-episode average)",
    )
    axes[2].legend(fontsize=9, frameon=False)
    show(fig)
    frames = [_robot_svg(before[: i + 1], "Early exploration") for i in range(len(before))]
    frames += [_robot_svg(after[: i + 1], "After learning") for i in range(len(after))]
    player(frames, "Robot navigation • simulated environment, actual Q-learning")
    return {"learned_route": after, "returns": returns}


def coverage(state):
    d = state["train"]
    sample = d[d.mnth.isin([6, 7, 8])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    bins = np.linspace(0, 40, 17)
    for ax, part, title in zip(axes, [sample, d], ["Collect only summer", "Cover a whole year"]):
        ax.hist(part.temperature_c, bins=bins, color=TEAL, edgecolor="white")
        ax.set(title=title, xlabel="Daily temperature (°C)", ylabel="Days observed")
    show(fig)
    cards(
        [
            ("Summer examples", len(sample), "Missing most cold-day conditions"),
            ("Whole training year", len(d), "Broader coverage; still only one city"),
        ],
    )


def cleaning(state):
    d = state["train"].iloc[:5]
    dirty = pd.DataFrame(
        {
            "Record": d.instant.astype(str).values,
            "Temperature": d.temperature_c.round(1).values.astype(object),
            "Day type": ["Working day" if x else "Day off" for x in d.workingday],
        },
    )
    dirty.loc[1, "Temperature"] = np.nan
    dirty.loc[2, "Temperature"] = 240.0
    dirty.loc[3, "Day type"] = "  working DAY "
    dirty = pd.concat([dirty, dirty.iloc[[0]]], ignore_index=True)
    fixed = dirty.drop_duplicates("Record").copy()
    fixed["Temperature"] = pd.to_numeric(fixed.Temperature, errors="coerce")
    fixed.loc[~fixed.Temperature.between(-40, 60), "Temperature"] = np.nan
    fixed["Temperature missing"] = fixed.Temperature.isna()
    training_median = state["train"].temperature_c.median()
    fixed["Temperature"] = fixed.Temperature.fillna(training_median).round(1)
    fixed["Day type"] = (
        fixed["Day type"]
        .str.strip()
        .str.lower()
        .map({"working day": "Working day", "day off": "Day off"})
    )
    table(dirty, "Before • intentionally altered teaching copy")
    table(fixed, "After • duplicate removed, invalid input flagged, training median used")
    return fixed


def feature_choices(state):
    results = []
    for label, cols in [
        ("Calendar", CALENDAR),
        ("Weather", WEATHER),
        ("Calendar + weather", FEATURES),
    ]:
        model = make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), Ridge(alpha=10))
        model.fit(state["train"][cols], state["train"].cnt)
        error = mean_absolute_error(
            state["validation"].cnt,
            model.predict(state["validation"][cols]),
        )
        results.append((label, error))
    error_bars(
        [x[0] for x in results],
        [x[1] for x in results],
        "Which clues help on the rehearsal days?",
        "rentals",
    )
    return results


def loss_demo():
    rng = np.random.default_rng(18)
    x = np.linspace(5, 30, 18)
    y = 20 + 6 * x + rng.normal(0, 13, len(x))
    z = (x - x.mean()) / x.std()
    design = np.column_stack([np.ones(len(z)), z])
    weights = np.array([25.0, -30.0])
    history = []
    for _ in range(61):
        prediction = design @ weights
        history.append((prediction.copy(), float(np.mean((prediction - y) ** 2))))
        weights -= 0.04 * 2 * design.T @ (prediction - y) / len(y)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, index in zip(axes[:2], [0, 60]):
        pred = history[index][0]
        ax.scatter(x, y, s=35, color=TEAL)
        ax.plot(x, pred, color=ORANGE, linewidth=3)
        ax.vlines(x, np.minimum(y, pred), np.maximum(y, pred), color=ORANGE, alpha=0.4)
        ax.set(
            title=f"Update {index}",
            xlabel="Temperature (°C)",
            ylabel="Rentals",
            ylim=(-20, 230),
        )
    axes[2].plot([h[1] for h in history], color=TEAL, linewidth=3)
    axes[2].set(
        title="Training loss falls",
        xlabel="Update",
        ylabel="Mean squared error (rentals²)",
    )
    show(fig)
    frames = []
    for step, (pred, loss) in enumerate(history):
        sx = lambda a: 70 + (a - 5) / 25 * 490
        sy = lambda a: 310 - (a + 20) / 250 * 260
        points = "".join(
            f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="5" fill="{TEAL}"/>' for a, b in zip(x, y)
        )
        gaps = "".join(
            f'<line x1="{sx(a):.1f}" x2="{sx(a):.1f}" y1="{sy(b):.1f}" y2="{sy(c):.1f}" stroke="{ORANGE}" opacity=".4"/>'
            for a, b, c in zip(x, y, pred)
        )
        path = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x, pred))
        frames.append(
            f'<svg viewBox="0 0 900 390" role="img" aria-label="Prediction line and errors"><rect width="900" height="390" fill="white"/>'
            f'<path d="M70 35 V310 H570" fill="none" stroke="{NAVY}"/>{gaps}{points}'
            f'<polyline points="{path}" fill="none" stroke="{ORANGE}" stroke-width="3"/>'
            f'<g font-family="sans-serif" fill="{NAVY}"><text x="200" y="350" font-size="18">Temperature: 5 to 30 °C</text>'
            f'<text x="12" y="25" font-size="17">Rentals</text><text x="620" y="100" font-size="26">Update {step}</text>'
            f'<text x="620" y="155" font-size="20">Loss: {loss:,.0f}</text><text x="620" y="195" font-size="15">mean squared error</text>'
            '<text x="620" y="255" font-size="16">Shorter gaps mean</text><text x="620" y="280" font-size="16">better training predictions.</text></g></svg>',
        )
    player(frames, "Watch the model adjust • simulated bike demand")
    return history


def complexity():
    rng = np.random.default_rng(8)
    x = np.linspace(0, 1, 14)
    underlying = lambda a: 30 + 180 * a - 150 * a**2
    y = underlying(x) + rng.normal(0, 7, len(x))
    new_x = np.linspace(0.02, 0.98, 70)
    new_y = underlying(new_x) + rng.normal(0, 7, len(new_x))
    grid = np.linspace(0, 1, 200)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    scores = []
    models = [
        make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-8, solver="svd"))
        for degree in [1, 2]
    ]
    models.append(DecisionTreeRegressor(random_state=42))  # type: ignore
    for ax, model, label in zip(
        axes,
        models,
        ["Too simple", "Captures the curve", "Memorizes each day"],
    ):
        model.fit(x[:, None], y)
        ax.scatter(5 + 30 * x, y, s=35, color=TEAL, label="Training")
        ax.scatter(
            5 + 30 * new_x,
            new_y,
            marker="x",
            color=ORANGE,
            alpha=0.5,
            s=22,
            label="Unseen rehearsal",
        )
        ax.plot(5 + 30 * grid, model.predict(grid[:, None]), color=NAVY)
        ax.set(title=label, xlabel="Temperature (°C)", ylabel="Rentals")
        scores.append(
            (
                label,
                mean_absolute_error(y, model.predict(x[:, None])),
                mean_absolute_error(new_y, model.predict(new_x[:, None])),
            ),
        )
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle("Model flexibility • simulated demand", fontsize=21, weight="bold")
    show(fig)
    table(pd.DataFrame(scores, columns=["Model", "Training MAE", "Rehearsal MAE"]).round(1))
    return scores


def select_model(state):
    candidates = {
        "Always training median": DummyRegressor(strategy="median"),
        "Linear model": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10)),
        "Small decision tree": DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=12,
            random_state=42,
        ),
        "Forest of trees": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
        ),
    }
    scores = {}
    for label, model in candidates.items():
        model.fit(state["train"][FEATURES], state["train"].cnt)
        scores[label] = mean_absolute_error(
            state["validation"].cnt,
            model.predict(state["validation"][FEATURES]),
        )
    winner = min(scores, key=scores.get)  # type: ignore
    state.update(candidates=candidates, validation_scores=scores, winner=winner)
    error_bars(
        list(scores),
        list(scores.values()),
        "Choose using validation, before the final test",
        "rentals",
    )
    return winner


def evaluate(state):
    development = pd.concat([state["train"], state["validation"]])
    model = clone(state["candidates"][state["winner"]]).fit(development[FEATURES], development.cnt)
    baseline = DummyRegressor(strategy="median").fit(development[FEATURES], development.cnt)
    test = state["test"]
    prediction = model.predict(test[FEATURES])
    error = mean_absolute_error(test.cnt, prediction)
    baseline_error = mean_absolute_error(test.cnt, baseline.predict(test[FEATURES]))
    state.update(final_model=model, test_prediction=prediction, test_mae=error)
    cards(
        [
            ("Chosen on validation", state["winner"], "Then refitted on all development days"),
            ("Final test error", f"{error:,.0f}", "Rentals off, on average"),
            ("Simple baseline", f"{baseline_error:,.0f}", "Rentals off, on average"),
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(test.dteday, test.cnt, color=TEAL, label="Observed", linewidth=1.5)
    axes[0].plot(test.dteday, prediction, color=ORANGE, label="Predicted", linewidth=1.5)
    axes[0].set(title="The final, later days", ylabel="Daily rentals")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(frameon=False)
    axes[1].scatter(test.cnt, prediction, color=TEAL, alpha=0.55, s=30)
    lim = max(test.cnt.max(), prediction.max()) * 1.06
    axes[1].plot([0, lim], [0, lim], "--", color=ORANGE, label="Perfect predictions")
    axes[1].set(
        title="How close are the guesses?",
        xlabel="Observed rentals",
        ylabel="Predicted rentals",
        xlim=(0, lim),
        ylim=(0, lim),
    )
    axes[1].legend(fontsize=10, frameon=False)
    show(fig)
    sample = test.iloc[[0, 30, 60, 90]].copy()
    table(
        pd.DataFrame(
            {
                "Date": sample.dteday.dt.strftime("%d %b %Y"),
                "Observed": sample.cnt,
                "Predicted": prediction[[0, 30, 60, 90]].round(),
                "Absolute error": np.abs(sample.cnt - prediction[[0, 30, 60, 90]]).round(),
            },
        ),
    )
    return error
