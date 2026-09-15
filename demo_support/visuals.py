"""Presentation helpers; no extra plotting or widget dependencies."""

import html
import json
import textwrap
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib.patches import FancyBboxPatch

NAVY = "#17324d"
TEAL = "#087f8c"
ORANGE = "#cc6114"
BLUE = "#3766ad"
PALE = "#edf4f7"
COLORS = [TEAL, ORANGE, BLUE, "#8c5383"]


def setup_style():
    plt.rcParams.update(
        {
            "figure.figsize": (11, 5),
            "figure.dpi": 110,
            "font.size": 13,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": NAVY,
            "text.color": NAVY,
            "axes.edgecolor": "#adbbc6",
            "axes.prop_cycle": plt.cycler(color=COLORS),  # type: ignore
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
        },
    )


def show(fig):
    fig.tight_layout(pad=1.8)
    display(fig)
    plt.close(fig)


def table(frame, caption=None):
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    markup = frame.to_html(index=False, border=0, escape=True, na_rep="Unknown")
    display(
        HTML(
            '<div style="max-width:1100px;font:17px/1.6 system-ui;color:#17324d">'
            + (f"<p><strong>{html.escape(caption)}</strong></p>" if caption else "")
            + "<style>.course-table td,.course-table th{padding:9px 14px;text-align:left;"
            "border-bottom:1px solid #dce5ea}.course-table th{background:#edf4f7}</style>"
            + markup.replace('class="dataframe"', 'class="course-table"')
            + "</div>",
        ),
    )


def cards(items, title=None):
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(12, 3.5), squeeze=False)
    for ax, (label, value, detail) in zip(axes[0], items):
        ax.set_axis_off()
        ax.add_patch(
            FancyBboxPatch(
                (0.02, 0.06),
                0.95,
                0.88,
                boxstyle="round,pad=.02",
                facecolor=PALE,
                edgecolor="none",
            ),
        )
        ax.text(0.08, 0.76, label, fontsize=13, weight="bold", transform=ax.transAxes)
        size = 27 if len(str(value)) < 15 else 20
        ax.text(
            0.08,
            0.48,
            str(value),
            fontsize=size,
            color=TEAL,
            weight="bold",
            transform=ax.transAxes,
        )
        wrapped = "\n".join(
            textwrap.fill(line, width=36 if n >= 3 else 55) for line in detail.split("\n")
        )
        ax.text(0.08, 0.12, wrapped, fontsize=11, transform=ax.transAxes, va="bottom")
    if title:
        fig.suptitle(title, fontsize=21, weight="bold")
    show(fig)


def progress(active=0):
    labels = ["Question", "Gather", "Prepare", "Learn", "Evaluate", "Use + monitor"]
    fig, ax = plt.subplots(figsize=(12, 1.1))
    for i, label in enumerate(labels):
        ax.text(
            i,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=12,
            color="white" if i == active else NAVY,
            bbox=dict(boxstyle="round,pad=.55", fc=TEAL if i == active else PALE, ec="none"),
        )
    ax.set(xlim=(-0.55, 5.8), ylim=(0, 1))
    ax.axis("off")
    show(fig)


def error_bars(labels, values, title, unit, colors=None):
    fig, ax = plt.subplots(figsize=(11, max(3.4, len(labels) * 0.65)))
    bars = ax.barh(labels, values, color=colors or TEAL)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{v:,.1f}" for v in values], padding=8, fontsize=13)
    ax.set(xlabel=f"Average absolute error ({unit}) — lower is better", title=title)
    ax.set_xlim(0, max(values) * 1.23 if max(values) else 1)
    show(fig)


def player(frames, title):
    """Small inline SVG player. Static matplotlib summaries accompany every use."""
    ident = "demo_" + uuid.uuid4().hex
    payload = json.dumps(frames).replace("</", "<\\/")
    display(
        HTML(f"""
    <div id="{ident}" style="font:16px system-ui;max-width:1000px;color:#17324d">
      <p><strong>{html.escape(title)}</strong></p>
      <div class="frame">{frames[0]}</div>
      <button class="play" type="button">Play / pause</button>
      <label>Step <input class="step" aria-label="Animation step" type="range"
        min="0" max="{len(frames) - 1}" value="0" style="width:55%"></label>
      <span class="count">1 / {len(frames)}</span>
      <p style="font-size:13px">If controls are unavailable, use the static plots above.
      Run the cell in your notebook to enable its controls.</p>
    </div>
    <script>(function(){{
      const root=document.getElementById('{ident}');
      const frames={payload}; let index=0,timer=null;
      const slider=root.querySelector('.step');
      function draw(){{root.querySelector('.frame').innerHTML=frames[index];
        slider.value=index;root.querySelector('.count').textContent=(index+1)+' / '+frames.length;}}
      slider.oninput=()=>{{index=Number(slider.value);draw();}};
      root.querySelector('.play').onclick=()=>{{
        if(timer){{clearInterval(timer);timer=null;return;}}
        if(index===frames.length-1) index=0;
        timer=setInterval(()=>{{draw();if(index===frames.length-1){{clearInterval(timer);timer=null;}}
          else index++;}},250);
      }};
    }})();</script>"""),
    )


def ai_landscape():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set(xlim=(0, 12), ylim=(0, 6))
    ax.axis("off")
    boxes = [
        (0.1, 0.1, 7.3, 5.7, PALE),
        (1.8, 0.45, 5.25, 3.9, "#cce9e9"),
        (3.55, 0.8, 3.1, 2.25, "#99d0ce"),
    ]
    for x, y, w, h, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.03", fc=c, ec="white"))
    ax.text(0.45, 5.12, "Artificial intelligence", fontsize=23, weight="bold")
    ax.text(0.45, 4.6, "Includes rule-based systems", fontsize=13)
    ax.text(2.08, 3.73, "Machine learning", fontsize=21, weight="bold")
    ax.text(2.08, 3.25, "Learns patterns from examples", fontsize=12)
    ax.text(3.8, 2.37, "Deep learning", fontsize=19, weight="bold")
    ax.text(3.8, 1.7, "Many-layered\nneural networks", fontsize=13)
    ax.text(7.8, 4.8, "Generative AI", fontsize=22, weight="bold", color=TEAL)
    ax.text(7.8, 3.95, "Creates text, images,\naudio and more.", fontsize=15)
    ax.text(7.8, 2.75, "Modern examples often\nuse deep learning.", fontsize=15)
    ax.text(7.8, 1.05, "A capability, not a fourth\nlearning type.", fontsize=14)
    show(fig)


def classification_errors():
    counts = np.array([[86, 4], [3, 7]])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(counts, cmap="Blues", vmin=0, vmax=90)
    labels = [
        ["Useful mail\narrives", "Useful mail\nhidden"],
        ["Spam slips\nthrough", "Spam\ncaught"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{counts[i, j]}\n{labels[i][j]}",
                ha="center",
                va="center",
                color="white" if counts[i, j] > 40 else NAVY,
                fontsize=14,
            )
    ax.set(
        xticks=[0, 1],
        xticklabels=["Send to inbox", "Send to spam"],
        yticks=[0, 1],
        yticklabels=["Actually useful", "Actually spam"],
        title="Which mistake costs more?\nIllustrative spam-filter results",
    )
    show(fig)
