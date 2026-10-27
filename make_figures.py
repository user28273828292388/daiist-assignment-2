"""
Optional helper: draw the figures for REPORT.md from artifacts/results.json.

    uv run python make_figures.py

It trains nothing and downloads nothing. It reads the results that `main.py train`
saved and writes PNG files to report_figures/, which you can commit and link from
REPORT.md with ![caption](report_figures/<file>.png).

Titles only describe what is plotted. What the figures mean is for your REPORT.
Costs use the assumptions stored in results.json, with the same formulas as train.py
and app.py. The script checks that it reproduces the stored costs before drawing.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
DAMAGE, NO_DAMAGE = "#c0392b", "#2c7fb8"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def cost_grid(scores, y, costs, base_rate):
    """Cost per 1,000 returns for every (lower, upper) edge on GRID. Same maths as train.py."""
    pos, neg = np.sort(scores[y == 1]), np.sort(scores[y == 0])
    p_pass = np.searchsorted(pos, GRID, side="left") / len(pos)
    n_pass = np.searchsorted(neg, GRID, side="left") / len(neg)
    p_charge, n_charge = 1 - p_pass, 1 - n_pass
    pos_term = base_rate * (p_pass[:, None] * costs["fn"] + (1 - p_pass[:, None] - p_charge[None, :]) * costs["review"])
    neg_term = (1 - base_rate) * (n_charge[None, :] * costs["fp"] + (1 - n_pass[:, None] - n_charge[None, :]) * costs["review"])
    total = 1000.0 * (pos_term + neg_term)
    total[GRID[None, :] < GRID[:, None]] = np.nan
    return total


def cost_at(scores, y, low, high, costs, base_rate):
    i, j = int(np.argmin(np.abs(GRID - low))), int(np.argmin(np.abs(GRID - high)))
    return cost_grid(scores, y, costs, base_rate)[i, j]


def save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    print("wrote", out / name)


# ----------------------------------------------------------------------------
def fig_cost_curve(R, S, costs, base, out):
    sg, th = R["design"]["single"], R["design"]["three"]
    do_nothing, check_all = 1000 * base * costs["fn"], 1000 * costs["review"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
    for ax, (xmax, ymax, title) in zip(axes, ((1.0, 1.2 * do_nothing, "Full range (y axis cut at 1.2 x 'do nothing')"),
                                              (0.15, 3 * check_all, "Zoom on thresholds 0 to 0.15"))):
        for key, label, style in (("valid", "Validation: single threshold", "-"), ("test", "Test: single threshold", "--")):
            ax.plot(GRID, np.diag(cost_grid(*S[key], costs, base)), style, label=label, lw=2)
        ax.axvline(sg["threshold"], color="crimson", lw=2, label=f"Chosen single threshold ({sg['threshold']:.2f})")
        ax.axhline(do_nothing, color="black", ls=":", label=f"Do nothing ({do_nothing:,.0f} EUR)")
        ax.axhline(check_all, color="gray", ls=":", label=f"A person checks everything ({check_all:,.0f} EUR)")
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
        ax.set_xlabel("Threshold on the image score"); ax.set_ylabel("Cost per 1,000 returns (EUR)")
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="none"))
    labels.append(f"Pass/review/charge band {th['low']:.2f} to {th['high']:.2f}: valid {th['valid_cost_per_1000']:,.0f} EUR, test {th['test_cost_per_1000']:,.0f} EUR")
    fig.suptitle("Expected cost per 1,000 returns against the threshold", fontsize=12)
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    fig.savefig(out / "cost_curve.png", dpi=160)
    plt.close(fig)
    print("wrote", out / "cost_curve.png")


def fig_band_map(R, S, costs, base, out):
    th = R["design"]["three"]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    total = cost_grid(*S["valid"], costs, base)
    cap = 1000 * base * costs["fn"] / 3
    mesh = ax.pcolormesh(GRID, GRID, np.ma.masked_invalid(total), cmap="viridis_r", vmin=0, vmax=cap, shading="nearest")
    ax.scatter([th["high"]], [th["low"]], marker="*", s=220, color="red", edgecolor="white", zorder=5, label="Chosen band")
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.3)
    ax.set_xlabel("Upper edge: charge from this score"); ax.set_ylabel("Lower edge: pass below this score")
    ax.set_title("Cost per 1,000 returns for every band, validation")
    fig.colorbar(mesh, label=f"EUR per 1,000 (colour capped at {cap:,.0f})")
    ax.legend(loc="upper left")
    save(fig, out, "cost_band_map.png")


def fig_scores(R, S, out):
    th, sg = R["design"]["three"], R["design"]["single"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    rng = np.random.default_rng(0)
    for ax, key in zip(axes, ("valid", "test")):
        scores, y = S[key]
        for level, color, label in ((1, DAMAGE, "With damage"), (0, NO_DAMAGE, "No damage")):
            s = scores[y == level]
            ax.scatter(s, np.full(len(s), level) + rng.uniform(-0.18, 0.18, len(s)), s=28, color=color, alpha=0.8, label=f"{label} ({len(s)})")
        ax.axvline(th["low"], color="gray", ls="--"); ax.axvline(th["high"], color="gray", ls="--")
        ax.axvspan(th["low"], th["high"], color="gold", alpha=0.18, label="Review band")
        ax.axvline(sg["threshold"], color="crimson", lw=1.5, label=f"Single threshold {sg['threshold']:.2f}")
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xticks([0, 0.01, 0.03, 0.1, 0.36, 1]); ax.set_xticklabels(["0", "0.01", "0.03", "0.1", "0.36", "1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No damage", "With damage"]); ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel("Image score (symlog scale)"); ax.set_title(f"Image scores, {key}")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out / "image_scores.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / "image_scores.png")


def fig_outcomes(R, S, out):
    th = R["design"]["three"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key in zip(axes, ("valid", "test")):
        scores, y = S[key]
        rows = []
        for level in (1, 0):
            g = scores[y == level]
            rows.append([int((g < th["low"]).sum()), int(((g >= th["low"]) & (g < th["high"])).sum()), int((g >= th["high"]).sum())])
        ax.imshow(np.array(rows), cmap="Blues", vmin=0)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                ax.text(j, i, v, ha="center", va="center", fontsize=16, color="white" if v > max(map(max, rows)) / 2 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(["Pass", "Send to a person", "Charge"])
        ax.set_yticks(range(2)); ax.set_yticklabels(["Has damage", "No damage"])
        ax.set_title(f"{key.capitalize()} ({len(y)} images)")
    fig.suptitle(f"Outcomes with pass below {th['low']:.2f} and charge from {th['high']:.2f}")
    save(fig, out, "outcomes.png")


def fig_detection_confusion(R, out):
    T = R["detection"]["test"]
    labels = T["confusion_labels"]
    keep = [i for i, n in enumerate(labels) if n not in set(R["assumptions"]["ignored_classes"])]
    M = np.array(T["confusion_matrix"])[np.ix_(keep, keep)]
    names = [labels[i] for i in keep]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(M, cmap="Blues", vmin=0)
    for i in range(len(names)):
        for j in range(len(names)):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("True class"); ax.set_ylabel("Predicted class")
    ax.set_title(f"Detection confusion matrix, test (boxes at confidence {T['confusion_conf']})")
    save(fig, out, "detection_confusion_test.png")


def fig_per_class(R, out):
    ds = {r["class"]: r for r in R["dataset"]["classes"]}
    ignored = set(R["assumptions"]["ignored_classes"])
    test = sorted((c for c in R["detection"]["test"]["per_class"] if c["class"] not in ignored), key=lambda c: -c["ap50"])
    valid = {c["class"]: c for c in R["detection"]["valid"]["per_class"]}
    names = [c["class"] for c in test]
    x = np.arange(len(names)); width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - width / 2, [c["ap50"] for c in test], width, label="Test", color="#2c7fb8")
    ax.bar(x + width / 2, [valid[n]["ap50"] for n in names], width, label="Validation", color="#a6bddb")
    for i, n in enumerate(names):
        ax.text(i - width / 2, test[i]["ap50"] + 0.02, f"n={ds[n]['test']}", ha="center", fontsize=8)
        ax.text(i + width / 2, valid[n]["ap50"] + 0.02, f"n={ds[n]['valid']}", ha="center", fontsize=8, color="dimgray")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("AP50")
    ax.set_title(f"AP50 per class (n = instances in that split). Overall test mAP50 {R['detection']['test']['map50']:.3f}")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    save(fig, out, "per_class_ap50.png")


def fig_composition(R, out):
    ds = R["dataset"]
    ignored = set(R["assumptions"]["ignored_classes"])
    classes = [r for r in ds["classes"] if r["class"] not in ignored]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(len(classes)); width = 0.27
    for k, (split, color) in enumerate((("train", "#2c7fb8"), ("valid", "#7fcdbb"), ("test", "#edf8b1"))):
        axes[0].bar(x + (k - 1) * width, [r[split] for r in classes], width, label=split, color=color, edgecolor="gray")
    axes[0].set_xticks(x); axes[0].set_xticklabels([r["class"] for r in classes], rotation=20, ha="right")
    axes[0].set_ylabel("Labelled objects"); axes[0].set_title("Objects per class and split"); axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
    kinds = {r["split"]: r for r in ds["image_level"]}
    splits = ["train", "valid", "test"]
    damaged = [kinds[s]["damage_only"] for s in splits]; clean = [kinds[s]["only_negative"] for s in splits]
    axes[1].bar(splits, damaged, color=DAMAGE, label="With damage"); axes[1].bar(splits, clean, bottom=damaged, color=NO_DAMAGE, label="No damage")
    for i, s in enumerate(splits):
        axes[1].text(i, damaged[i] + clean[i] + 3, f"{clean[i]} without damage", ha="center", fontsize=8)
    axes[1].set_title("Images per split"); axes[1].set_ylabel("Images"); axes[1].legend(loc="center right")
    axes[1].set_ylim(0, 1.12 * (damaged[0] + clean[0]))
    save(fig, out, "dataset_composition.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="artifacts/results.json")
    parser.add_argument("--out", default="report_figures")
    args = parser.parse_args()

    R = load(Path(args.results))
    out = Path(args.out); out.mkdir(exist_ok=True)
    costs = R["assumptions"]["costs"]; base = R["assumptions"]["base_rate"]
    S = {k: (np.array(R["image_scores"][k]["scores"]), np.array(R["image_scores"][k]["y"])) for k in ("valid", "test")}

    th = R["design"]["three"]
    for key in ("valid", "test"):
        value, stored = cost_at(*S[key], th["low"], th["high"], costs, base), th[f"{key}_cost_per_1000"]
        assert abs(value - stored) < 1e-6, f"{key}: figures would not match the stored cost ({value} vs {stored})"
    print("Check passed: the figures reproduce the costs stored by training.")

    fig_cost_curve(R, S, costs, base, out)
    fig_band_map(R, S, costs, base, out)
    fig_scores(R, S, out)
    fig_outcomes(R, S, out)
    fig_detection_confusion(R, out)
    fig_per_class(R, out)
    fig_composition(R, out)


if __name__ == "__main__":
    main()
