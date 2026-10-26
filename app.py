"""
Assignment 2: Gradio dashboard.

Run with:  uv run python main.py app

This file trains nothing and downloads nothing. It loads two things that the
training stage produced and that are committed to the repo:

  weights/best.pt           the finetuned YOLO model (used for the "Try an image" tab)
  artifacts/results.json    per-image scores on validation and test, per-class results,
                            the chosen thresholds and the assumptions behind the costs

A case is one image. Its score is the highest confidence among its damage-class boxes
(0 if there is none). The two edges of the escalation band turn that score into a decision:

    score <  lower edge            -> PASS    (close the return, no extra charge)
    lower edge <= score < upper    -> REVIEW  (a person checks the car)
    score >= upper edge            -> CHARGE  (apply the damage fee)

Costs, confusion matrices and the review load are recomputed live from the saved scores,
so moving a slider or changing an assumption updates everything together.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import beta
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "artifacts" / "results.json"
WEIGHTS_DIR = ROOT / "weights"

# ----------------------------------------------------------------------------
# Load what training produced (never retrain, never download)
# ----------------------------------------------------------------------------
if not RESULTS_PATH.exists():
    raise SystemExit("artifacts/results.json is missing. It is produced by `main.py train` and must be committed.")
weights_found = sorted(WEIGHTS_DIR.rglob("*.pt"))
if not weights_found:
    raise SystemExit("No .pt file in weights/. Commit the weights your training run produced.")

RESULTS = json.loads(RESULTS_PATH.read_text())
WEIGHTS = next((p for p in weights_found if p.name == "best.pt"), weights_found[0])
MODEL = YOLO(str(WEIGHTS))

ASSUMED = RESULTS["assumptions"]
NEGATIVE = {c.lower() for c in ASSUMED["negative_classes"]}
IGNORED = {c.lower() for c in ASSUMED["ignored_classes"]}
DAMAGE_CLASSES = {n for n in MODEL.names.values() if n.lower() not in NEGATIVE | IGNORED}
IMGSZ = int(RESULTS["environment"]["imgsz"])

SCORES = {
    split: (np.array(RESULTS["image_scores"][split]["scores"]), np.array(RESULTS["image_scores"][split]["y"]))
    for split in ("valid", "test")
}
SPLITS = {
    "Validation (the thresholds were chosen here)": "valid",
    "Test (final figure only)": "test",
}
CHOSEN_LOW = RESULTS["design"]["three"]["low"]
CHOSEN_HIGH = RESULTS["design"]["three"]["high"]
CHOSEN_SINGLE = RESULTS["design"]["single"]["threshold"]
GRID = np.round(np.linspace(0.0, 1.0, 101), 2)

DEFAULT_COSTS = ASSUMED["costs"]
DEFAULT_HOURLY = DEFAULT_COSTS["review"] * 60 / DEFAULT_COSTS["review_minutes"]
DEFAULT_PEOPLE = ASSUMED["reviewers_available"] or 1.0  # placeholder until the office size is decided


# ----------------------------------------------------------------------------
# Costs. The same formulas as train.py, so the dashboard reproduces its numbers.
# ----------------------------------------------------------------------------
def make_costs(cost_fn, cost_fp, review_minutes, hourly):
    return {"fn": float(cost_fn), "fp": float(cost_fp), "review": float(hourly) * float(review_minutes) / 60}


def outcome_counts(scores, y, low, high):
    """(pass, review, charge) counts for damaged ('pos') and undamaged ('neg') images."""
    out = {}
    for key, group in (("pos", scores[y == 1]), ("neg", scores[y == 0])):
        passed, charged = int((group < low).sum()), int((group >= high).sum())
        out[key] = (passed, len(group) - passed - charged, charged)
    return out


def rates_of(counts):
    return {
        key: dict(zip(("pass", "review", "charge"), (np.array(c) / max(sum(c), 1)).tolist()))
        for key, c in counts.items()
    }


def cost_per_1000(rates, costs, base_rate):
    pos, neg = rates["pos"], rates["neg"]
    per_case = base_rate * (pos["pass"] * costs["fn"] + pos["review"] * costs["review"]) \
        + (1 - base_rate) * (neg["charge"] * costs["fp"] + neg["review"] * costs["review"])
    return 1000.0 * per_case


def review_share(rates, base_rate):
    return base_rate * rates["pos"]["review"] + (1 - base_rate) * rates["neg"]["review"]


def cost_grid(scores, y, costs, base_rate):
    """Expected cost per 1,000 for every (lower edge, upper edge) pair on GRID.
    Rows are the lower edge, columns the upper edge. The diagonal is a single threshold."""
    pos, neg = np.sort(scores[y == 1]), np.sort(scores[y == 0])
    p_pass = np.searchsorted(pos, GRID, side="left") / len(pos)   # P(score < t | damage)
    n_pass = np.searchsorted(neg, GRID, side="left") / len(neg)
    p_charge, n_charge = 1 - p_pass, 1 - n_pass                    # P(score >= t | ...)
    pos_term = base_rate * (p_pass[:, None] * costs["fn"] + (1 - p_pass[:, None] - p_charge[None, :]) * costs["review"])
    neg_term = (1 - base_rate) * (n_charge[None, :] * costs["fp"] + (1 - n_pass[:, None] - n_charge[None, :]) * costs["review"])
    total = 1000.0 * (pos_term + neg_term)
    total[GRID[None, :] < GRID[:, None]] = np.nan                  # upper edge below lower edge: not allowed
    return total


def plateau_centre(values):
    tied = np.flatnonzero(values <= np.nanmin(values) + 1e-9)
    return int(tied[len(tied) // 2])


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


def review_load(share, minutes, cases_per_day, work_minutes=480):
    cases = share * cases_per_day
    hours = cases * minutes / 60
    return cases, hours, hours / (work_minutes / 60)


def money(value):
    return f"{value:,.0f} €"


def self_check():
    """The dashboard's numbers must equal the ones training stored. Say so loudly if not."""
    costs = DEFAULT_COSTS
    problems = []
    for split in ("valid", "test"):
        counts = outcome_counts(*SCORES[split], CHOSEN_LOW, CHOSEN_HIGH)
        value = cost_per_1000(rates_of(counts), costs, ASSUMED["base_rate"])
        stored = RESULTS["design"]["three"][f"{split}_cost_per_1000"]
        if abs(value - stored) > 1e-6:
            problems.append(f"{split}: dashboard {value:.4f} vs stored {stored:.4f}")
    print("Self-check OK: dashboard costs match training." if not problems
          else "WARNING: dashboard costs differ from training: " + "; ".join(problems), flush=True)


# ----------------------------------------------------------------------------
# Tab 1: upload an image
# ----------------------------------------------------------------------------
def decide(score, low, high):
    low, high = float(low), max(float(low), float(high))
    if score is None:
        return "Upload an image to see its decision."
    if score < low:
        outcome, meaning = "PASS", "close the return with no extra charge"
    elif score >= high:
        outcome, meaning = "CHARGE", "apply the damage fee"
    else:
        outcome, meaning = "SEND TO A PERSON", "a person checks the car and decides"
    return (f"### Decision: **{outcome}**\n"
            f"Image score **{score:.3f}** (highest confidence among damage boxes). "
            f"Band: pass below {low:.2f}, review from {low:.2f} to {high:.2f}, charge from {high:.2f}. "
            f"Meaning: {meaning}.")


def predict_image(image, display_conf, low, high):
    if image is None:
        return None, pd.DataFrame(columns=["class", "confidence", "counts as damage"]), None, decide(None, low, high)
    result = MODEL.predict(source=image, conf=0.001, iou=0.6, imgsz=IMGSZ, verbose=False)[0]
    classes = [MODEL.names[int(c)] for c in result.boxes.cls.tolist()]
    confidences = result.boxes.conf.tolist()
    damage = [conf for name, conf in zip(classes, confidences) if name in DAMAGE_CLASSES]
    score = max(damage) if damage else 0.0

    keep = [i for i, conf in enumerate(confidences) if conf >= display_conf]
    shown = np.array(image.convert("RGB")) if not keep else result[keep].plot()[..., ::-1]
    table = pd.DataFrame(
        [{"class": classes[i], "confidence": round(confidences[i], 3), "counts as damage": classes[i] in DAMAGE_CLASSES}
         for i in sorted(keep, key=lambda i: -confidences[i])][:25]
    )
    return shown, table, score, decide(score, low, high)


# ----------------------------------------------------------------------------
# Tabs 2 and 3: confusion matrix, cost, curve
# ----------------------------------------------------------------------------
def confusion_figure(counts, title):
    z = [list(counts["pos"]), list(counts["neg"])]
    text = [[str(v) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Pass", "Send to a person", "Charge"], y=["Has damage", "No damage"],
        text=text, texttemplate="%{text}", textfont=dict(size=24), colorscale="Blues", showscale=False))
    fig.update_layout(title=title, height=330, margin=dict(t=60, b=30), yaxis_autorange="reversed",
                      xaxis_title="What the system does", yaxis_title="What the car really has")
    return fig


def curve_figure(costs, base_rate, split_key, current_low, current_high):
    fig = go.Figure()
    optimum = None
    for key, label, dash in (("valid", "Validation", "solid"), ("test", "Test", "dot")):
        curve = np.diag(cost_grid(*SCORES[key], costs, base_rate))
        fig.add_trace(go.Scatter(x=GRID, y=curve, mode="lines", name=f"{label} (single threshold)", line=dict(dash=dash)))
        if key == "valid":
            optimum = plateau_centre(curve)
            fig.add_trace(go.Scatter(x=[GRID[optimum]], y=[curve[optimum]], mode="markers", name="Validation minimum, current assumptions",
                                     marker=dict(symbol="diamond", size=12, color="green")))
    fig.add_vline(x=CHOSEN_SINGLE, line_color="crimson", line_width=2,
                  annotation_text=f"chosen single threshold {CHOSEN_SINGLE:.2f}", annotation_position="top right")
    fig.add_hline(y=1000 * costs["review"], line_dash="dash", line_color="gray",
                  annotation_text="a person checks everything", annotation_position="bottom right")
    fig.add_hline(y=1000 * base_rate * costs["fn"], line_dash="dash", line_color="black",
                  annotation_text="do nothing (never charge)", annotation_position="top right")
    cut = 1.2 * 1000 * base_rate * costs["fn"]
    fig.update_layout(title="Expected cost per 1,000 returns against the confidence threshold",
                      xaxis_title="Threshold on the image score",
                      yaxis_title="Cost per 1,000 returns (€), axis cut at 1.2 x 'do nothing'",
                      yaxis_range=[0, cut], height=460, margin=dict(t=60), legend=dict(orientation="h", y=-0.25))
    return fig


def heatmap_figure(split_key, costs, base_rate, low, high):
    total = cost_grid(*SCORES[split_key], costs, base_rate)
    cap = 1000 * base_rate * costs["fn"] / 3  # colour scale capped so the cheap region is visible
    fig = go.Figure(go.Heatmap(x=GRID, y=GRID, z=total, zmin=0, zmax=cap, colorscale="Viridis_r",
                               colorbar=dict(title=f"€ per 1,000<br>(capped at {cap:,.0f})"),
                               hovertemplate="lower edge %{y:.2f}<br>upper edge %{x:.2f}<br>%{z:,.0f} €<extra></extra>"))
    fig.add_trace(go.Scatter(x=[CHOSEN_HIGH], y=[CHOSEN_LOW], mode="markers", name="your chosen band",
                             marker=dict(symbol="star", size=16, color="red", line=dict(width=1, color="white"))))
    fig.add_trace(go.Scatter(x=[max(high, low)], y=[low], mode="markers", name="current sliders",
                             marker=dict(symbol="circle-open", size=16, line=dict(width=3, color="black"))))
    fig.update_layout(title="Cost for every band (zoomed to scores up to 0.6; double-click to zoom out)",
                      xaxis_title="Upper edge: charge from", yaxis_title="Lower edge: pass below",
                      xaxis_range=[0, 0.6], yaxis_range=[0, 0.3],
                      height=520, margin=dict(t=60), legend=dict(orientation="h", y=-0.2))
    return fig


def analyse(low, high, split_label, base_rate, cost_fn, cost_fp, review_minutes, hourly, cases_per_day, people):
    split_key = SPLITS[split_label]
    low, high = float(low), float(high)
    band_note = ""
    if high < low:
        high, band_note = low, "The upper edge cannot be below the lower edge, so it is treated as equal (no review band). "
    scores, y = SCORES[split_key]
    costs = make_costs(cost_fn, cost_fp, review_minutes, hourly)
    base_rate = float(base_rate)

    counts = outcome_counts(scores, y, low, high)
    rates = rates_of(counts)
    cost, share = cost_per_1000(rates, costs, base_rate), review_share(rates, base_rate)
    cases, hours, needed = review_load(share, float(review_minutes), float(cases_per_day))
    n_pos, n_neg = sum(counts["pos"]), sum(counts["neg"])
    missed, wrong = counts["pos"][0], counts["neg"][2]
    (fn_lo, fn_hi), (fp_lo, fp_hi) = clopper_pearson(missed, n_pos), clopper_pearson(wrong, n_neg)
    fits = "fits" if needed <= float(people) + 1e-9 else "does NOT fit"

    summary = (
        f"### Expected cost: **{money(cost)} per 1,000 returns**\n"
        f"Assuming {base_rate:.0%} of real returns have damage, a missed damage costs {money(costs['fn'])}, "
        f"a wrong charge {money(costs['fp'])} and a review {costs['review']:.2f} €. {band_note}\n\n"
        f"**Sent to a person:** {share:.1%} of returns → {cases:.1f} cases a day, {hours:.1f} hours a day, "
        f"**{needed:.2f} people**. With {float(people):g} available, that {fits}.\n\n"
        f"**On this split** ({n_pos} damaged and {n_neg} undamaged images):\n"
        f"- Damage that passes unnoticed: {missed} of {n_pos} (95% interval {fn_lo:.0%} to {fn_hi:.0%})\n"
        f"- Undamaged cars charged: {wrong} of {n_neg} (95% interval {fp_lo:.0%} to {fp_hi:.0%})\n\n"
        f"*With so few images every rate is imprecise. The intervals show how imprecise.*"
    )

    single_t = CHOSEN_SINGLE
    single_rates = rates_of(outcome_counts(scores, y, single_t, single_t))
    policies = pd.DataFrame([
        ["Do nothing (never charge)", 1000 * base_rate * costs["fn"], 0.0, 0.0],
        ["Charge every return", 1000 * (1 - base_rate) * costs["fp"], 0.0, 0.0],
        ["A person checks every return", 1000 * costs["review"], 1.0,
         review_load(1.0, float(review_minutes), float(cases_per_day))[2]],
        [f"Model alone (single threshold {single_t:.2f})", cost_per_1000(single_rates, costs, base_rate), 0.0, 0.0],
        [f"Model + person (band {low:.2f} to {high:.2f})", cost, share, needed],
    ], columns=["Policy", "Cost per 1,000 returns (€)", "Share sent to a person", "People needed"])
    policies["Cost per 1,000 returns (€)"] = policies["Cost per 1,000 returns (€)"].round(0)
    policies["Share sent to a person"] = (policies["Share sent to a person"] * 100).round(1).astype(str) + " %"
    policies["People needed"] = policies["People needed"].round(2)

    title = f"Outcomes on {split_key} ({n_pos + n_neg} images)"
    return (confusion_figure(counts, title), summary, policies,
            curve_figure(costs, base_rate, split_key, low, high), heatmap_figure(split_key, costs, base_rate, low, high))


# ----------------------------------------------------------------------------
# Tab 4: per-class results
# ----------------------------------------------------------------------------
def per_class_table():
    counts = pd.DataFrame(RESULTS["dataset"]["classes"]).set_index("class")
    rows = []
    for name in MODEL.names.values():
        if name not in counts.index or name.lower() in IGNORED:
            continue  # the class left out of the analysis does not get a row
        row = {"Class": name, "Instances train / valid / test":
               f"{int(counts.at[name, 'train'])} / {int(counts.at[name, 'valid'])} / {int(counts.at[name, 'test'])}"}
        for split, tag in (("test", "test"), ("valid", "valid")):
            entry = next((c for c in RESULTS["detection"][split]["per_class"] if c["class"] == name), None)
            for field, label in (("precision", "P"), ("recall", "R"), ("ap50", "AP50"), ("ap50_95", "AP50-95")):
                row[f"{label} {tag}"] = round(entry[field], 3) if entry else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("AP50 test", ascending=False)


def class_figure(split_label):
    split_key = SPLITS[split_label]
    data = RESULTS["detection"][split_key]
    matrix = np.array(data["confusion_matrix"])
    labels = data["confusion_labels"]
    text = [["" if v == 0 else str(int(v)) for v in row] for row in matrix]
    fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels, text=text, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(
        title=f"Detection confusion matrix, {split_key} (boxes counted at confidence {data['confusion_conf']})",
        xaxis_title="True class", yaxis_title="Predicted class", yaxis_autorange="reversed", height=560, margin=dict(t=60))
    return fig


def class_notes():
    test = RESULTS["detection"]["test"]
    coco = RESULTS["baselines"]["coco_zero_shot"]
    small = sorted(r["class"] for r in RESULTS["dataset"]["classes"] if r["test"] < 40 and r["class"].lower() not in IGNORED)
    return (
        f"**Overall on test:** mAP50 {test['map50']:.3f}, mAP50-95 {test['map50_95']:.3f}.\n\n"
        f"**Precision and recall** are taken at the confidence that maximises F1 for each class (Ultralytics' convention), "
        f"not at your band. **Classes with fewer than 40 test instances** (the brief's example of a count you cannot trust): "
        f"{', '.join(small)}.\n\n"
        f"**Baselines on the same test split.** The pretrained COCO model has no damage class, so its damage mAP50 is "
        f"{coco['damage_map50']:.1f}; it detected something in {coco['images_with_any_detection']} of {coco['images']} images, "
        f"but only COCO classes such as {', '.join(list(coco['classes_seen'])[:4])}. "
        f"Predicting nothing is the same as never charging. "
        f"The class {', '.join(sorted(IGNORED))} has a single training object and is left out of the analysis."
    )


def dataset_tables():
    splits = pd.DataFrame(RESULTS["dataset"]["splits"])
    image_level = pd.DataFrame(RESULTS["dataset"]["image_level"])
    return splits, image_level


def sensitivity_table():
    rows = pd.DataFrame(RESULTS["sensitivity"])
    out = pd.DataFrame({
        "Assumption moved": rows["assumption"], "Value": rows["value"],
        "Single threshold chosen": rows["single_threshold"],
        "Single, cost on test (€/1000)": rows["single_test_cost_per_1000"].round(0),
        "Band chosen": rows["three_low"].astype(str) + " to " + rows["three_high"].astype(str),
        "Band, cost on test (€/1000)": rows["three_test_cost_per_1000"].round(0),
        "Share to a person": (rows["three_review_share"] * 100).round(1).astype(str) + " %",
        "People needed": rows["three_people_needed"].round(2),
        "Do nothing (€/1000)": rows["do_nothing"].round(0),
        "Person checks all (€/1000)": rows["check_everything"].round(0),
    })
    return out


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
self_check()

with gr.Blocks(title="Rental car return: damage check") as demo:
    gr.Markdown(
        "# Rental car return: pass, review or charge\n"
        "A YOLO model finetuned on a small public dataset of car damage decides, for each return photo, whether to "
        "**pass** it, **charge** the damage fee, or **send it to a person**. The model and the saved results were "
        "produced by `train.py`; this app only loads them. Every cost is an assumption, not a measurement."
    )
    with gr.Row():
        low = gr.Slider(0, 1, value=CHOSEN_LOW, step=0.01, label="Lower edge: pass below this score")
        high = gr.Slider(0, 1, value=CHOSEN_HIGH, step=0.01, label="Upper edge: charge from this score")
        split = gr.Dropdown(list(SPLITS), value=list(SPLITS)[0], label="Evaluate on")
    with gr.Row():
        reset = gr.Button("Reset the band to my chosen one")
        gr.Markdown(
            "Thresholds were chosen on validation. Test is for the final figure: moving the sliders while "
            "watching test would be tuning on test."
        )
    with gr.Accordion("Assumptions (change any of them and everything updates)", open=False):
        with gr.Row():
            base_rate = gr.Slider(0.01, 0.5, value=ASSUMED["base_rate"], step=0.01, label="Share of real returns with damage")
            cost_fn = gr.Number(value=DEFAULT_COSTS["fn"], label="Cost of a missed damage (€)")
            cost_fp = gr.Number(value=DEFAULT_COSTS["fp"], label="Cost of charging an undamaged car (€)")
        with gr.Row():
            review_minutes = gr.Number(value=DEFAULT_COSTS["review_minutes"], label="Minutes to review one return")
            hourly = gr.Number(value=round(DEFAULT_HOURLY, 2), label="Cost of an agent's hour (€)")
            cases_per_day = gr.Number(value=ASSUMED["cases_per_day"], label="Returns per day at the office")
            people = gr.Number(value=DEFAULT_PEOPLE, label="People who can do reviews (placeholder: set yours)")
    controls = [low, high, split, base_rate, cost_fn, cost_fp, review_minutes, hourly, cases_per_day, people]

    with gr.Tabs():
        with gr.Tab("1. Try an image"):
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type="pil", label="Return photo")
                    display_conf = gr.Slider(0.01, 0.9, value=0.25, step=0.01, label="Draw boxes above this confidence (display only)")
                with gr.Column():
                    image_out = gr.Image(label="Predictions", interactive=False)
                    decision_md = gr.Markdown("Upload an image to see its decision.")
            detections = gr.Dataframe(value=pd.DataFrame(columns=["class", "confidence", "counts as damage"]),
                                      headers=["class", "confidence", "counts as damage"],
                                      label="Detections drawn", interactive=False)
            last_score = gr.State(None)
            for trigger in (image_in.change, display_conf.release):
                trigger(predict_image, [image_in, display_conf, low, high], [image_out, detections, last_score, decision_md],
                        api_name=False, show_progress="minimal")
            for slider in (low, high):
                slider.change(decide, [last_score, low, high], decision_md, api_name=False, show_progress="hidden")

        with gr.Tab("2. Confusion matrix and cost"):
            with gr.Row():
                cm_plot = gr.Plot(label="Confusion matrix")
                summary_md = gr.Markdown()
            policy_df = gr.Dataframe(label="Against the alternatives", interactive=False)

        with gr.Tab("3. Cost against threshold"):
            gr.Markdown("The red line is the single threshold chosen on validation. "
                        "The star is the band you chose. The green diamond moves if you change the assumptions.")
            curve_plot = gr.Plot(label="Cost curve")
            heat_plot = gr.Plot(label="Cost of every band")
            with gr.Accordion("Sensitivity stored by training (one assumption moved at a time)", open=False):
                gr.Dataframe(value=sensitivity_table(), interactive=False)

        with gr.Tab("4. Per-class results"):
            gr.Markdown(class_notes())
            gr.Dataframe(value=per_class_table(), label="Per class (test first, then validation)", interactive=False)
            class_split = gr.Dropdown(list(SPLITS), value=list(SPLITS)[1], label="Confusion matrix on")
            class_plot = gr.Plot(value=class_figure(list(SPLITS)[1]), label="Confusion matrix")
            class_split.change(class_figure, class_split, class_plot, api_name=False)
            splits_df, image_level_df = dataset_tables()
            with gr.Accordion("Dataset: images per split and per kind", open=False):
                gr.Dataframe(value=splits_df, interactive=False)
                gr.Dataframe(value=image_level_df, interactive=False)

    outputs = [cm_plot, summary_md, policy_df, curve_plot, heat_plot]
    gr.on([c.change for c in controls], analyse, controls, outputs, api_name="analyse", show_progress="hidden")
    demo.load(analyse, controls, outputs, api_name=False)
    reset.click(lambda: (CHOSEN_LOW, CHOSEN_HIGH), None, [low, high], api_name=False)


if __name__ == "__main__":
    demo.launch()
