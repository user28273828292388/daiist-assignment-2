"""
Assignment 2: training pipeline.

Run with:  uv run python main.py train

Start to finish, with no manual steps, this script:
  1. downloads the Roboflow dataset version named below (needs your API key),
  2. finetunes yolo11n from the pretrained COCO checkpoint,
  3. evaluates it on the test split, per class, against two baselines,
  4. turns detections into ONE score per image (the highest confidence among the
     damage classes) and sweeps thresholds on the VALIDATION split to find the
     cheapest operating point, for a single threshold and for the three-outcome
     design (pass / human review / charge),
  5. reports that operating point on the TEST split, which is touched only for
     the final figure,
  6. saves the weights to weights/ and everything app.py needs to artifacts/.

Every number in the "DECISIONS" block below is an assumption of the author, not a
measurement. REPORT.md says where each one comes from. Change them there first.
"""

import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import beta

from api_key import load_api_key
from explore_dataset import IMAGE_SUFFIXES, class_names, read_boxes, summarize

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
ARTIFACTS = ROOT / "artifacts"

# ----------------------------------------------------------------------------
# The dataset (record these three values in REPORT.md)
# ----------------------------------------------------------------------------
WORKSPACE, PROJECT, VERSION = "celebalworkspace", "car-damage-detection-dy5hu", 1
DATASET_DIR = ROOT / "datasets" / f"{PROJECT}-v{VERSION}"

# ----------------------------------------------------------------------------
# Training settings (record these in REPORT.md). The environment variables only
# exist so the pipeline can be smoke-tested quickly; leave them unset to train.
# ----------------------------------------------------------------------------
BASE_MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 100))
IMGSZ = int(os.environ.get("TRAIN_IMGSZ", 640))
BATCH = int(os.environ.get("TRAIN_BATCH", 16))
PATIENCE = 30
SEED = 0

# ----------------------------------------------------------------------------
# DECISIONS: how a detection becomes a case, and what being wrong costs
# ----------------------------------------------------------------------------
# A case is one image. It is positive ("damage") if it carries at least one label
# of a damage class, negative if it carries only the "no damage" class.
NEGATIVE_CLASSES = {"undamaged"}   # lower case
IGNORED_CLASSES = {"15"}           # one stray object in train: kept in training, left out of the analysis
# The score of an image is the highest confidence among its damage-class boxes,
# so "damaged if any damage box passes the threshold" is "score >= threshold".

# Costs in euros per case, convention A: cost = what the company loses compared with
# deciding correctly. A correct decision (TP, TN) costs 0.
COST_FN = 450.0                    # 350 repair not recovered + 2 days x 50 loss-of-use fee
COST_FP = 107.5                    # 25 claim handling + 7.5 refund + 0.15 x 500 customer value
REVIEW_MINUTES = 20                # time for a person to check one return
HOURLY_COST = 15.5                 # euros per hour of an agent, loaded (middle of 13-18)
BASE_RATE = 0.10                   # share of real returns that carry damage
CASES_PER_DAY = 75                 # returns per day at one office (middle of 50-100)
WORK_MINUTES_PER_DAY = 480
REVIEWERS_AVAILABLE = None         # people who can do reviews; None = do not constrain
# The reviewer is assumed to always decide correctly, so a review costs only their time.

GRID = np.round(np.linspace(0.0, 1.0, 101), 2)


# ----------------------------------------------------------------------------
# 1. Dataset
# ----------------------------------------------------------------------------
def download_dataset() -> Path:
    if not (DATASET_DIR / "data.yaml").exists():
        from roboflow import Roboflow

        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov8", location=str(DATASET_DIR)
        )
    return DATASET_DIR


def write_absolute_yaml(root: Path) -> Path:
    """Roboflow's data.yaml uses relative paths that Ultralytics resolves unreliably."""
    config = {
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": class_names(root),
    }
    target = root / "data_abs.yaml"
    target.write_text(yaml.safe_dump(config))
    return target


def load_truth(root: Path, split: str, names: list[str]) -> tuple[list[str], np.ndarray]:
    """Image paths and 0/1 truth (1 = carries at least one damage label)."""
    negative = {i for i, n in enumerate(names) if n.lower() in NEGATIVE_CLASSES}
    ignored = {i for i, n in enumerate(names) if n.lower() in IGNORED_CLASSES}
    paths, truth = [], []
    images = sorted(p for p in (root / split / "images").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    for image in images:
        ids = {c for c, _, _ in read_boxes(root / split / "labels" / f"{image.stem}.txt")} - ignored
        if not ids:
            continue  # nothing usable to judge this image by
        paths.append(str(image))
        truth.append(int(bool(ids - negative)))
    return paths, np.array(truth)


# ----------------------------------------------------------------------------
# 2. Finetuning
# ----------------------------------------------------------------------------
def train_model(yaml_path: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(yaml_path), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=PATIENCE,
        seed=SEED, deterministic=True, project=str(ROOT / "runs"), name="finetune",
        exist_ok=True, plots=False, verbose=False,
    )
    WEIGHTS_DIR.mkdir(exist_ok=True)
    final = WEIGHTS_DIR / "best.pt"
    shutil.copy(Path(model.trainer.best), final)
    return final


# ----------------------------------------------------------------------------
# 3. Detection-level evaluation and baselines
# ----------------------------------------------------------------------------
def detection_metrics(model, yaml_path: Path, split: str) -> dict:
    """Ultralytics validation. Precision and recall are at the confidence that
    maximises F1 for each class (Ultralytics' convention), not at a fixed threshold."""
    ultralytics_split = "val" if split == "valid" else split  # the key used in data_abs.yaml
    m = model.val(
        data=str(yaml_path), split=ultralytics_split, imgsz=IMGSZ, conf=0.001, iou=0.6,
        plots=False, verbose=False, project=str(ROOT / "runs"), name=f"val_{split}", exist_ok=True,
    )
    per_class = []
    for i, class_index in enumerate(m.box.ap_class_index):
        per_class.append({
            "class": m.names[int(class_index)],
            "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
            "ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i]),
        })
    return {
        "map50": float(m.box.map50), "map50_95": float(m.box.map),
        "per_class": per_class,
        "confusion_matrix": m.confusion_matrix.matrix.tolist(),
        "confusion_labels": [m.names[i] for i in sorted(m.names)] + ["background"],
        "confusion_conf": float(getattr(m.confusion_matrix, "conf", float("nan"))),
    }


def score_images(model, paths: list[str], damage_names: set[str]) -> np.ndarray:
    """One score per image: highest confidence among damage-class boxes (0 if none)."""
    scores = []
    for result in model.predict(source=paths, conf=0.001, iou=0.6, imgsz=IMGSZ, verbose=False, stream=True):
        best = 0.0
        for class_index, confidence in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            if result.names[int(class_index)] in damage_names:
                best = max(best, confidence)
        scores.append(best)
    return np.array(scores)


def coco_zero_shot(paths: list[str]) -> dict:
    """The COCO model has no damage class, so it cannot flag damage: at image level it
    predicts 'no damage' for every image. We still run it and record what it does see."""
    from collections import Counter
    from ultralytics import YOLO

    model = YOLO(BASE_MODEL)
    seen, with_any = Counter(), 0
    for result in model.predict(source=paths, conf=0.25, imgsz=IMGSZ, verbose=False, stream=True):
        classes = {result.names[int(c)] for c in result.boxes.cls.tolist()}
        with_any += bool(classes)
        seen.update(classes)
    return {
        "images": len(paths), "images_with_any_detection": with_any,
        "classes_seen": dict(seen.most_common(10)),
        "damage_map50": 0.0,
        "note": "No COCO class corresponds to a damage class, so it can never flag damage.",
    }


# ----------------------------------------------------------------------------
# 4. Costs and thresholds
# ----------------------------------------------------------------------------
def make_costs(fn=COST_FN, fp=COST_FP, review_minutes=REVIEW_MINUTES, hourly=HOURLY_COST) -> dict:
    return {"fn": fn, "fp": fp, "review": hourly * review_minutes / 60, "review_minutes": review_minutes}


def outcome_rates(scores: np.ndarray, y: np.ndarray, low: float, high: float) -> dict:
    """Share of positives and of negatives that end in each outcome.
    pass: score < low | charge: score >= high | review: in between (low == high: no review)."""
    rates = {}
    for key, group in (("pos", scores[y == 1]), ("neg", scores[y == 0])):
        if len(group) == 0:
            rates[key] = {"pass": float("nan"), "charge": float("nan"), "review": float("nan")}
            continue
        passed, charged = group < low, group >= high
        rates[key] = {
            "pass": float(passed.mean()), "charge": float(charged.mean()),
            "review": float((~passed & ~charged).mean()),
        }
    return rates


def cost_per_1000(rates: dict, costs: dict, base_rate: float) -> float:
    """Expected cost of 1000 real returns when `base_rate` of them have damage.
    Uses the rates measured on each class separately, so the dataset's own
    share of damaged images does not matter. TP and TN cost 0 (convention A)."""
    pos, neg = rates["pos"], rates["neg"]
    per_case = base_rate * (pos["pass"] * costs["fn"] + pos["review"] * costs["review"]) \
        + (1 - base_rate) * (neg["charge"] * costs["fp"] + neg["review"] * costs["review"])
    return 1000.0 * per_case


def review_share(rates: dict, base_rate: float) -> float:
    return base_rate * rates["pos"]["review"] + (1 - base_rate) * rates["neg"]["review"]


def review_load(share: float, minutes: float = REVIEW_MINUTES) -> dict:
    cases = share * CASES_PER_DAY
    hours = cases * minutes / 60
    return {"cases_per_day": cases, "hours_per_day": hours, "people": hours / (WORK_MINUTES_PER_DAY / 60)}


def best_single(scores, y, costs, base_rate) -> dict:
    """Cheapest single threshold. If several thresholds tie, take the middle of the plateau."""
    values = np.array([cost_per_1000(outcome_rates(scores, y, t, t), costs, base_rate) for t in GRID])
    tied = np.flatnonzero(values <= values.min() + 1e-9)
    index = int(tied[len(tied) // 2])
    return {"threshold": float(GRID[index]), "cost_per_1000": float(values[index]),
            "curve": values.tolist(), "plateau": [float(GRID[tied[0]]), float(GRID[tied[-1]])]}


def best_three(scores, y, costs, base_rate, max_share: float | None = None) -> dict:
    """Cheapest pair (low, high). Ties are broken like best_single, so that when the
    best design uses no review at all it picks the same threshold: among equally cheap
    pairs keep those with the fewest reviews, then take the middle one."""
    candidates = []
    for i, low in enumerate(GRID):
        for high in GRID[i:]:
            rates = outcome_rates(scores, y, low, high)
            share = review_share(rates, base_rate)
            if max_share is not None and share > max_share + 1e-12:
                continue
            candidates.append({"low": float(low), "high": float(high), "share": share,
                               "cost": cost_per_1000(rates, costs, base_rate)})
    cheapest = min(c["cost"] for c in candidates)
    tied = [c for c in candidates if c["cost"] <= cheapest + 1e-9]
    fewest = min(c["share"] for c in tied)
    tied = sorted((c for c in tied if c["share"] <= fewest + 1e-12), key=lambda c: (c["low"], c["high"]))
    chosen = tied[len(tied) // 2]
    return {"low": chosen["low"], "high": chosen["high"],
            "cost_per_1000": float(chosen["cost"]), "review_share": float(chosen["share"])}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float | None]:
    if n == 0:
        return [None, None]
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lower, upper]


def evaluate_design(valid, test, costs, base_rate, max_share=None) -> dict:
    """Pick thresholds on VALIDATION, then measure them on TEST."""
    (sv, yv), (st, yt) = valid, test
    single = best_single(sv, yv, costs, base_rate)
    three = best_three(sv, yv, costs, base_rate, max_share)
    r_single = outcome_rates(st, yt, single["threshold"], single["threshold"])
    r_three = outcome_rates(st, yt, three["low"], three["high"])
    return {
        "single": {**{k: single[k] for k in ("threshold", "plateau")},
                   "valid_cost_per_1000": single["cost_per_1000"],
                   "test_cost_per_1000": cost_per_1000(r_single, costs, base_rate),
                   "test_rates": r_single},
        "three": {"low": three["low"], "high": three["high"],
                  "valid_cost_per_1000": three["cost_per_1000"],
                  "test_cost_per_1000": cost_per_1000(r_three, costs, base_rate),
                  "test_review_share": review_share(r_three, base_rate),
                  "test_rates": r_three,
                  "load": review_load(review_share(r_three, base_rate), costs["review_minutes"])},
    }


def reference_policies(costs: dict, base_rate: float) -> dict:
    """Three ways of running the office that need no threshold."""
    return {
        "do_nothing_never_charge": 1000 * base_rate * costs["fn"],
        "always_charge": 1000 * (1 - base_rate) * costs["fp"],
        "person_checks_everything": 1000 * costs["review"],
        "person_checks_everything_load": review_load(1.0, costs["review_minutes"]),
    }


def sensitivity(valid, test) -> list[dict]:
    """Move one assumption at a time and redo the whole threshold choice."""
    cases = [("cost_fn", v) for v in (150, 300, 450, 600, 800)] \
        + [("base_rate", v) for v in (0.05, 0.10, 0.15)] \
        + [("review_minutes", v) for v in (10, 20, 35)]
    rows = []
    for name, value in cases:
        costs = make_costs(fn=value if name == "cost_fn" else COST_FN,
                           review_minutes=value if name == "review_minutes" else REVIEW_MINUTES)
        base = value if name == "base_rate" else BASE_RATE
        result = evaluate_design(valid, test, costs, base)
        rows.append({
            "assumption": name, "value": value,
            "single_threshold": result["single"]["threshold"],
            "single_test_cost_per_1000": result["single"]["test_cost_per_1000"],
            "three_low": result["three"]["low"], "three_high": result["three"]["high"],
            "three_test_cost_per_1000": result["three"]["test_cost_per_1000"],
            "three_review_share": result["three"]["test_review_share"],
            "three_people_needed": result["three"]["load"]["people"],
            "do_nothing": reference_policies(costs, base)["do_nothing_never_charge"],
            "check_everything": reference_policies(costs, base)["person_checks_everything"],
        })
    return rows


# ----------------------------------------------------------------------------
# 5. Environment record
# ----------------------------------------------------------------------------
def environment() -> dict:
    import torch
    import ultralytics

    return {
        "where": "Google Colab" if "COLAB_RELEASE_TAG" in os.environ else platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(), "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "base_model": BASE_MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
        "patience": PATIENCE, "seed": SEED,
    }


def jsonable(value):
    if isinstance(value, float) and value != value:
        return None  # NaN is not valid JSON
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    started = time.time()
    root = download_dataset()
    yaml_path = write_absolute_yaml(root)
    names = class_names(root)
    damage_names = {n for n in names if n.lower() not in NEGATIVE_CLASSES | IGNORED_CLASSES}

    tables = summarize(root)
    dataset_stats = {
        "workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
        "splits": tables["splits"].reset_index().to_dict("records"),
        "classes": tables["classes"].reset_index().to_dict("records"),
        "image_level": tables["image_level"].reset_index().to_dict("records"),
    }

    from ultralytics import YOLO

    weights = train_model(yaml_path)
    model = YOLO(str(weights))

    detection = {split: detection_metrics(model, yaml_path, split) for split in ("valid", "test")}

    data = {}
    for split in ("valid", "test"):
        paths, truth = load_truth(root, split, names)
        data[split] = {"paths": [Path(p).name for p in paths], "y": truth,
                       "scores": score_images(model, paths, damage_names)}
    valid = (data["valid"]["scores"], data["valid"]["y"])
    test = (data["test"]["scores"], data["test"]["y"])

    costs = make_costs()
    max_share = None
    if REVIEWERS_AVAILABLE is not None:
        max_share = REVIEWERS_AVAILABLE * WORK_MINUTES_PER_DAY / (CASES_PER_DAY * REVIEW_MINUTES)
    design = evaluate_design(valid, test, costs, BASE_RATE, max_share)

    # Exact 95% intervals on the test rates at the chosen single threshold.
    t = design["single"]["threshold"]
    yt, st = test[1], test[0]
    k_fn, n_pos = int(((st < t) & (yt == 1)).sum()), int((yt == 1).sum())
    k_fp, n_neg = int(((st >= t) & (yt == 0)).sum()), int((yt == 0).sum())
    design["single"]["test_counts"] = {"false_negatives": k_fn, "positives": n_pos,
                                       "false_positives": k_fp, "negatives": n_neg}
    design["single"]["test_fnr_ci95"] = clopper_pearson(k_fn, n_pos)
    design["single"]["test_fpr_ci95"] = clopper_pearson(k_fp, n_neg)

    results = {
        "dataset": dataset_stats, "environment": environment(),
        "assumptions": {"costs": costs, "base_rate": BASE_RATE, "cases_per_day": CASES_PER_DAY,
                        "work_minutes_per_day": WORK_MINUTES_PER_DAY, "reviewers_available": REVIEWERS_AVAILABLE,
                        "negative_classes": sorted(NEGATIVE_CLASSES), "ignored_classes": sorted(IGNORED_CLASSES)},
        "detection": detection,
        "baselines": {
            "coco_zero_shot": coco_zero_shot(load_truth(root, "test", names)[0]),
            "predict_nothing_image_level": "scores of 0 for every image: never charges",
            "test_images": {"positives": n_pos, "negatives": n_neg},
        },
        "image_scores": {s: {"files": data[s]["paths"], "y": data[s]["y"], "scores": data[s]["scores"]} for s in data},
        "design": design,
        "single_threshold_curves": {
            "grid": GRID, "valid": best_single(*valid, costs, BASE_RATE)["curve"],
            "test": best_single(*test, costs, BASE_RATE)["curve"],
        },
        "reference_policies": reference_policies(costs, BASE_RATE),
        "sensitivity": sensitivity(valid, test),
        "runtime_seconds": time.time() - started,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "results.json").write_text(json.dumps(jsonable(results), indent=2))

    print("\n=== Done ===")
    print(f"Weights: {weights}")
    print(f"Test mAP50 {detection['test']['map50']:.3f} | mAP50-95 {detection['test']['map50_95']:.3f}")
    print(f"Single threshold (chosen on valid): {design['single']['threshold']:.2f} -> "
          f"{design['single']['test_cost_per_1000']:,.0f} EUR per 1000 cases on test")
    three = design["three"]
    print(f"Pass below {three['low']:.2f}, review {three['low']:.2f}-{three['high']:.2f}, charge from {three['high']:.2f} -> "
          f"{three['test_cost_per_1000']:,.0f} EUR per 1000, {100 * three['test_review_share']:.1f}% to review "
          f"({three['load']['people']:.2f} people)")
    print("Do nothing / person checks everything:",
          f"{results['reference_policies']['do_nothing_never_charge']:,.0f} /",
          f"{results['reference_policies']['person_checks_everything']:,.0f} EUR per 1000")
    print(f"Saved artifacts/results.json in {results['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
