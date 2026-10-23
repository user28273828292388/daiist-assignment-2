"""
Optional helper: look at a Roboflow Universe dataset BEFORE training on it.

This is not part of the pipeline (main.py does not run it). It downloads one
dataset version and computes, from the label files, the numbers the brief asks
you to describe: split sizes, instances per class and per split, how many
images have no labels at all (pure background), and how big the objects are
relative to the image.

    uv run python explore_dataset.py WORKSPACE PROJECT VERSION

The three values are in the dataset's Universe URL:
    https://universe.roboflow.com/<WORKSPACE>/<PROJECT>/dataset/<VERSION>

Downloads go to datasets/, which is in .gitignore. Never commit a dataset.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_FOLDERS = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
SMALL_CLASS = 40  # the brief's example of a count you cannot trust
# Class names this tool treats as "no damage" when it builds the image-level view.
# It prints which classes it matched, so check that list against the real images.
NEGATIVE_NAMES = {"undamaged", "undamged", "no-damage", "no_damage", "nodamage", "not-damage", "intact"}


def class_names(root: Path) -> list[str]:
    config = yaml.safe_load((root / "data.yaml").read_text())
    names = config["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def read_boxes(label_file: Path) -> list[tuple[int, float, float]]:
    """(class id, width, height) per object, normalised to the image.

    Handles box labels (class cx cy w h) and polygon labels (class x1 y1 x2 y2 ...),
    for which it takes the bounding box of the polygon.
    """
    if not label_file.is_file():
        return []
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id, values = int(float(parts[0])), [float(v) for v in parts[1:]]
        if len(values) == 4:
            width, height = values[2], values[3]
        else:
            xs, ys = values[0::2], values[1::2]
            width, height = max(xs) - min(xs), max(ys) - min(ys)
        boxes.append((class_id, width, height))
    return boxes


def image_kind(row) -> str:
    if not row["has_objects"]:
        return "no_labels"
    if row["has_negative_label"] and row["has_damage_label"]:
        return "both"
    return "only_negative" if row["has_negative_label"] else "damage_only"


def summarize(root: Path) -> dict:
    names = class_names(root)
    negative_ids = {i for i, n in enumerate(names) if n.strip().lower() in NEGATIVE_NAMES}
    object_rows, image_rows = [], []
    for folder in sorted(p.name for p in root.iterdir() if p.is_dir()):
        if folder not in SPLIT_FOLDERS:
            continue
        split = SPLIT_FOLDERS[folder]
        images = [p for p in (root / folder / "images").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
        for image in images:
            boxes = read_boxes(root / folder / "labels" / f"{image.stem}.txt")
            ids = {class_id for class_id, _, _ in boxes}
            image_rows.append({
                "split": split,
                "has_objects": bool(boxes),
                "has_negative_label": bool(ids & negative_ids),
                "has_damage_label": bool(ids - negative_ids),
            })
            for class_id, width, height in boxes:
                object_rows.append({
                    "split": split,
                    "class": names[class_id] if class_id < len(names) else f"id{class_id}",
                    "area_pct": 100 * width * height,
                })
    images_df, objects_df = pd.DataFrame(image_rows), pd.DataFrame(object_rows)

    splits = images_df.groupby("split").agg(
        images=("has_objects", "size"), background_images=("has_objects", lambda s: int((~s).sum()))
    )
    splits["background_pct"] = (100 * splits["background_images"] / splits["images"]).round(1)

    per_class = (
        objects_df.pivot_table(index="class", columns="split", values="area_pct", aggfunc="size", fill_value=0)
        .reindex(columns=[c for c in ["train", "valid", "test"] if c in objects_df["split"].unique()])
    )
    per_class["total"] = per_class.sum(axis=1)
    per_class["median_box_%_of_image"] = objects_df.groupby("class")["area_pct"].median().round(2)
    images_df["kind"] = images_df.apply(image_kind, axis=1)
    kinds = ["damage_only", "only_negative", "both", "no_labels"]
    image_level = pd.crosstab(images_df["split"], images_df["kind"]).reindex(columns=kinds, fill_value=0)
    return {
        "splits": splits,
        "classes": per_class.sort_values("total", ascending=False),
        "image_level": image_level,
        "negative_classes": [names[i] for i in sorted(negative_ids)],
    }


def report(tables: dict) -> None:
    print("\nImages per split (background = no labelled object in the image):\n")
    print(tables["splits"].to_string())
    print("\nObjects per class and split, and typical object size:\n")
    print(tables["classes"].to_string())
    print(f"\nImage-level view. Classes treated as 'no damage': {tables['negative_classes'] or 'NONE MATCHED'}")
    print("  damage_only   = image has at least one damage label and no 'no damage' label")
    print("  only_negative = image has only 'no damage' labels (your usable negatives)")
    print("  both          = image has both kinds of label")
    print("  no_labels     = image with no objects at all\n")
    print(tables["image_level"].to_string())
    split_cols = [c for c in ["train", "valid", "test"] if c in tables["classes"].columns]
    thin = tables["classes"][split_cols].min(axis=1) < SMALL_CLASS
    for name in tables["classes"].index[thin]:
        print(f"\nWARNING: '{name}' has fewer than {SMALL_CLASS} objects in at least one split. "
              f"Any per-class number for it will be unreliable.")
    if "test" not in tables["splits"].index:
        print("\nWARNING: no test split. The brief needs a test split.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workspace")
    parser.add_argument("project")
    parser.add_argument("version", type=int)
    args = parser.parse_args()

    from api_key import load_api_key
    from roboflow import Roboflow

    target = Path("datasets") / f"{args.project}-v{args.version}"
    if not (target / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(args.workspace).project(args.project).version(args.version).download(
            "yolov8", location=str(target)
        )
    report(summarize(target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
