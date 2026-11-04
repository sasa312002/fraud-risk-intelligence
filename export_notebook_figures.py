"""Extract saved chart outputs from a large Jupyter notebook without rerunning it.

Usage from project root:
    python export_notebook_figures.py
    python export_notebook_figures.py --notebook notebooks/01_fraud_risk_analysis.ipynb

Saves all saved chart images, plus copies matching six suggested report figures.
Does not modify the notebook, models, predictions, or evaluation results.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import shutil
from pathlib import Path

FIGURES = [
    (
        "Figure_01_Fraud_Class_Distribution.png",
        r"Fraud\s+vs\s+Legitimate\s+Transaction\s+Distribution",
    ),
    (
        "Figure_02_Transaction_Velocity.png",
        r"Fraud\s+Rate\s+by\s+Transactions?\s+in\s+the\s+Last\s+Hour|Fraud\s+Rate\s+by\s+Transactions?\s+in\s+Last\s+1\s*Hour",
    ),
    (
        "Figure_03_Validation_Model_AP.png",
        r"Section\s*D\s*[—–\-:]?\s*Validation\s+Average\s+Precision|Validation\s+Average\s+Precision\s*\(Higher",
    ),
    (
        "Figure_04_Calibration_Reliability.png",
        r"Section\s*F\s*[—–\-:]?\s*Validation\s+reliability\s+diagrams|Validation\s+reliability\s+diagrams",
    ),
    (
        "Figure_05_Global_SHAP_Importance.png",
        r"Section\s*H\s*[—–\-:]?\s*Global\s+Base.Logit\s+SHAP\s+Importance|Global\s+Base.Logit\s+SHAP\s+Importance",
    ),
    (
        "Figure_06_Robustness_Action_Changes.png",
        r"Section\s*J\s*[—–\-:]?\s*Decision\s+Changes\s+Under\s+Input\s+Perturbations|Decision\s+Changes\s+Under\s+Input\s+Perturbations",
    ),
]


def as_text(value):
    if isinstance(value, list):
        return "".join(value)
    return str(value or "")


def get_cell_image(output):
    data = output.get("data", {})
    for mime_type, extension in (
        ("image/png", ".png"),
        ("image/jpeg", ".jpg"),
        ("image/svg+xml", ".svg"),
    ):
        if mime_type in data:
            content = as_text(data[mime_type])
            try:
                if mime_type == "image/svg+xml":
                    return extension, content.encode("utf-8")
                return extension, base64.b64decode(content)
            except (ValueError, TypeError) as exc:
                print(f"WARNING: Image could not be decoded ({mime_type}): {exc}")
                return None
    return None


def detect_title(source):
    for pattern in (
        r"(?:plt\.title|\.set_title|\.suptitle|plt\.suptitle)\s*\(\s*['\"]([^'\"]+)['\"]",
        r"(?:title)\s*=\s*['\"]([^'\"]+)['\"]",
    ):
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            return match.group(1)[:150]
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/01_fraud_risk_analysis.ipynb"),
        help="Path to a saved .ipynb file (relative to current terminal directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/report_figures"),
        help="Destination folder for extracted charts.",
    )
    args = parser.parse_args()
    notebook = args.notebook.resolve()
    output_folder = args.output.resolve()
    if not notebook.is_file():
        raise SystemExit(f"Notebook not found: {notebook}\nRun from the repository root or specify --notebook.")

    # Read stored Jupyter outputs; intentionally DO NOT execute notebook cells.
    with notebook.open("r", encoding="utf-8") as fh:
        contents = json.load(fh)
    all_folder = output_folder / "all_images"
    all_folder.mkdir(parents=True, exist_ok=True)

    images = []
    for cell_number, cell in enumerate(contents.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = as_text(cell.get("source", ""))
        for image_number, entry in enumerate(cell.get("outputs", []), start=1):
            result = get_cell_image(entry)
            if result is None:
                continue
            suffix, image_bytes = result
            image_file = all_folder / f"cell_{cell_number:03d}_image_{image_number:02d}{suffix}"
            image_file.write_bytes(image_bytes)
            images.append({
                "cell_number": cell_number,
                "image_in_cell": image_number,
                "title_guess": detect_title(source),
                "image_path": str(image_file.relative_to(output_folder)),
                "source": source,
                "file": image_file,
            })

    with (output_folder / "image_index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_number", "image_in_cell", "title_guess", "image_path"])
        for item in images:
            writer.writerow([
                item["cell_number"],
                item["image_in_cell"],
                item["title_guess"],
                item["image_path"],
            ])

    print(f"Notebook: {notebook}")
    print(f"Saved images found: {len(images)}")
    print(f"Output folder: {output_folder}")
    print("\nSuggested report figures:")
    missing = []
    for filename, pattern in FIGURES:
        selected = next((image for image in images if re.search(pattern, image["source"], flags=re.I | re.S)), None)
        if selected is None:
            print(f"  NOT FOUND: {filename}")
            missing.append(filename)
            continue
        destination = output_folder / filename
        # Preserve image format if the notebook stored SVG or JPEG.
        if selected["file"].suffix != ".png":
            destination = destination.with_suffix(selected["file"].suffix)
        shutil.copyfile(selected["file"], destination)
        print(f"  OK: {destination.name} (notebook cell {selected['cell_number']})")

    if not images:
        print("\nNo embedded charts found. Save the notebook with its outputs (Ctrl+S) and rerun this script.")
        print("If outputs were cleared, the images cannot be recovered from the .ipynb itself.")
    elif missing:
        print("\nUnmatched figures can still be selected visually from the 'all_images' folder.")
        print("See 'image_index.csv' for notebook cell numbers; no manual scrolling is needed.")
    else:
        print("\nAll six requested figures extracted. Open 'outputs/report_figures' in File Explorer.")


if __name__ == "__main__":
    main()
