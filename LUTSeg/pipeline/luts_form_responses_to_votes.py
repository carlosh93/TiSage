#!/usr/bin/env python3
"""
Convert Google Form responses to votes_filled.csv for LUTS review.

Input:
- Google Form CSV export (columns like "Select best option for img_0001")
- votes_template.csv (image_id,image_key,selected_option)
- form_option_mapping.json (optional, used to validate available options per image)

Output:
- votes_filled.csv (image_id,image_key,selected_option)

Aggregation:
- Majority vote across reviewers for each image_id.
- Ties can be resolved randomly (seeded), by first option alphabetically, or as error.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_ID_RE = re.compile(r"(img_\d{4})", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Google Form responses to votes_filled.csv.")
    parser.add_argument(
        "--responses-csv",
        default="data/LUTS/Annotations/raw/LUTSeg Dataset Golden Set Review.csv",
        help="Google Form responses CSV export.",
    )
    parser.add_argument(
        "--votes-template-csv",
        default="data/LUTS/Annotations/processed/form_review/votes_template.csv",
        help="votes_template.csv from luts_generate_form_images.py.",
    )
    parser.add_argument(
        "--form-mapping-json",
        default="data/LUTS/Annotations/processed/form_review/form_option_mapping.json",
        help="Optional form_option_mapping.json for per-image option validation.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/LUTS/Annotations/processed/form_review/votes_filled.csv",
        help="Output votes CSV with selected_option filled.",
    )
    parser.add_argument(
        "--tie-break",
        choices=("random", "first", "error"),
        default="random",
        help="How to resolve ties among top-voted options.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --tie-break random.",
    )
    parser.add_argument(
        "--report-ties-csv",
        default="data/LUTS/Annotations/processed/form_review/vote_ties_report.csv",
        help="Optional CSV report for tie cases.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any image has no valid vote after filtering.",
    )
    return parser.parse_args()


def parse_selected_option(raw_value: str) -> str:
    value = (raw_value or "").strip().upper()
    if not value:
        return ""
    if value in {"SKIP", "N/A", "NA", "NONE"}:
        return ""
    if value.startswith("OPTION "):
        value = value.split()[-1].strip()
    if len(value) == 1 and value.isalpha():
        return value
    match = re.search(r"\b([A-Z])\b", value)
    return match.group(1) if match else ""


def load_template(template_csv: Path) -> list[dict[str, str]]:
    with open(template_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            image_id = str(row.get("image_id") or "").strip()
            image_key = str(row.get("image_key") or "").strip()
            if not image_id:
                continue
            rows.append({"image_id": image_id, "image_key": image_key})
    if not rows:
        raise SystemExit(f"No template rows found in {template_csv}")
    return rows


def load_allowed_options(mapping_json: Path) -> dict[str, set[str]]:
    if not mapping_json.exists():
        return {}
    with open(mapping_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get("items", []) if isinstance(payload, dict) else []
    allowed: dict[str, set[str]] = {}
    for item in items:
        image_id = str(item.get("image_id") or "").strip()
        options = item.get("options", {}) or {}
        if not image_id:
            continue
        allowed[image_id] = {str(label).strip().upper() for label in options.keys() if label}
    return allowed


def load_votes_from_form(responses_csv: Path) -> tuple[dict[str, list[str]], int]:
    votes_by_image: dict[str, list[str]] = defaultdict(list)
    invalid_tokens = 0
    with open(responses_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for column_name, raw_value in row.items():
                match = IMAGE_ID_RE.search(str(column_name or ""))
                if not match:
                    continue
                image_id = match.group(1).lower()
                selected = parse_selected_option(str(raw_value or ""))
                if not selected:
                    continue
                if not selected.isalpha() or len(selected) != 1:
                    invalid_tokens += 1
                    continue
                votes_by_image[image_id].append(selected)
    return votes_by_image, invalid_tokens


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    responses_csv = Path(args.responses_csv)
    template_csv = Path(args.votes_template_csv)
    mapping_json = Path(args.form_mapping_json)
    output_csv = Path(args.output_csv)
    ties_csv = Path(args.report_ties_csv)

    template_rows = load_template(template_csv)
    template_ids = {row["image_id"] for row in template_rows}
    allowed_options = load_allowed_options(mapping_json)
    votes_by_image, invalid_tokens = load_votes_from_form(responses_csv)

    ties: list[dict[str, str]] = []
    output_rows: list[dict[str, str]] = []
    no_vote_images = 0
    filtered_out_votes = 0

    for row in template_rows:
        image_id = row["image_id"]
        image_key = row["image_key"]
        raw_votes = votes_by_image.get(image_id, [])

        allowed = allowed_options.get(image_id, set())
        if allowed:
            valid_votes = [option for option in raw_votes if option in allowed]
            filtered_out_votes += len(raw_votes) - len(valid_votes)
        else:
            valid_votes = list(raw_votes)

        selected = ""
        if valid_votes:
            counts = Counter(valid_votes)
            top_count = max(counts.values())
            winners = sorted([option for option, count in counts.items() if count == top_count])
            if len(winners) == 1:
                selected = winners[0]
            else:
                if args.tie_break == "error":
                    raise SystemExit(
                        f"Tie for {image_id}: winners={winners} counts={dict(counts)}. "
                        "Use --tie-break random|first."
                    )
                selected = rng.choice(winners) if args.tie_break == "random" else winners[0]
                ties.append(
                    {
                        "image_id": image_id,
                        "image_key": image_key,
                        "winners": ",".join(winners),
                        "winning_count": str(top_count),
                        "all_counts": ";".join(f"{k}:{v}" for k, v in sorted(counts.items())),
                        "selected_option": selected,
                        "tie_break": args.tie_break,
                    }
                )
        else:
            no_vote_images += 1

        output_rows.append(
            {"image_id": image_id, "image_key": image_key, "selected_option": selected}
        )

    if args.strict and no_vote_images > 0:
        raise SystemExit(
            f"{no_vote_images} image(s) have no valid vote. "
            "Run without --strict to keep them blank."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image_key", "selected_option"])
        writer.writeheader()
        writer.writerows(output_rows)

    ties_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(ties_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "image_key",
                "winners",
                "winning_count",
                "all_counts",
                "selected_option",
                "tie_break",
            ],
        )
        writer.writeheader()
        writer.writerows(ties)

    voted_images = sum(1 for row in output_rows if row["selected_option"])
    extra_response_image_ids = sorted(set(votes_by_image.keys()) - template_ids)

    print(f"Wrote votes: {output_csv}")
    print(f"Template images: {len(template_rows)}")
    print(f"Images with selected option: {voted_images}")
    print(f"Images without valid vote: {no_vote_images}")
    print(f"Ties resolved: {len(ties)} (mode={args.tie_break}, seed={args.seed})")
    print(f"Filtered-out votes (invalid per image options): {filtered_out_votes}")
    print(f"Invalid tokens skipped while parsing: {invalid_tokens}")
    print(f"Tie report: {ties_csv}")
    if extra_response_image_ids:
        print(
            "Response columns with image_id not in template: "
            + ", ".join(extra_response_image_ids[:10])
            + (" ..." if len(extra_response_image_ids) > 10 else "")
        )


if __name__ == "__main__":
    main()
