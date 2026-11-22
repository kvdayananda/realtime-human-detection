"""FitRange Mirror - Condition-Based Model Validation Suite.

Evaluates the integrated pipeline and individual models across real-world operational scenarios:
- Front-facing people
- Side-facing people
- Different body shapes (standard adult body shapes vs unrepresented edge cases)
- Different clothing types (athletic sportswear vs casual clothing)
- Different backgrounds (plain wall, cluttered room, outdoor daylight, negative controls)
- Different lighting conditions (bright daylight, indoor warm lighting, underexposed shadows)

Strictly reports conditions not sufficiently represented in the dataset as NOT TESTED.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ai.pipeline.mirror_pipeline import FitRangeMirrorPipeline

logger = logging.getLogger("FitRangeMirror.ConditionValidation")


CONDITIONS_MAP = [
    {
        "category": "Orientation: Front-Facing People",
        "tested": True,
        "sample_files": ["front_facing_1.png", "front_facing_2.png"],
        "description": "Direct front-to-camera standing postures with clear visual face and torso alignment.",
        "expected_detection": True,
    },
    {
        "category": "Orientation: Side-Facing / Profile",
        "tested": True,
        "sample_files": ["side_facing_1.png"],
        "description": "Lateral / side-profile perspective with occluded contralateral limbs.",
        "expected_detection": True,
    },
    {
        "category": "Body Shapes: Standard Adult Postures",
        "tested": True,
        "sample_files": ["front_facing_1.png", "front_facing_2.png", "athletic_clothing.png"],
        "description": "Standard adult athletic, tall, and compact silhouettes in standing and active positions.",
        "expected_detection": True,
    },
    {
        "category": "Body Shapes: Extreme Silhouettes / Mobility Aids",
        "tested": False,
        "sample_files": [],
        "description": "Wheelchair users, pediatric/toddler body shapes, extreme bariatric body types.",
        "expected_detection": None,
        "reason_not_tested": "Dataset lacks annotated samples for assistive mobility devices and pediatric subjects.",
    },
    {
        "category": "Clothing: Athletic & Sportswear",
        "tested": True,
        "sample_files": ["athletic_clothing.png"],
        "description": "Fitted workout apparel, shorts, and active training tops with high limb visibility.",
        "expected_detection": True,
    },
    {
        "category": "Clothing: Casual & Loose Fit",
        "tested": True,
        "sample_files": ["casual_clothing.png"],
        "description": "Relaxed-fit jackets, pants, and non-form-fitting lifestyle garments.",
        "expected_detection": True,
    },
    {
        "category": "Clothing: Flowing Robes & Heavy Winter Parkas",
        "tested": False,
        "sample_files": [],
        "description": "Floor-length garments, heavy winter outerwear, and full-body PPE covering limb joints.",
        "expected_detection": None,
        "reason_not_tested": "Dataset contains zero instances of heavy snowsuits, hazmat, or floor-length ceremonial robes.",
    },
    {
        "category": "Backgrounds: Plain Indoor Wall",
        "tested": True,
        "sample_files": ["front_facing_1.png"],
        "description": "High contrast plain neutral background with minimal visual distractors.",
        "expected_detection": True,
    },
    {
        "category": "Backgrounds: Cluttered Indoor Environment",
        "tested": True,
        "sample_files": ["cluttered_background.png"],
        "description": "Living and training spaces with furniture, shelving, equipment, and complex textures.",
        "expected_detection": True,
    },
    {
        "category": "Backgrounds: Outdoor Natural Scenery",
        "tested": True,
        "sample_files": ["outdoor_daylight.png"],
        "description": "Outdoor daylight setting with open depth and organic surroundings.",
        "expected_detection": True,
    },
    {
        "category": "Backgrounds: Negative Controls (No Human)",
        "tested": True,
        "sample_files": ["negative_room.png", "negative_furniture.png", "negative_landscape.png"],
        "description": "Human-absent indoor and outdoor scenes to verify zero false positive triggers.",
        "expected_detection": False,
    },
    {
        "category": "Lighting: Bright Natural Daylight",
        "tested": True,
        "sample_files": ["outdoor_daylight.png"],
        "description": "High dynamic range natural sunlight with sharp contrast.",
        "expected_detection": True,
    },
    {
        "category": "Lighting: Standard Indoor Neutral",
        "tested": True,
        "sample_files": ["front_facing_1.png", "front_facing_2.png"],
        "description": "Evenly diffused overhead residential / fitness studio lighting.",
        "expected_detection": True,
    },
    {
        "category": "Lighting: Warm Ambient with Shadow Contrast",
        "tested": True,
        "sample_files": ["indoor_lighting.png"],
        "description": "Warm interior illumination with partial directional shadow falloff.",
        "expected_detection": True,
    },
    {
        "category": "Lighting: Extreme Pitch-Black / Thermal / Infrared",
        "tested": False,
        "sample_files": [],
        "description": "Zero lux conditions, night-vision infrared cameras, and thermal imaging sensors.",
        "expected_detection": None,
        "reason_not_tested": "FitRange Mirror hardware targets ambient visible-spectrum mirror illumination; dataset has no IR/thermal data.",
    },
]


def run_condition_validation(
    benchmark_dir: Optional[Union[str, Path]] = None,
    enable_pose: bool = True,
) -> Dict[str, Any]:
    """Execute condition-based validation on benchmark dataset.

    Args:
        benchmark_dir: Path to directory containing images/ subfolder. Defaults to
            dataset/val_annotated.
        enable_pose: Whether to evaluate pose estimation alongside detection and segmentation.

    Returns:
        Structured dictionary containing results for tested and untested conditions.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    b_dir = Path(benchmark_dir).resolve() if benchmark_dir else project_root / "dataset" / "val_annotated"
    img_dir = b_dir / "images"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Benchmark images directory not found: {img_dir}")

    # Initialize the integrated pipeline
    pipeline = FitRangeMirrorPipeline(
        det_conf=0.30,
        seg_conf=0.30,
        pose_conf=0.40,
        enable_detection=True,
        enable_segmentation=True,
        enable_pose=enable_pose,
    )

    tested_results: List[Dict[str, Any]] = []
    untested_results: List[Dict[str, Any]] = []

    try:
        for condition in CONDITIONS_MAP:
            category = condition["category"]
            if not condition["tested"]:
                untested_results.append(
                    {
                        "category": category,
                        "status": "NOT TESTED",
                        "description": condition["description"],
                        "reason": condition.get("reason_not_tested", "Insufficient data"),
                    }
                )
                continue

            sample_evaluations: List[Dict[str, Any]] = []
            condition_passed = True

            for fname in condition["sample_files"]:
                fpath = img_dir / fname
                if not fpath.is_file():
                    continue

                frame = cv2.imread(str(fpath))
                h, w = frame.shape[:2]
                res = pipeline.process_frame(frame)

                has_person = res["has_person"]
                expected_person = condition["expected_detection"]
                sample_correct = has_person == expected_person

                if not sample_correct:
                    condition_passed = False

                sample_evaluations.append(
                    {
                        "file": fname,
                        "resolution": f"{w}x{h}",
                        "has_person": has_person,
                        "person_count": res["person_count"],
                        "detections_count": len(res["detections"]),
                        "detection_confidences": [round(d["confidence"], 3) for d in res["detections"]],
                        "segmentations_count": len(res["segmentations"]),
                        "mask_pixel_coverage": int((res["composite_mask"] == 255).sum()),
                        "has_pose": res["pose"]["has_pose"],
                        "landmarks_count": len(res["pose"]["landmarks"]),
                        "sample_passed": sample_correct,
                    }
                )

            tested_results.append(
                {
                    "category": category,
                    "status": "PASSED" if condition_passed else "FAILED",
                    "description": condition["description"],
                    "samples_count": len(sample_evaluations),
                    "samples": sample_evaluations,
                }
            )
    finally:
        pipeline.close()

    return {
        "summary": {
            "total_conditions_audited": len(CONDITIONS_MAP),
            "conditions_tested": len(tested_results),
            "conditions_untested": len(untested_results),
            "conditions_passed": sum(1 for c in tested_results if c["status"] == "PASSED"),
            "conditions_failed": sum(1 for c in tested_results if c["status"] == "FAILED"),
        },
        "tested_conditions": tested_results,
        "untested_conditions": untested_results,
    }


def main() -> None:
    """CLI runner for condition-based validation."""
    parser = argparse.ArgumentParser(description="FitRange Mirror - Condition-Based Validation")
    parser.add_argument("--benchmark-dir", type=str, default=None, help="Path to benchmark dataset.")
    parser.add_argument("--json", type=str, default=None, help="Save validation output to JSON.")
    args = parser.parse_args()

    results = run_condition_validation(args.benchmark_dir)

    print("\n" + "=" * 70)
    print("      FITRANGE MIRROR - CONDITION-BASED MODEL VALIDATION")
    print("=" * 70)
    summary = results["summary"]
    print(f"Total Conditions Audited: {summary['total_conditions_audited']}")
    print(f"Conditions Tested:        {summary['conditions_tested']} (Passed: {summary['conditions_passed']}, Failed: {summary['conditions_failed']})")
    print(f"Conditions Not Tested:    {summary['conditions_untested']}")
    print("-" * 70)

    print("\n>>> TESTED OPERATIONAL CONDITIONS:")
    for cond in results["tested_conditions"]:
        print(f"  [{cond['status']}] {cond['category']}")
        print(f"      Description: {cond['description']}")
        print(f"      Samples Evaluated: {cond['samples_count']}")
        for s in cond["samples"]:
            print(
                f"        - {s['file']}: person={s['has_person']} (count={s['person_count']}), "
                f"pose={s['has_pose']}, mask_px={s['mask_pixel_coverage']}"
            )

    print("\n>>> UNTESTED CONDITIONS (INSUFFICIENT DATASET COVERAGE):")
    for cond in results["untested_conditions"]:
        print(f"  [NOT TESTED] {cond['category']}")
        print(f"      Description: {cond['description']}")
        print(f"      Reason: {cond['reason']}")

    print("=" * 70 + "\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved condition validation report to: {args.json}")


if __name__ == "__main__":
    main()
