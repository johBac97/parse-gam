"""Extract cropped die images from YOLO detections for classification.

Two uses:
  1. Collect training data: run on existing inference results, then manually
     label the crops into folders (1/ 2/ 3/ 4/ 5/ 6/) for classifier training.
  2. Inference: crop dice, run classifier, inject values back into BoardState.

YOLO label format: class x_center y_center width height confidence
All coordinates are normalized [0, 1] relative to image dimensions.
"""

from pathlib import Path
import argparse

from PIL import Image
from tqdm import tqdm

from parse_gam.models import CLASS_MAPPING


DIE_CLASS = CLASS_MAPPING["DIE"]


def __parse_args():
    parser = argparse.ArgumentParser(
        description="Extract cropped die images from YOLO detections."
    )
    parser.add_argument("labels", type=Path, help="Directory of YOLO label .txt files")
    parser.add_argument("frames", type=Path, help="Directory of corresponding frame images")
    parser.add_argument("output", type=Path, help="Output directory for die crops")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Fractional padding around die bounding box (default: 0.15)",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.3,
        help="Minimum confidence to extract a die crop (default: 0.3)",
    )
    return parser.parse_args()


def extract_die_bboxes(label_path: Path, min_conf: float = 0.3):
    """Parse a YOLO label file and return die bounding boxes.

    Returns list of (x_center, y_center, width, height, confidence) tuples,
    all in normalized [0,1] coordinates.
    """
    bboxes = []
    with label_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls != DIE_CLASS:
                continue
            x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            conf = float(parts[5]) if len(parts) > 5 else 1.0
            if conf >= min_conf:
                bboxes.append((x_c, y_c, w, h, conf))
    return bboxes


def crop_die(image: Image.Image, bbox: tuple, padding: float = 0.15) -> Image.Image:
    """Crop a die from an image given its normalized bounding box.

    Adds `padding` fraction of the bbox size around the crop.
    """
    img_w, img_h = image.size
    x_c, y_c, w, h, _ = bbox

    # Convert normalized coords to pixels
    px_x = x_c * img_w
    px_y = y_c * img_h
    px_w = w * img_w
    px_h = h * img_h

    # Add padding
    pad_x = px_w * padding
    pad_y = px_h * padding

    left = max(0, px_x - px_w / 2 - pad_x)
    top = max(0, px_y - px_h / 2 - pad_y)
    right = min(img_w, px_x + px_w / 2 + pad_x)
    bottom = min(img_h, px_y + px_h / 2 + pad_y)

    return image.crop((int(left), int(top), int(right), int(bottom)))


def main():
    args = __parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    label_files = sorted(args.labels.glob("*.txt"))
    total_crops = 0

    for label_path in tqdm(label_files, desc="Extracting die crops"):
        bboxes = extract_die_bboxes(label_path, min_conf=args.min_conf)
        if not bboxes:
            continue

        # Find corresponding frame image
        frame_path = args.frames / label_path.with_suffix(".jpg").name
        if not frame_path.exists():
            frame_path = args.frames / label_path.with_suffix(".png").name
        if not frame_path.exists():
            continue

        image = Image.open(frame_path)

        for die_idx, bbox in enumerate(bboxes):
            crop = crop_die(image, bbox, padding=args.padding)
            crop_name = f"{label_path.stem}_die{die_idx}.jpg"
            crop.save(args.output / crop_name)
            total_crops += 1

    print(f"Extracted {total_crops} die crops to {args.output}")


if __name__ == "__main__":
    main()
