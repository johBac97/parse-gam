"""Extract cropped doubling cube images from YOLO detections for classification.

Similar to extract_die_crops.py but for the doubling cube.
Crops are sorted manually into folders (1/ 2/ 4/ 8/ 16/ 32/ 64/) for
training a cube value classifier.

YOLO label format: class x_center y_center width height confidence
"""

from pathlib import Path
import argparse

from PIL import Image
from tqdm import tqdm

from parse_gam.models import CLASS_MAPPING


CUBE_CLASS = CLASS_MAPPING["DOUBLING_CUBE"]


def __parse_args():
    parser = argparse.ArgumentParser(
        description="Extract cropped doubling cube images from YOLO detections."
    )
    parser.add_argument("labels", type=Path, help="Directory of YOLO label .txt files")
    parser.add_argument("frames", type=Path, help="Directory of corresponding frame images")
    parser.add_argument("output", type=Path, help="Output directory for cube crops")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.2,
        help="Fractional padding around cube bounding box (default: 0.2)",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.3,
        help="Minimum confidence to extract a cube crop (default: 0.3)",
    )
    return parser.parse_args()


def extract_cube_bboxes(label_path: Path, min_conf: float = 0.3):
    """Parse a YOLO label file and return doubling cube bounding boxes."""
    bboxes = []
    with label_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls != CUBE_CLASS:
                continue
            x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            conf = float(parts[5]) if len(parts) > 5 else 1.0
            if conf >= min_conf:
                bboxes.append((x_c, y_c, w, h, conf))
    return bboxes


def crop_bbox(image: Image.Image, bbox: tuple, padding: float = 0.2) -> Image.Image:
    """Crop a region from an image given its normalized bounding box."""
    img_w, img_h = image.size
    x_c, y_c, w, h, _ = bbox

    px_x = x_c * img_w
    px_y = y_c * img_h
    px_w = w * img_w
    px_h = h * img_h

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

    for label_path in tqdm(label_files, desc="Extracting cube crops"):
        bboxes = extract_cube_bboxes(label_path, min_conf=args.min_conf)
        if not bboxes:
            continue

        frame_path = args.frames / label_path.with_suffix(".jpg").name
        if not frame_path.exists():
            frame_path = args.frames / label_path.with_suffix(".png").name
        if not frame_path.exists():
            continue

        image = Image.open(frame_path)

        for idx, bbox in enumerate(bboxes):
            crop = crop_bbox(image, bbox, padding=args.padding)
            crop_name = f"{label_path.stem}_cube{idx}.jpg"
            crop.save(args.output / crop_name)
            total_crops += 1

    print(f"Extracted {total_crops} cube crops to {args.output}")


if __name__ == "__main__":
    main()
