import sys

sys.dont_write_bytecode = True

import argparse
import json
import re
from decimal import Decimal
from html import escape
from pathlib import Path

import pdfplumber


DEFAULT_DPI = 800
PDF_POINTS_PER_INCH = 72
SUPPORTED_STROKE_COLORS = (
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
)
SIZE_LABEL_PATTERN = re.compile(r"^\d{1,2}[,.]\d{1,2}$")
STAGE1_FIELD_ORDER = (
    "adres",
    "id",
    "idtur",
    "kadastr",
    "kadastrurl",
    "names",
    "number",
    "price",
    "size",
    "status",
    "coordinates",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract vector parcel contours and their area labels directly from a PDF."
        )
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        help="Input PDF. If omitted, the first PDF in the current directory is used.",
    )
    parser.add_argument("--page", type=int, default=1, help="One-based page number.")
    parser.add_argument(
        "--dpi",
        type=float,
        default=DEFAULT_DPI,
        help="Coordinate scale used for compatibility with the raster pipeline.",
    )
    parser.add_argument(
        "--output",
        default="polygon_from_pdf.json",
        help="Stage 1 compatible JSON output.",
    )
    parser.add_argument(
        "--raw-output",
        default="pdf_parcels_raw.json",
        help="Diagnostic JSON with source labels and PDF coordinates.",
    )
    parser.add_argument(
        "--preview",
        default="pdf_parcels_preview.svg",
        help="SVG preview of extracted parcels.",
    )
    return parser.parse_args()


def find_pdf(explicit_path):
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        return path

    pdf_files = sorted(
        path for path in Path.cwd().glob("*.pdf") if not path.name.startswith("__")
    )
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the current directory.")
    return pdf_files[0]


def colors_are_close(first, second, tolerance=0.02):
    return len(first) == len(second) and all(
        abs(float(a) - float(b)) <= tolerance for a, b in zip(first, second)
    )


def is_supported_color(color):
    if not isinstance(color, (list, tuple)):
        return False
    return any(colors_are_close(color, expected) for expected in SUPPORTED_STROKE_COLORS)


def close_polygon(points):
    result = [(float(x), float(y)) for x, y in points]
    if result and result[0] != result[-1]:
        result.append(result[0])
    return result


def polygon_area(points):
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:])
        )
    ) / 2


def polygon_bbox(points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_polygon(x, y, polygon):
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:]):
        crosses_horizontal_ray = (y1 > y) != (y2 > y)
        if not crosses_horizontal_ray:
            continue
        crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        if x < crossing_x:
            inside = not inside
    return inside


def extract_candidate_contours(page):
    candidates = []
    for source_index, curve in enumerate(page.curves):
        if not curve.get("stroke") or not is_supported_color(curve.get("stroking_color")):
            continue

        points = close_polygon(curve.get("pts") or [])
        if len(points) < 4:
            continue

        area = polygon_area(points)
        if area <= 0:
            continue

        candidates.append(
            {
                "source_index": source_index,
                "points": points,
                "bbox": polygon_bbox(points),
                "area": area,
                "stroke_color": [float(value) for value in curve["stroking_color"]],
                "characters": [],
            }
        )
    return candidates


def assign_characters_to_contours(page, candidates):
    for character in page.chars:
        x = (float(character["x0"]) + float(character["x1"])) / 2
        y = (float(character["top"]) + float(character["bottom"])) / 2
        containing = []

        for candidate in candidates:
            x0, y0, x1, y1 = candidate["bbox"]
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
            if point_in_polygon(x, y, candidate["points"]):
                containing.append(candidate)

        if containing:
            smallest = min(containing, key=lambda item: item["area"])
            smallest["characters"].append(str(character.get("text") or ""))


def normalize_size_label(characters):
    label = "".join(characters).replace(" ", "").strip()
    if not SIZE_LABEL_PATTERN.fullmatch(label):
        return None
    return label.replace(".", ",")


def sotkas_to_hectares(label):
    hectares = Decimal(label.replace(",", ".")) / Decimal("100")
    return format(hectares.normalize(), "f")


def extract_parcels(page):
    candidates = extract_candidate_contours(page)
    assign_characters_to_contours(page, candidates)

    parcels = []
    rejected = []
    for candidate in candidates:
        source_text = "".join(candidate["characters"]).replace(" ", "").strip()
        size_label = normalize_size_label(candidate["characters"])
        if size_label is None:
            rejected.append(
                {
                    "source_index": candidate["source_index"],
                    "source_text": source_text,
                    "area_pdf_points": round(candidate["area"], 6),
                }
            )
            continue

        parcels.append(
            {
                "source_index": candidate["source_index"],
                "points": candidate["points"],
                "bbox": candidate["bbox"],
                "stroke_color": candidate["stroke_color"],
                "source_size_sotkas": size_label,
                "size_hectares": sotkas_to_hectares(size_label),
            }
        )

    return parcels, rejected


def scaled_points(points, scale):
    return [(x * scale, y * scale) for x, y in points]


def stage1_coordinates(points, page_width, page_height, scale):
    center_x = page_width * scale / 2
    center_y = page_height * scale / 2
    transformed = []
    for x, y in scaled_points(points, scale):
        shifted_x = x - center_x
        shifted_y = y - center_y
        transformed.append([round(-shifted_y, 6), round(shifted_x, 6)])
    return transformed


def make_stage1_payload(parcels, page_width, page_height, dpi):
    scale = dpi / PDF_POINTS_PER_INCH
    data = []
    for index, parcel in enumerate(parcels, start=1):
        number = str(index)
        idtur = number.zfill(5)
        item = {
            "adres": "",
            "id": index,
            "idtur": idtur,
            "kadastr": "",
            "kadastrurl": "",
            "names": idtur,
            "number": number,
            "price": "",
            "size": parcel["size_hectares"],
            "status": "sale",
            "coordinates": [
                stage1_coordinates(
                    parcel["points"], page_width, page_height, scale
                )
            ],
        }
        data.append({key: item[key] for key in STAGE1_FIELD_ORDER})
    return {"inc": len(data), "data": data}


def make_raw_payload(pdf_path, page_number, page, dpi, parcels, rejected):
    scale = dpi / PDF_POINTS_PER_INCH
    raw_parcels = []
    for index, parcel in enumerate(parcels, start=1):
        raw_parcels.append(
            {
                "id": index,
                "source_index": parcel["source_index"],
                "source_size_sotkas": parcel["source_size_sotkas"],
                "size_hectares": parcel["size_hectares"],
                "stroke_color_rgb": parcel["stroke_color"],
                "coordinates_pdf_points": [
                    [round(x, 6), round(y, 6)] for x, y in parcel["points"]
                ],
                "coordinates_scaled": [
                    [round(x, 6), round(y, 6)]
                    for x, y in scaled_points(parcel["points"], scale)
                ],
            }
        )

    return {
        "source_pdf": str(pdf_path),
        "page": page_number,
        "page_width_pdf_points": float(page.width),
        "page_height_pdf_points": float(page.height),
        "coordinate_dpi": dpi,
        "parcel_count": len(raw_parcels),
        "rejected_candidate_count": len(rejected),
        "rejected_candidates": rejected,
        "parcels": raw_parcels,
    }


def rgb_to_hex(color):
    channels = [max(0, min(255, round(value * 255))) for value in color[:3]]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def polygon_centroid(points):
    open_points = points[:-1] if points and points[0] == points[-1] else points
    return (
        sum(point[0] for point in open_points) / len(open_points),
        sum(point[1] for point in open_points) / len(open_points),
    )


def save_preview_svg(path, parcels, page_width, page_height, dpi):
    scale = dpi / PDF_POINTS_PER_INCH
    width = page_width * scale
    height = page_height * scale
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.3f}" '
            f'height="{height:.3f}" viewBox="0 0 {width:.3f} {height:.3f}">'
        ),
        '  <rect width="100%" height="100%" fill="white"/>',
        '  <g fill="none" stroke-linejoin="round">',
    ]

    for index, parcel in enumerate(parcels, start=1):
        points = scaled_points(parcel["points"], scale)
        point_text = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
        color = rgb_to_hex(parcel["stroke_color"])
        lines.append(
            f'    <polygon id="parcel_{index:05d}" points="{point_text}" '
            f'stroke="{color}" stroke-width="{max(1.0, scale * 0.84):.3f}"/>'
        )

    lines.append("  </g>")
    lines.append(
        '  <g fill="#202020" font-family="Arial, sans-serif" '
        f'font-size="{5.2 * scale:.3f}" text-anchor="middle" '
        'dominant-baseline="central">'
    )
    for index, parcel in enumerate(parcels, start=1):
        center_x, center_y = polygon_centroid(parcel["points"])
        label = escape(parcel["source_size_sotkas"])
        lines.append(
            f'    <text x="{center_x * scale:.3f}" y="{center_y * scale:.3f}" '
            f'data-size-ha="{escape(parcel["size_hectares"])}">{label}</text>'
        )
    lines.extend(["  </g>", "</svg>"])
    write_text(path, "\n".join(lines) + "\n")


def write_json(path, payload):
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_text(path, text):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    args = parse_args()
    pdf_path = find_pdf(args.pdf)
    if args.page < 1:
        raise ValueError("Page number must be at least 1.")
    if args.dpi <= 0:
        raise ValueError("DPI must be greater than zero.")

    with pdfplumber.open(pdf_path) as document:
        if args.page > len(document.pages):
            raise ValueError(
                f"Page {args.page} does not exist; PDF has {len(document.pages)} page(s)."
            )
        page = document.pages[args.page - 1]
        parcels, rejected = extract_parcels(page)

        if not parcels:
            raise RuntimeError(
                "No parcel contours with decimal sotka labels were found on the page."
            )

        stage1_payload = make_stage1_payload(
            parcels, float(page.width), float(page.height), args.dpi
        )
        raw_payload = make_raw_payload(
            pdf_path, args.page, page, args.dpi, parcels, rejected
        )
        save_preview_svg(
            args.preview, parcels, float(page.width), float(page.height), args.dpi
        )

    write_json(args.output, stage1_payload)
    write_json(args.raw_output, raw_payload)

    print(f"PDF: {pdf_path}")
    print(f"Page: {args.page}")
    print(f"Parcels extracted: {len(parcels)}")
    print(f"Rejected colored contours: {len(rejected)}")
    for item in rejected:
        text = item["source_text"] or "<empty>"
        print(f"  Rejected source contour {item['source_index']}: {text}")
    print(f"Stage 1 JSON: {args.output}")
    print(f"Raw JSON: {args.raw_output}")
    print(f"Preview SVG: {args.preview}")


if __name__ == "__main__":
    main()
