import json
import math
import re
import sys
import zlib

sys.dont_write_bytecode = True

INPUT_PDF = "Тимково_Мастер_план_С_Х_без_топо_1.pdf"
OUTPUT_SVG = "pdf_vector_lines.svg"
OUTPUT_JSON = "pdf_vector_lines.json"
OUTPUT_STYLE_SVG_PREFIX = "pdf_vector_lines_style"
OUTPUT_BOUNDARY_SVG = "pdf_vector_boundary_candidate.svg"
OUTPUT_BOUNDARY_JSON = "pdf_vector_boundary_candidate.json"
OUTPUT_ROAD_SVG = "pdf_vector_road_candidate.svg"
OUTPUT_ROAD_JSON = "pdf_vector_road_candidate.json"
OUTPUT_BOUNDARY_NO_ROADS_SVG = "pdf_vector_boundary_no_roads.svg"
OUTPUT_BOUNDARY_NO_ROADS_JSON = "pdf_vector_boundary_no_roads.json"

SVG_STROKE_COLOR = "#ff0000"
SVG_LINE_WIDTH = 1

# Keep this extractor diagnostic and conservative. It reads vector strokes from
# the PDF without trying to polygonize them yet.
MIN_SEGMENT_LENGTH = 2.0
KEEP_GRAY_STROKES_ONLY = True
MAX_GRAY_VALUE = 0.85
MAX_GRAY_DELTA = 0.08
BOUNDARY_MIN_SEGMENT_LENGTH = 20.0
BOUNDARY_STYLE_CANDIDATES = {
    ((0.0, 0.0, 0.0), 0.0),
}
ROAD_PAIR_MIN_LENGTH = 70.0
ROAD_PAIR_MIN_DISTANCE = 6.0
ROAD_PAIR_MAX_DISTANCE = 24.0
ROAD_PAIR_MAX_ANGLE_DEGREES = 4.0
ROAD_PAIR_MIN_OVERLAP_RATIO = 0.55


NUMBER_RE = re.compile(rb"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")
STREAM_RE = re.compile(rb"<<(?P<dict>.*?)>>\s*stream\r?\n(?P<data>.*?)\r?\nendstream", re.S)
TOKEN_RE = re.compile(
    r"""
    /[^\s<>\[\]\(\)]+
    |[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?
    |\[[^\]]*\]
    |\([^\)]*\)
    |[A-Za-z\*'"]+
    |.
    """,
    re.X,
)


def read_pdf(path):
    with open(path, "rb") as f:
        return f.read()


def parse_media_box(pdf_bytes):
    match = re.search(rb"/MediaBox\s*\[\s*(.*?)\s*\]", pdf_bytes, re.S)
    if not match:
        return (0.0, 0.0, 1.0, 1.0)
    nums = [float(x) for x in NUMBER_RE.findall(match.group(1))]
    if len(nums) >= 4:
        return tuple(nums[:4])
    return (0.0, 0.0, 1.0, 1.0)


def decode_streams(pdf_bytes):
    streams = []
    for match in STREAM_RE.finditer(pdf_bytes):
        stream_dict = match.group("dict")
        data = match.group("data")
        if b"/FlateDecode" in stream_dict:
            try:
                data = zlib.decompress(data)
            except zlib.error:
                continue
        elif b"/Filter" in stream_dict:
            continue
        try:
            text = data.decode("latin1")
        except UnicodeDecodeError:
            continue
        streams.append(text)
    return streams


def is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def matrix_multiply(m1, m2):
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def transform_point(matrix, x, y):
    a, b, c, d, e, f = matrix
    return (a * x + c * y + e, b * x + d * y + f)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def is_gray_stroke(color):
    r, g, b = color
    return max(r, g, b) <= MAX_GRAY_VALUE and max(r, g, b) - min(r, g, b) <= MAX_GRAY_DELTA


def path_segments(path, close_open=False):
    segments = []
    for subpath in path:
        if len(subpath) < 2:
            continue
        for i in range(1, len(subpath)):
            a = subpath[i - 1]
            b = subpath[i]
            if distance(a, b) >= MIN_SEGMENT_LENGTH:
                segments.append((a, b))
        if close_open and len(subpath) > 2:
            a = subpath[-1]
            b = subpath[0]
            if distance(a, b) >= MIN_SEGMENT_LENGTH:
                segments.append((a, b))
    return segments


def extract_segments_from_stream(text):
    tokens = [m.group(0) for m in TOKEN_RE.finditer(text) if m.group(0).strip()]
    stack = []
    state = {
        "ctm": (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        "stroke": (0.0, 0.0, 0.0),
        "line_width": 1.0,
    }
    state_stack = []
    path = []
    current_subpath = None
    in_text = False
    segments = []

    def nums(count):
        if len(stack) < count:
            return None
        values = stack[-count:]
        if not all(is_number(v) for v in values):
            return None
        return [float(v) for v in values]

    def reset_path():
        nonlocal path, current_subpath
        path = []
        current_subpath = None

    def stroke_path(close_open=False):
        nonlocal segments
        if KEEP_GRAY_STROKES_ONLY and not is_gray_stroke(state["stroke"]):
            reset_path()
            return
        for a, b in path_segments(path, close_open=close_open):
            segments.append(
                {
                    "a": [round(a[0], 3), round(a[1], 3)],
                    "b": [round(b[0], 3), round(b[1], 3)],
                    "stroke": [round(v, 3) for v in state["stroke"]],
                    "line_width": round(state["line_width"], 3),
                }
            )
        reset_path()

    operators = {
        "q",
        "Q",
        "cm",
        "w",
        "RG",
        "G",
        "m",
        "l",
        "c",
        "v",
        "y",
        "h",
        "re",
        "S",
        "s",
        "B",
        "B*",
        "b",
        "b*",
        "n",
        "BT",
        "ET",
    }

    for token in tokens:
        if token not in operators:
            stack.append(token)
            continue

        if token == "BT":
            in_text = True
            stack.clear()
            continue
        if token == "ET":
            in_text = False
            stack.clear()
            continue
        if in_text:
            stack.clear()
            continue

        if token == "q":
            state_stack.append(dict(state))
        elif token == "Q":
            if state_stack:
                state = state_stack.pop()
        elif token == "cm":
            values = nums(6)
            if values:
                state["ctm"] = matrix_multiply(state["ctm"], tuple(values))
        elif token == "w":
            values = nums(1)
            if values:
                state["line_width"] = values[0]
        elif token == "RG":
            values = nums(3)
            if values:
                state["stroke"] = tuple(values)
        elif token == "G":
            values = nums(1)
            if values:
                state["stroke"] = (values[0], values[0], values[0])
        elif token == "m":
            values = nums(2)
            if values:
                current_subpath = [transform_point(state["ctm"], values[0], values[1])]
                path.append(current_subpath)
        elif token == "l":
            values = nums(2)
            if values and current_subpath is not None:
                current_subpath.append(transform_point(state["ctm"], values[0], values[1]))
        elif token in {"c", "v", "y"}:
            values = nums(6 if token == "c" else 4)
            if values and current_subpath is not None:
                current_subpath.append(transform_point(state["ctm"], values[-2], values[-1]))
        elif token == "h":
            if current_subpath and len(current_subpath) > 2:
                current_subpath.append(current_subpath[0])
        elif token == "re":
            values = nums(4)
            if values:
                x, y, w, h = values
                rect = [
                    transform_point(state["ctm"], x, y),
                    transform_point(state["ctm"], x + w, y),
                    transform_point(state["ctm"], x + w, y + h),
                    transform_point(state["ctm"], x, y + h),
                    transform_point(state["ctm"], x, y),
                ]
                path.append(rect)
                current_subpath = rect
        elif token in {"S", "B", "B*"}:
            stroke_path(close_open=False)
        elif token in {"s", "b", "b*"}:
            stroke_path(close_open=True)
        elif token == "n":
            reset_path()

        stack.clear()

    return segments


def extract_segments(pdf_path):
    pdf_bytes = read_pdf(pdf_path)
    media_box = parse_media_box(pdf_bytes)
    # Page /Contents may be an array of streams. PDF readers treat them as one
    # concatenated content stream, so graphic state and CTM can continue from one
    # stream to the next.
    content = "\n".join(decode_streams(pdf_bytes))
    segments = extract_segments_from_stream(content)
    return media_box, segments


def segments_bbox(segments, fallback_box):
    if not segments:
        return fallback_box
    xs = []
    ys = []
    for segment in segments:
        x1, y1 = segment["a"]
        x2, y2 = segment["b"]
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    return (min(xs), min(ys), max(xs), max(ys))


def write_svg(path, media_box, segments):
    min_x, min_y, max_x, max_y = segments_bbox(segments, media_box)
    width = max_x - min_x
    height = max_y - min_y
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width:.3f} {height:.3f}" '
            f'width="{width:.3f}" height="{height:.3f}">\n'
        )
        f.write('<g fill="none" stroke-linecap="round" stroke-linejoin="round">\n')
        for segment in segments:
            x1, y1 = segment["a"]
            x2, y2 = segment["b"]
            sx1 = x1 - min_x
            sy1 = max_y - y1
            sx2 = x2 - min_x
            sy2 = max_y - y2
            f.write(
                f'<path d="M {sx1:.3f} {sy1:.3f} L {sx2:.3f} {sy2:.3f}" '
                f'stroke="{SVG_STROKE_COLOR}" stroke-width="{SVG_LINE_WIDTH}"/>\n'
            )
        f.write("</g>\n</svg>\n")


def write_style_svgs(segments, media_box):
    groups = {}
    for segment in segments:
        key = (tuple(segment["stroke"]), segment["line_width"])
        groups.setdefault(key, []).append(segment)

    for index, (key, group_segments) in enumerate(
        sorted(groups.items(), key=lambda item: len(item[1]), reverse=True), start=1
    ):
        if index > 9:
            break
        stroke, line_width = key
        suffix = (
            f"{index:02d}_rgb_{stroke[0]:.3f}_{stroke[1]:.3f}_{stroke[2]:.3f}"
            f"_w_{line_width:.3f}"
        ).replace(".", "p")
        write_svg(f"{OUTPUT_STYLE_SVG_PREFIX}_{suffix}.svg", media_box, group_segments)


def segment_length(segment):
    x1, y1 = segment["a"]
    x2, y2 = segment["b"]
    return math.hypot(x2 - x1, y2 - y1)


def segment_geometry(segment):
    x1, y1 = segment["a"]
    x2, y2 = segment["b"]
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return None
    ux = dx / length
    uy = dy / length
    nx = -uy
    ny = ux
    return {
        "point": (x1, y1),
        "unit": (ux, uy),
        "normal": (nx, ny),
        "length": length,
        "projection": (0.0, length),
    }


def angle_between_units(a, b):
    dot = abs(a[0] * b[0] + a[1] * b[1])
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(math.acos(dot))


def road_candidate_indexes(segments):
    geometries = [segment_geometry(segment) for segment in segments]
    candidates = [
        index
        for index, geometry in enumerate(geometries)
        if geometry and geometry["length"] >= ROAD_PAIR_MIN_LENGTH
    ]
    road_indexes = set()

    for pos, i in enumerate(candidates):
        gi = geometries[i]
        p_i = gi["point"]
        ux, uy = gi["unit"]
        nx, ny = gi["normal"]
        len_i = gi["length"]

        for j in candidates[pos + 1 :]:
            gj = geometries[j]
            if angle_between_units(gi["unit"], gj["unit"]) > ROAD_PAIR_MAX_ANGLE_DEGREES:
                continue

            x1, y1 = segments[j]["a"]
            x2, y2 = segments[j]["b"]
            cross_dist_1 = (x1 - p_i[0]) * nx + (y1 - p_i[1]) * ny
            cross_dist_2 = (x2 - p_i[0]) * nx + (y2 - p_i[1]) * ny
            distance_mean = abs((cross_dist_1 + cross_dist_2) * 0.5)
            distance_delta = abs(cross_dist_1 - cross_dist_2)
            if not (ROAD_PAIR_MIN_DISTANCE <= distance_mean <= ROAD_PAIR_MAX_DISTANCE):
                continue
            if distance_delta > ROAD_PAIR_MAX_DISTANCE * 0.35:
                continue

            proj_1 = (x1 - p_i[0]) * ux + (y1 - p_i[1]) * uy
            proj_2 = (x2 - p_i[0]) * ux + (y2 - p_i[1]) * uy
            start = max(0.0, min(proj_1, proj_2))
            end = min(len_i, max(proj_1, proj_2))
            overlap = max(0.0, end - start)
            min_length = min(len_i, gj["length"])
            if min_length == 0:
                continue
            if overlap / min_length < ROAD_PAIR_MIN_OVERLAP_RATIO:
                continue

            road_indexes.add(i)
            road_indexes.add(j)

    return road_indexes


def boundary_candidate_segments(segments):
    result = []
    for segment in segments:
        key = (tuple(segment["stroke"]), segment["line_width"])
        if key not in BOUNDARY_STYLE_CANDIDATES:
            continue
        if segment_length(segment) < BOUNDARY_MIN_SEGMENT_LENGTH:
            continue
        result.append(segment)
    return result


def without_indexes(items, indexes):
    return [item for index, item in enumerate(items) if index not in indexes]


def only_indexes(items, indexes):
    return [item for index, item in enumerate(items) if index in indexes]


def main():
    media_box, segments = extract_segments(INPUT_PDF)
    bbox = segments_bbox(segments, media_box)
    boundary_segments = boundary_candidate_segments(segments)
    road_indexes = road_candidate_indexes(boundary_segments)
    road_segments = only_indexes(boundary_segments, road_indexes)
    boundary_no_roads_segments = without_indexes(boundary_segments, road_indexes)
    payload = {
        "source": INPUT_PDF,
        "media_box": media_box,
        "segments_bbox": bbox,
        "segments_count": len(segments),
        "segments": segments,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_BOUNDARY_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": INPUT_PDF,
                "media_box": media_box,
                "segments_bbox": segments_bbox(boundary_segments, media_box),
                "segments_count": len(boundary_segments),
                "segments": boundary_segments,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(OUTPUT_ROAD_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": INPUT_PDF,
                "media_box": media_box,
                "segments_bbox": segments_bbox(road_segments, media_box),
                "segments_count": len(road_segments),
                "segments": road_segments,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(OUTPUT_BOUNDARY_NO_ROADS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": INPUT_PDF,
                "media_box": media_box,
                "segments_bbox": segments_bbox(boundary_no_roads_segments, media_box),
                "segments_count": len(boundary_no_roads_segments),
                "segments": boundary_no_roads_segments,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    write_svg(OUTPUT_SVG, media_box, segments)
    write_svg(OUTPUT_BOUNDARY_SVG, media_box, boundary_segments)
    write_svg(OUTPUT_ROAD_SVG, media_box, road_segments)
    write_svg(OUTPUT_BOUNDARY_NO_ROADS_SVG, media_box, boundary_no_roads_segments)
    write_style_svgs(segments, media_box)
    print(f"PDF media box: {media_box}")
    print(f"Segments bbox: {bbox}")
    print(f"Vector segments: {len(segments)}")
    print(f"Boundary candidate segments: {len(boundary_segments)}")
    print(f"Road candidate segments: {len(road_segments)}")
    print(f"Boundary without roads segments: {len(boundary_no_roads_segments)}")
    print(f"SVG saved: {OUTPUT_SVG}")
    print(f"Boundary candidate SVG saved: {OUTPUT_BOUNDARY_SVG}")
    print(f"JSON saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
