
import cv2
import json
import math
import os
import matplotlib.pyplot as plt
import numpy as np

# === РќРђРЎРўР РћР™РљР ===
INPUT_IMAGE = "page_1.png"
OUTPUT_IMAGE = "output_detected.png"
OUTPUT_JSON = "polygon.json"
OUTPUT_RAW_JSON = "detected_polygons.json"
OUTPUT_SVG = "polygon_grid.svg"
MIN_AREA = 5000
APPROX_EPSILON_COEFF = 0.002
MIN_AREA_RATIO = 0.0009
MAX_AREA_RATIO = 0.04
PAGE_MARGIN_RATIO = 0.003
TOP_DECORATION_Y_RATIO = 0.12
LEGEND_X_RATIO = 0.62
LEGEND_Y_RATIO = 0.28
MIN_NOISE_EXTENT = 0.08
MIN_NOISE_SOLIDITY = 0.20
MERGED_CONTOUR_AREA_RATIO = 0.01
MAX_MERGED_CONTOUR_EXTENT = 0.35
POLYGON_ALPHA = 0.35
POLYGON_LINE_WIDTH = 0.6
PREVIEW_DPI = 600
APPLY_ROTATE_AND_MIRROR = True
SVG_STROKE_COLOR = "#ff0000"
SVG_LINE_WIDTH = 4
SNAP_TOLERANCE = 12
EDGE_SNAP_TOLERANCE = 12
HULL_RECOVERY_EXTENT = 0.35
HULL_RECOVERY_SOLIDITY = 0.50
HULL_RECOVERY_AREA_RATIO = 0.003
HATCH_ANGLE_DEGREES = 137.5
HATCH_KERNEL_SIZE = 31
HATCH_KERNEL_LENGTH = 25
HATCH_GRAY_MIN = 120
HATCH_GRAY_MAX = 210
GRAY_FILL_SAT_MAX = 80
GRAY_FILL_MIN = 125
GRAY_FILL_MAX = 245
LOWER_GRAY_RECOVERY_MIN_Y_RATIO = 0.55
LOWER_GRAY_RECOVERY_MAX_CENTER_X_RATIO = 0.55
LOWER_GRAY_RECOVERY_MIN_WIDTH_RATIO = 0.25
LOWER_GRAY_RECOVERY_MIN_HEIGHT_RATIO = 0.15
LOWER_GRAY_RECOVERY_MIN_SOLIDITY = 0.45
OUTER_SHELL_AREA_RATIO = 0.01
OUTER_SHELL_CHILD_AREA_RATIO = 0.002
OUTER_SHELL_MIN_CHILDREN = 3
DUPLICATE_BBOX_IOU = 0.88
DUPLICATE_AREA_RATIO = 0.80
NESTED_FRAGMENT_PARENT_AREA_RATIO = 0.008
NESTED_FRAGMENT_MAX_AREA_RATIO = 0.30
NESTED_FRAGMENT_MIN_CENTER_X_RATIO = 0.20
NESTED_FRAGMENT_MAX_CENTER_X_RATIO = 0.38
NESTED_FRAGMENT_MIN_CENTER_Y_RATIO = 0.56
NESTED_FRAGMENT_MAX_CENTER_Y_RATIO = 0.68


def is_candidate_contour(contour, area, image_width, image_height):
    x, y, width, height = cv2.boundingRect(contour)
    center_x = x + width / 2
    center_y = y + height / 2
    image_area = image_width * image_height
    min_area = max(MIN_AREA, image_area * MIN_AREA_RATIO)
    max_area = image_area * MAX_AREA_RATIO
    margin = max(20, int(min(image_width, image_height) * PAGE_MARGIN_RATIO))

    if area < min_area or area > max_area:
        return False
    if x <= margin or y <= margin:
        return False
    if x + width >= image_width - margin or y + height >= image_height - margin:
        return False
    if width < 40 or height < 40:
        return False
    if center_y < image_height * TOP_DECORATION_Y_RATIO:
        return False
    if center_x > image_width * LEGEND_X_RATIO and center_y < image_height * LEGEND_Y_RATIO:
        return False

    rect_area = width * height
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    extent = area / rect_area if rect_area else 0
    solidity = area / hull_area if hull_area else 0
    if area > image_area * MERGED_CONTOUR_AREA_RATIO and extent < MAX_MERGED_CONTOUR_EXTENT:
        return False
    if extent < MIN_NOISE_EXTENT or solidity < MIN_NOISE_SOLIDITY:
        return False

    return True


def contour_for_polygon(contour, area, image_width, image_height):
    x, y, width, height = cv2.boundingRect(contour)
    rect_area = width * height
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    image_area = image_width * image_height
    extent = area / rect_area if rect_area else 0
    solidity = area / hull_area if hull_area else 0

    if (
        area < image_area * HULL_RECOVERY_AREA_RATIO
        and extent < HULL_RECOVERY_EXTENT
        and solidity > HULL_RECOVERY_SOLIDITY
    ):
        return hull

    return contour


def is_recoverable_lower_gray_massif(contour, area, image_width, image_height):
    x, y, width, height = cv2.boundingRect(contour)
    center_x = x + width / 2
    image_area = image_width * image_height
    rect_area = width * height
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    extent = area / rect_area if rect_area else 0
    solidity = area / hull_area if hull_area else 0

    return (
        image_area * MERGED_CONTOUR_AREA_RATIO < area < image_area * MAX_AREA_RATIO
        and extent < MAX_MERGED_CONTOUR_EXTENT
        and solidity > LOWER_GRAY_RECOVERY_MIN_SOLIDITY
        and y > image_height * LOWER_GRAY_RECOVERY_MIN_Y_RATIO
        and center_x < image_width * LOWER_GRAY_RECOVERY_MAX_CENTER_X_RATIO
        and width > image_width * LOWER_GRAY_RECOVERY_MIN_WIDTH_RATIO
        and height > image_height * LOWER_GRAY_RECOVERY_MIN_HEIGHT_RATIO
    )


def make_line_kernel(size, length, angle_degrees):
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    angle = math.radians(angle_degrees)
    x1 = int(center - math.cos(angle) * length / 2)
    y1 = int(center - math.sin(angle) * length / 2)
    x2 = int(center + math.cos(angle) * length / 2)
    y2 = int(center + math.sin(angle) * length / 2)
    cv2.line(kernel, (x1, y1), (x2, y2), 1, 1)
    return kernel


def remove_hatching(thresh, gray):
    kernel = make_line_kernel(HATCH_KERNEL_SIZE, HATCH_KERNEL_LENGTH, HATCH_ANGLE_DEGREES)
    weak_line_mask = (
        ((gray >= HATCH_GRAY_MIN) & (gray <= HATCH_GRAY_MAX)).astype(np.uint8) * 255
    )
    hatch_mask = cv2.morphologyEx(weak_line_mask, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(thresh, hatch_mask)


def remove_gray_fill(thresh, img, gray):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    fill_mask = (
        (hsv[:, :, 1] < GRAY_FILL_SAT_MAX)
        & (gray > GRAY_FILL_MIN)
        & (gray < GRAY_FILL_MAX)
    ).astype(np.uint8) * 255
    return cv2.subtract(thresh, fill_mask)


def polygon_bbox(polygon):
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0

    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0


def is_outer_shell(item, items, image_area):
    area, polygon = item
    if area < image_area * OUTER_SHELL_AREA_RATIO:
        return False

    contour = np.array(polygon, dtype=np.int32)
    children = 0
    for child_area, child_polygon in items:
        if child_area >= area or child_area < image_area * OUTER_SHELL_CHILD_AREA_RATIO:
            continue

        bbox = polygon_bbox(child_polygon)
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        if cv2.pointPolygonTest(contour, center, False) >= 0:
            children += 1
            if children >= OUTER_SHELL_MIN_CHILDREN:
                return True

    return False


def is_nested_fragment(item, items, image_area, image_width, image_height):
    area, polygon = item
    bbox = polygon_bbox(polygon)
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    if not (
        image_width * NESTED_FRAGMENT_MIN_CENTER_X_RATIO < center[0] < image_width * NESTED_FRAGMENT_MAX_CENTER_X_RATIO
        and image_height * NESTED_FRAGMENT_MIN_CENTER_Y_RATIO < center[1] < image_height * NESTED_FRAGMENT_MAX_CENTER_Y_RATIO
    ):
        return False

    for parent_area, parent_polygon in items:
        if parent_area < image_area * NESTED_FRAGMENT_PARENT_AREA_RATIO:
            continue
        if area >= parent_area * NESTED_FRAGMENT_MAX_AREA_RATIO:
            continue

        parent_contour = np.array(parent_polygon, dtype=np.int32)
        if cv2.pointPolygonTest(parent_contour, center, False) >= 0:
            return True

    return False


def filter_polygon_items(polygon_items, image_width, image_height):
    image_area = image_width * image_height
    filtered = [
        item
        for item in polygon_items
        if not is_outer_shell(item, polygon_items, image_area)
        and not is_nested_fragment(item, polygon_items, image_area, image_width, image_height)
    ]

    selected = []
    for item in sorted(filtered, key=lambda current: current[0], reverse=True):
        area, polygon = item
        bbox = polygon_bbox(polygon)
        duplicate_index = None

        for index, (selected_area, selected_polygon) in enumerate(selected):
            selected_bbox = polygon_bbox(selected_polygon)
            area_ratio = min(area, selected_area) / max(area, selected_area)
            if (
                bbox_iou(bbox, selected_bbox) > DUPLICATE_BBOX_IOU
                and area_ratio > DUPLICATE_AREA_RATIO
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            selected.append(item)
            continue

        selected_area, selected_polygon = selected[duplicate_index]
        if len(polygon) < len(selected_polygon):
            selected[duplicate_index] = item

    return selected


def snap_shared_vertices(polygons, tolerance=SNAP_TOLERANCE):
    vertices = []
    for polygon_index, polygon in enumerate(polygons):
        source_points = polygon[:-1] if polygon and polygon[0] == polygon[-1] else polygon
        for vertex_index, point in enumerate(source_points):
            vertices.append((polygon_index, vertex_index, point[0], point[1]))

    parents = list(range(len(vertices)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(vertices):
        for second_index in range(first_index + 1, len(vertices)):
            second = vertices[second_index]
            if first[0] == second[0]:
                continue
            if math.hypot(first[2] - second[2], first[3] - second[3]) <= tolerance:
                union(first_index, second_index)

    clusters = {}
    for index, vertex in enumerate(vertices):
        root = find(index)
        clusters.setdefault(root, []).append(vertex)

    snapped_points = {}
    for cluster in clusters.values():
        x = int(round(sum(vertex[2] for vertex in cluster) / len(cluster)))
        y = int(round(sum(vertex[3] for vertex in cluster) / len(cluster)))
        for polygon_index, vertex_index, _, _ in cluster:
            snapped_points[(polygon_index, vertex_index)] = [x, y]

    snapped_polygons = []
    for polygon_index, polygon in enumerate(polygons):
        source_points = polygon[:-1] if polygon and polygon[0] == polygon[-1] else polygon
        snapped_polygon = [
            snapped_points.get((polygon_index, vertex_index), point)
            for vertex_index, point in enumerate(source_points)
        ]
        deduplicated = []
        for point in snapped_polygon:
            if not deduplicated or point != deduplicated[-1]:
                deduplicated.append(point)

        if len(deduplicated) < 3:
            snapped_polygons.append(polygon)
            continue

        if deduplicated[0] != deduplicated[-1]:
            deduplicated.append(deduplicated[0])
        snapped_polygons.append(deduplicated)

    return snapped_polygons


def point_to_segment_projection(point, segment_start, segment_end):
    px, py = point
    ax, ay = segment_start
    bx, by = segment_end
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return None

    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    if t <= 0 or t >= 1:
        return None

    projected_x = ax + t * dx
    projected_y = ay + t * dy
    distance = math.hypot(px - projected_x, py - projected_y)
    return t, projected_x, projected_y, distance


def insert_vertices_on_neighbor_edges(polygons, tolerance=EDGE_SNAP_TOLERANCE):
    open_polygons = [
        polygon[:-1] if polygon and polygon[0] == polygon[-1] else polygon
        for polygon in polygons
    ]
    insertions = {}

    for source_index, source_polygon in enumerate(open_polygons):
        for source_point in source_polygon:
            best_match = None
            for target_index, target_polygon in enumerate(open_polygons):
                if source_index == target_index or len(target_polygon) < 2:
                    continue

                for segment_index, segment_start in enumerate(target_polygon):
                    segment_end = target_polygon[
                        (segment_index + 1) % len(target_polygon)
                    ]
                    projection = point_to_segment_projection(
                        source_point,
                        segment_start,
                        segment_end,
                    )
                    if projection is None:
                        continue

                    t, projected_x, projected_y, distance = projection
                    projected_point = [int(round(projected_x)), int(round(projected_y))]
                    if distance > tolerance:
                        continue
                    distance_to_start = math.hypot(
                        projected_point[0] - segment_start[0],
                        projected_point[1] - segment_start[1],
                    )
                    distance_to_end = math.hypot(
                        projected_point[0] - segment_end[0],
                        projected_point[1] - segment_end[1],
                    )
                    if (
                        distance_to_start <= tolerance
                        or distance_to_end <= tolerance
                    ):
                        continue
                    if best_match is None or distance < best_match[0]:
                        best_match = (
                            distance,
                            target_index,
                            segment_index,
                            t,
                            projected_point,
                        )

            if best_match is None:
                continue

            _, target_index, segment_index, t, projected_point = best_match
            insertions.setdefault(target_index, {}).setdefault(segment_index, []).append(
                (t, projected_point)
            )

    if not insertions:
        return polygons

    result = []
    inserted_count = 0
    for polygon_index, polygon in enumerate(open_polygons):
        polygon_insertions = insertions.get(polygon_index, {})
        new_polygon = []
        for vertex_index, point in enumerate(polygon):
            if not new_polygon or point != new_polygon[-1]:
                new_polygon.append(point)

            segment_insertions = polygon_insertions.get(vertex_index, [])
            seen_points = set()
            for _, inserted_point in sorted(segment_insertions, key=lambda item: item[0]):
                key = (inserted_point[0], inserted_point[1])
                if key in seen_points:
                    continue
                seen_points.add(key)
                if inserted_point != new_polygon[-1]:
                    new_polygon.append(inserted_point)
                    inserted_count += 1

        deduplicated = []
        for point in new_polygon:
            if not deduplicated or point != deduplicated[-1]:
                deduplicated.append(point)

        if len(deduplicated) < 3:
            result.append(polygons[polygon_index])
            continue

        if deduplicated[0] != deduplicated[-1]:
            deduplicated.append(deduplicated[0])
        result.append(deduplicated)

    print(f"Edge snap inserted vertices: {inserted_count}")
    return snap_shared_vertices(result, tolerance)


def make_stage1_polygon_json(polygons, image_width, image_height):
    center_x = image_width / 2
    center_y = image_height / 2
    result_data = []

    for i, polygon in enumerate(polygons):
        shifted = [(x - center_x, y - center_y) for x, y in polygon]
        if APPLY_ROTATE_AND_MIRROR:
            transformed = [[round(y, 6), round(x, 6)] for x, y in shifted]
        else:
            transformed = [[round(x, 6), round(y, 6)] for x, y in shifted]

        number = str(i + 1)
        idtur = number.zfill(5)
        result_data.append({
            "adres": "",
            "id": i + 1,
            "idtur": idtur,
            "kadastr": "",
            "kadastrurl": "",
            "names": idtur,
            "number": number,
            "price": "",
            "size": "",
            "status": "sale",
            "coordinates": [transformed],
        })

    return {"inc": len(result_data), "data": result_data}


def save_svg_grid(polygons, image_width, image_height, filename):
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{image_width}" height="{image_height}" '
            f'viewBox="0 0 {image_width} {image_height}">'
        ),
        (
            f'  <g fill="none" stroke="{SVG_STROKE_COLOR}" '
            f'stroke-width="{SVG_LINE_WIDTH}" stroke-linejoin="round">'
        ),
    ]

    for i, polygon in enumerate(polygons, start=1):
        points = " ".join(f"{x},{y}" for x, y in polygon)
        lines.append(f'    <polygon id="poly_{i:05d}" points="{points}" />')

    lines.extend(["  </g>", "</svg>"])

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except PermissionError:
        print(f"Could not save SVG: {filename} is locked or not writable.")
        return False

    return True

def main():
    if not os.path.exists(INPUT_IMAGE):
        print(f"File not found: {INPUT_IMAGE}")
        return

    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"Could not read image: {INPUT_IMAGE}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_height, image_width = gray.shape[:2]
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    thresh = remove_hatching(thresh, gray)
    thresh = remove_gray_fill(thresh, img, gray)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Р Р°СЃС‡С‘С‚ РїР»РѕС‰Р°РґРµР№
    areas = [cv2.contourArea(c) for c in contours]
    if not areas:
        print("No contours found.")
        return

    polygon_items = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if not (
            is_candidate_contour(cnt, area, image_width, image_height)
            or is_recoverable_lower_gray_massif(cnt, area, image_width, image_height)
        ):
            continue

        source_contour = contour_for_polygon(cnt, area, image_width, image_height)
        epsilon = APPROX_EPSILON_COEFF * cv2.arcLength(source_contour, True)
        approx = cv2.approxPolyDP(source_contour, epsilon, True)
        points = approx.reshape(-1, 2)
        polygon = [[int(x), int(y)] for x, y in points]
        if len(polygon) < 3:
            continue

        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])

        polygon_items.append((area, polygon))

    polygon_items = filter_polygon_items(polygon_items, image_width, image_height)
    polygons = [polygon for _, polygon in sorted(polygon_items, reverse=True)]
    polygons = snap_shared_vertices(polygons)
    polygons = insert_vertices_on_neighbor_edges(polygons)

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab20")

    for id_counter, polygon in enumerate(polygons):
        xs, ys = zip(*polygon)
        ax.fill(
            xs,
            ys,
            facecolor=cmap(id_counter % 20),
            edgecolor=(0, 0, 0, 0.55),
            linewidth=POLYGON_LINE_WIDTH,
            alpha=POLYGON_ALPHA,
        )

    if not polygons:
        print("No polygons passed the filter.")
        return

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    try:
        plt.savefig(OUTPUT_IMAGE, bbox_inches="tight", dpi=PREVIEW_DPI)
        print(f"Preview saved: {OUTPUT_IMAGE}")
    except PermissionError:
        print(f"Could not save preview: {OUTPUT_IMAGE} is locked or not writable.")
    finally:
        plt.close()

    if save_svg_grid(polygons, image_width, image_height, OUTPUT_SVG):
        print(f"SVG grid saved: {OUTPUT_SVG}")

    try:
        with open(OUTPUT_RAW_JSON, "w", encoding="utf-8") as f:
            json.dump(polygons, f, indent=2)
    except PermissionError:
        print(f"Could not save raw JSON: {OUTPUT_RAW_JSON} is locked or not writable.")

    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(
                make_stage1_polygon_json(polygons, image_width, image_height),
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Polygon JSON saved: {OUTPUT_JSON}")
    except PermissionError:
        print(f"Could not save polygon JSON: {OUTPUT_JSON} is locked or not writable.")

if __name__ == "__main__":
    main()
