import sys

sys.dont_write_bytecode = True

import cv2
import json
import math
import os
import matplotlib.pyplot as plt
import numpy as np

import detect_polygons_from_image as base


INPUT_IMAGE = base.INPUT_IMAGE
OUTPUT_MASK = "skeleton_line_mask.png"
OUTPUT_SKELETON = "skeleton_centerline.png"
OUTPUT_CENTERLINE_SVG = "skeleton_centerline_grid.svg"
OUTPUT_PREVIEW = "skeleton_output_detected.png"
OUTPUT_JSON = "skeleton_polygon.json"
OUTPUT_RAW_JSON = "skeleton_detected_polygons.json"
OUTPUT_SVG = "skeleton_polygon_grid.svg"

SKELETON_WALL_DILATE = 3
SKELETON_CLOSE_GAP = 3
SKELETON_MIN_AREA_RATIO = 0.0007
SKELETON_MAX_AREA_RATIO = 0.05
SKELETON_APPROX_EPSILON_COEFF = 0.0015
SKELETON_PAGE_MARGIN_RATIO = 0.003
GRAPH_SIMPLIFY_EPSILON = 2.0
GRAPH_MIN_EDGE_LENGTH = 10
GRAPH_MIN_FACE_AREA_RATIO = 0.0007
GRAPH_MAX_FACE_AREA_RATIO = 0.05
PROJECT_TO_SKELETON_RADIUS = 8
PROJECT_TO_GRAPH_NODE_RADIUS = 12

NEIGHBOR_OFFSETS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
)


def save_image(filename, image):
    try:
        cv2.imwrite(filename, image)
        print(f"Debug image saved: {filename}")
    except PermissionError:
        print(f"Could not save debug image: {filename} is locked or not writable.")


def build_line_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask = base.remove_hatching(mask, gray)
    mask = base.remove_gray_fill(mask, img, gray)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = (
        ((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170))
        & (hsv[:, :, 1] > 80)
        & (hsv[:, :, 2] > 80)
    ).astype(np.uint8) * 255
    mask = cv2.subtract(mask, red_mask)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    image_area = mask.shape[0] * mask.shape[1]
    min_component_area = max(8, int(image_area * 0.0000008))

    for label in range(1, component_count):
        x, y, width, height, area = stats[label]
        if area < min_component_area and width < 60 and height < 60:
            continue
        cleaned[labels == label] = 255

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (SKELETON_CLOSE_GAP, SKELETON_CLOSE_GAP),
    )
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)


def morphological_skeleton(mask):
    source = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(source)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(source):
        eroded = cv2.erode(source, element)
        opened = cv2.dilate(eroded, element)
        residue = cv2.subtract(source, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        source = eroded

    return skeleton


def contour_to_polygon(contour):
    epsilon = SKELETON_APPROX_EPSILON_COEFF * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    polygon = [[int(x), int(y)] for x, y in approx.reshape(-1, 2)]
    if len(polygon) < 3:
        return None
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    return polygon


def project_point_to_skeleton(point, skeleton, radius=PROJECT_TO_SKELETON_RADIUS):
    x, y = point
    height, width = skeleton.shape[:2]
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(width, x + radius + 1)
    y2 = min(height, y + radius + 1)
    window = skeleton[y1:y2, x1:x2]
    candidates = np.argwhere(window > 0)
    if len(candidates) == 0:
        return [int(x), int(y)]

    best_point = None
    best_distance = None
    for cy, cx in candidates:
        candidate_x = int(x1 + cx)
        candidate_y = int(y1 + cy)
        distance = (candidate_x - x) ** 2 + (candidate_y - y) ** 2
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_point = [candidate_x, candidate_y]

    return best_point


def graph_nodes_from_paths(paths):
    nodes = set()
    for path in paths:
        if len(path) < 2:
            continue
        nodes.add(tuple(path[0]))
        nodes.add(tuple(path[-1]))
    return nodes


def junction_nodes_from_skeleton(skeleton, min_degree=3):
    point_set = skeleton_point_set(skeleton)
    nodes = set()
    for point in point_set:
        if len(point_neighbors(point, point_set)) >= min_degree:
            nodes.add(point)
    return nodes


def build_graph_node_index(graph_nodes, cell_size=PROJECT_TO_GRAPH_NODE_RADIUS):
    index = {}
    for node_x, node_y in graph_nodes:
        key = (node_x // cell_size, node_y // cell_size)
        index.setdefault(key, []).append((node_x, node_y))
    return index


def project_point_to_graph_node(
    point,
    graph_node_index,
    radius=PROJECT_TO_GRAPH_NODE_RADIUS,
):
    if not graph_node_index:
        return point

    best_node = None
    best_distance = None
    radius_sq = radius * radius
    x, y = point
    cell_x = x // radius
    cell_y = y // radius

    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            candidates = graph_node_index.get(
                (cell_x + offset_x, cell_y + offset_y),
                [],
            )
            for node_x, node_y in candidates:
                distance = (node_x - x) ** 2 + (node_y - y) ** 2
                if distance > radius_sq:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_node = [node_x, node_y]

    return best_node if best_node is not None else point


def remove_short_corner_segments(polygon, min_length=PROJECT_TO_GRAPH_NODE_RADIUS):
    if len(polygon) < 4:
        return polygon

    source = polygon[:-1] if polygon[0] == polygon[-1] else polygon
    changed = True
    points = source

    while changed and len(points) >= 3:
        changed = False
        result = []
        count = len(points)

        for index, point in enumerate(points):
            previous_point = points[(index - 1) % count]
            next_point = points[(index + 1) % count]
            if (
                point_distance(point, previous_point) < min_length
                and point_distance(point, next_point) < min_length
            ):
                changed = True
                continue
            result.append(point)

        if len(result) < 3:
            break
        points = result

    if points[0] != points[-1]:
        points = points + [points[0]]
    return points


def contour_to_centerline_polygon(contour, skeleton, graph_node_index=None):
    points = contour.reshape(-1, 2)
    projected = [
        project_point_to_skeleton([int(x), int(y)], skeleton)
        for x, y in points
    ]
    projected = [
        project_point_to_graph_node(point, graph_node_index or {})
        for point in projected
    ]

    deduplicated = []
    for point in projected:
        if not deduplicated or deduplicated[-1] != point:
            deduplicated.append(point)

    if len(deduplicated) > 1 and deduplicated[0] == deduplicated[-1]:
        deduplicated.pop()

    if len(deduplicated) < 3:
        return None

    simplified = simplify_path(deduplicated, SKELETON_APPROX_EPSILON_COEFF * 1200)
    if len(simplified) < 3:
        return None
    simplified = remove_short_corner_segments(simplified)
    if len(simplified) < 3:
        return None
    if simplified[0] != simplified[-1]:
        simplified.append(simplified[0])
    return simplified


def point_distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def simplify_path(path, epsilon=GRAPH_SIMPLIFY_EPSILON):
    if len(path) <= 2:
        return path

    contour = np.array(path, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(contour, epsilon, False)
    simplified = [[int(round(x)), int(round(y))] for x, y in approx.reshape(-1, 2)]
    if simplified[0] != path[0]:
        simplified.insert(0, path[0])
    if simplified[-1] != path[-1]:
        simplified.append(path[-1])
    return simplified


def skeleton_point_set(skeleton):
    return {(int(x), int(y)) for y, x in np.argwhere(skeleton > 0)}


def point_neighbors(point, point_set):
    x, y = point
    return [
        (x + dx, y + dy)
        for dx, dy in NEIGHBOR_OFFSETS
        if (x + dx, y + dy) in point_set
    ]


def edge_key(first, second):
    return tuple(sorted((first, second)))


def vectorize_skeleton(skeleton):
    point_set = skeleton_point_set(skeleton)
    neighbor_map = {point: point_neighbors(point, point_set) for point in point_set}
    graph_nodes = {
        point
        for point, neighbors in neighbor_map.items()
        if len(neighbors) != 2
    }
    visited_edges = set()
    paths = []

    for start in graph_nodes:
        for neighbor in neighbor_map[start]:
            first_edge = edge_key(start, neighbor)
            if first_edge in visited_edges:
                continue

            path = [start, neighbor]
            visited_edges.add(first_edge)
            previous = start
            current = neighbor

            while current not in graph_nodes:
                next_points = [
                    point for point in neighbor_map[current] if point != previous
                ]
                if not next_points:
                    break

                next_point = next_points[0]
                current_edge = edge_key(current, next_point)
                if current_edge in visited_edges:
                    break

                path.append(next_point)
                visited_edges.add(current_edge)
                previous, current = current, next_point

            if len(path) >= 2 and point_distance(path[0], path[-1]) >= GRAPH_MIN_EDGE_LENGTH:
                paths.append(simplify_path(path))

    print(f"Skeleton graph paths: {len(paths)}")
    return paths


def save_graph_svg(paths, image_width, image_height, filename):
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{image_width}" height="{image_height}" '
            f'viewBox="0 0 {image_width} {image_height}">'
        ),
        (
            f'  <g fill="none" stroke="{base.SVG_STROKE_COLOR}" '
            f'stroke-width="{base.SVG_LINE_WIDTH}" stroke-linejoin="round" '
            f'stroke-linecap="round">'
        ),
    ]

    for path in paths:
        commands = [f"M {path[0][0]},{path[0][1]}"]
        commands.extend(f"L {x},{y}" for x, y in path[1:])
        lines.append(f'    <path d="{" ".join(commands)}" />')

    lines.extend(["  </g>", "</svg>"])

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Centerline SVG saved: {filename} ({len(paths)} paths)")
    except PermissionError:
        print(f"Could not save centerline SVG: {filename} is locked or not writable.")


def signed_area(polygon):
    area = 0
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area / 2


def path_angle_from_node(path, node):
    if tuple(path[0]) == node:
        first = path[0]
        second = path[1]
    else:
        first = path[-1]
        second = path[-2]
    return math.atan2(second[1] - first[1], second[0] - first[0])


def build_vector_graph(paths):
    adjacency = {}
    edge_geometries = {}

    for path in paths:
        start = tuple(path[0])
        end = tuple(path[-1])
        if start == end:
            continue

        adjacency.setdefault(start, set()).add(end)
        adjacency.setdefault(end, set()).add(start)
        edge_geometries[(start, end)] = path
        edge_geometries[(end, start)] = list(reversed(path))

    sorted_adjacency = {}
    for node, neighbors in adjacency.items():
        sorted_adjacency[node] = sorted(
            neighbors,
            key=lambda neighbor: path_angle_from_node(
                edge_geometries[(node, neighbor)],
                node,
            ),
        )

    return sorted_adjacency, edge_geometries


def next_face_edge(current_edge, sorted_adjacency, turn_direction):
    previous, current = current_edge
    neighbors = sorted_adjacency.get(current, [])
    if previous not in neighbors:
        return None

    previous_index = neighbors.index(previous)
    next_neighbor = neighbors[
        (previous_index + turn_direction) % len(neighbors)
    ]
    return current, next_neighbor


def normalize_polygon(points):
    normalized = []
    for point in points:
        point = [int(round(point[0])), int(round(point[1]))]
        if not normalized or normalized[-1] != point:
            normalized.append(point)

    if len(normalized) > 1 and normalized[0] == normalized[-1]:
        normalized.pop()

    if len(normalized) < 3:
        return None

    if normalized[0] != normalized[-1]:
        normalized.append(normalized[0])
    return normalized


def graph_faces_from_paths(paths, image_width, image_height):
    sorted_adjacency, edge_geometries = build_vector_graph(paths)
    directed_edges = list(edge_geometries.keys())
    visited = set()
    image_area = image_width * image_height
    min_area = image_area * GRAPH_MIN_FACE_AREA_RATIO
    max_area = image_area * GRAPH_MAX_FACE_AREA_RATIO
    margin = max(20, int(min(image_width, image_height) * SKELETON_PAGE_MARGIN_RATIO))
    face_items = []
    closed_walk_count = 0

    for turn_direction in (-1, 1):
        visited = set()
        for start_edge in directed_edges:
            if start_edge in visited:
                continue

            current_edge = start_edge
            face_edges = []
            face_points = []

            for _ in range(len(directed_edges) + 1):
                if current_edge in face_edges:
                    break
                if current_edge in visited and current_edge != start_edge:
                    break

                face_edges.append(current_edge)
                path = edge_geometries[current_edge]
                if not face_points:
                    face_points.extend(path)
                else:
                    face_points.extend(path[1:])

                next_edge = next_face_edge(
                    current_edge,
                    sorted_adjacency,
                    turn_direction,
                )
                if next_edge is None:
                    break
                current_edge = next_edge
                if current_edge == start_edge:
                    break

            if current_edge != start_edge:
                continue

            closed_walk_count += 1
            for edge in face_edges:
                visited.add(edge)

            polygon = normalize_polygon(face_points)
            if polygon is None:
                continue

            area = abs(signed_area(polygon))
            if area < min_area or area > max_area:
                continue

            bbox = base.polygon_bbox(polygon)
            if bbox[0] <= margin or bbox[1] <= margin:
                continue
            if bbox[2] >= image_width - margin or bbox[3] >= image_height - margin:
                continue

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            if center_y < image_height * base.TOP_DECORATION_Y_RATIO:
                continue
            if (
                center_x > image_width * base.LEGEND_X_RATIO
                and center_y < image_height * base.LEGEND_Y_RATIO
            ):
                continue

            face_items.append((area, polygon))

    print(f"Graph closed walks: {closed_walk_count}")
    print(f"Graph faces: {len(face_items)}")
    return [polygon for _, polygon in sorted(face_items, reverse=True)]


def polygons_from_skeleton(skeleton, image_width, image_height, graph_node_index=None):
    wall_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (SKELETON_WALL_DILATE, SKELETON_WALL_DILATE),
    )
    walls = cv2.dilate(skeleton, wall_kernel)
    free_space = cv2.bitwise_not(walls)

    image_area = image_width * image_height
    min_area = image_area * SKELETON_MIN_AREA_RATIO
    max_area = image_area * SKELETON_MAX_AREA_RATIO
    margin = max(20, int(min(image_width, image_height) * SKELETON_PAGE_MARGIN_RATIO))

    polygon_items = []
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(free_space, 4)
    print(f"Skeleton free-space components: {component_count - 1}")

    for label in range(1, component_count):
        x, y, width, height, area = stats[label]
        center_x = x + width / 2
        center_y = y + height / 2
        if x <= margin or y <= margin:
            continue
        if x + width >= image_width - margin or y + height >= image_height - margin:
            continue
        if width < 40 or height < 40:
            continue
        if center_y < image_height * base.TOP_DECORATION_Y_RATIO:
            continue
        if (
            center_x > image_width * base.LEGEND_X_RATIO
            and center_y < image_height * base.LEGEND_Y_RATIO
        ):
            continue
        if area < min_area or area > max_area:
            continue

        component_mask = np.zeros_like(free_space)
        component_mask[labels == label] = 255
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        if contour_area < min_area or contour_area > max_area:
            continue

        polygon = contour_to_centerline_polygon(contour, skeleton, graph_node_index)
        if polygon is None:
            continue

        centerline_area = abs(signed_area(polygon))
        if centerline_area < min_area or centerline_area > max_area:
            continue

        polygon_items.append((centerline_area, polygon))

    return [polygon for _, polygon in sorted(polygon_items, reverse=True)]


def save_preview(polygons):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab20")

    for index, polygon in enumerate(polygons):
        xs, ys = zip(*polygon)
        ax.fill(
            xs,
            ys,
            facecolor=cmap(index % 20),
            edgecolor=(0, 0, 0, 0.55),
            linewidth=base.POLYGON_LINE_WIDTH,
            alpha=base.POLYGON_ALPHA,
        )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    try:
        plt.savefig(OUTPUT_PREVIEW, bbox_inches="tight", dpi=base.PREVIEW_DPI)
        print(f"Preview saved: {OUTPUT_PREVIEW}")
    except PermissionError:
        print(f"Could not save preview: {OUTPUT_PREVIEW} is locked or not writable.")
    finally:
        plt.close()


def save_centerline_svg(skeleton, image_width, image_height, filename):
    paths = vectorize_skeleton(skeleton)
    save_graph_svg(paths, image_width, image_height, filename)
    return paths


def main():
    if not os.path.exists(INPUT_IMAGE):
        print(f"File not found: {INPUT_IMAGE}")
        return

    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"Could not read image: {INPUT_IMAGE}")
        return

    image_height, image_width = img.shape[:2]
    line_mask = build_line_mask(img)
    skeleton = morphological_skeleton(line_mask)

    save_image(OUTPUT_MASK, line_mask)
    save_image(OUTPUT_SKELETON, skeleton)
    graph_paths = save_centerline_svg(
        skeleton,
        image_width,
        image_height,
        OUTPUT_CENTERLINE_SVG,
    )

    polygons = graph_faces_from_paths(graph_paths, image_width, image_height)
    if not polygons:
        print("Graph faces are empty; using projected component polygons.")
        polygons = polygons_from_skeleton(
            skeleton,
            image_width,
            image_height,
            build_graph_node_index(junction_nodes_from_skeleton(skeleton)),
        )
    polygons = base.snap_shared_vertices(
        polygons,
        tolerance=base.SNAP_TOLERANCE,
        line_mask=skeleton,
    )
    print(f"Skeleton polygons: {len(polygons)}")

    if not polygons:
        print("No polygons detected from skeleton.")
        return

    save_preview(polygons)

    if base.save_svg_grid(polygons, image_width, image_height, OUTPUT_SVG):
        print(f"SVG grid saved: {OUTPUT_SVG}")

    try:
        with open(OUTPUT_RAW_JSON, "w", encoding="utf-8") as f:
            json.dump(polygons, f, indent=2)
    except PermissionError:
        print(f"Could not save raw JSON: {OUTPUT_RAW_JSON} is locked or not writable.")

    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(
                base.make_stage1_polygon_json(polygons, image_width, image_height),
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Polygon JSON saved: {OUTPUT_JSON}")
    except PermissionError:
        print(f"Could not save polygon JSON: {OUTPUT_JSON} is locked or not writable.")


if __name__ == "__main__":
    main()
