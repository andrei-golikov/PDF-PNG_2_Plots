import json
import math
import sys

import cv2
import numpy as np

sys.dont_write_bytecode = True

INPUT_JSON = "pdf_vector_boundary_candidate.json"
OUTPUT_JSON = "pdf_vector_graph_clean.json"
OUTPUT_SVG = "pdf_vector_graph_clean.svg"
OUTPUT_PNG = "pdf_vector_graph_clean.png"
OUTPUT_NODES_JSON = "pdf_vector_graph_nodes.json"
OUTPUT_FACES_JSON = "pdf_vector_faces.json"
OUTPUT_FACES_SVG = "pdf_vector_faces.svg"
OUTPUT_FACES_PNG = "pdf_vector_faces.png"

SVG_STROKE_COLOR = "#ff0000"
SVG_LINE_WIDTH = 1
PREVIEW_HEIGHT = 1600

# PDF coordinate units. This is deliberately smaller than road width, because
# roads are often two close parallel boundaries that must not be collapsed.
ENDPOINT_CLUSTER_RADIUS = 15.0
COLLINEAR_MAX_OFFSET = 2.5
JUNCTION_MAX_INTERSECTION_SHIFT = 18.0
MIN_OUTPUT_SEGMENT_LENGTH = 5.0
EDGE_JOIN_DISTANCE = 7.0
EDGE_JOIN_MIN_ANGLE_DEGREES = 18.0
INTERSECTION_EPSILON = 1e-6
NODE_ROUND_DIGITS = 3
MIN_FACE_AREA = 1000.0
MAX_FACE_AREA = 120000.0
MAX_FACE_THINNESS = 40.0


def load_segments(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, data["segments"]


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def segment_length(segment):
    return distance(segment["a"], segment["b"])


def endpoint_direction(segment, endpoint_name):
    ax, ay = segment["a"]
    bx, by = segment["b"]
    if endpoint_name == "a":
        dx = bx - ax
        dy = by - ay
    else:
        dx = ax - bx
        dy = ay - by
    length = math.hypot(dx, dy)
    if length == 0:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def angle_between_dirs(a, b):
    dot = abs(a[0] * b[0] + a[1] * b[1])
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def line_intersection(p1, d1, p2, d2):
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-6:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    t = (dx * d2[1] - dy * d2[0]) / cross
    return (p1[0] + d1[0] * t, p1[1] + d1[1] * t)


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def collect_endpoints(segments):
    endpoints = []
    for segment_index, segment in enumerate(segments):
        endpoints.append(
            {
                "segment_index": segment_index,
                "endpoint": "a",
                "point": tuple(segment["a"]),
                "direction": endpoint_direction(segment, "a"),
            }
        )
        endpoints.append(
            {
                "segment_index": segment_index,
                "endpoint": "b",
                "point": tuple(segment["b"]),
                "direction": endpoint_direction(segment, "b"),
            }
        )
    return endpoints


def cluster_endpoints(endpoints):
    uf = UnionFind(len(endpoints))
    cell_size = ENDPOINT_CLUSTER_RADIUS
    grid = {}

    for index, endpoint in enumerate(endpoints):
        x, y = endpoint["point"]
        key = (int(math.floor(x / cell_size)), int(math.floor(y / cell_size)))
        for gx in range(key[0] - 1, key[0] + 2):
            for gy in range(key[1] - 1, key[1] + 2):
                for other_index in grid.get((gx, gy), []):
                    other = endpoints[other_index]
                    if distance(endpoint["point"], other["point"]) <= ENDPOINT_CLUSTER_RADIUS:
                        uf.union(index, other_index)
        grid.setdefault(key, []).append(index)

    groups = {}
    for index in range(len(endpoints)):
        groups.setdefault(uf.find(index), []).append(index)
    return list(groups.values())


def dominant_direction_groups(cluster, endpoints):
    groups = []
    for endpoint_index in cluster:
        direction = endpoints[endpoint_index]["direction"]
        point = endpoints[endpoint_index]["point"]
        for group in groups:
            if angle_between_dirs(direction, group["direction"]) <= 12.0:
                group["items"].append(endpoint_index)
                group["points"].append(point)
                # Weighted enough for diagnostics; no need for perfect circular mean.
                gx, gy = group["direction"]
                nx = gx + direction[0]
                ny = gy + direction[1]
                length = math.hypot(nx, ny)
                if length > 0:
                    group["direction"] = (nx / length, ny / length)
                break
        else:
            groups.append({"direction": direction, "items": [endpoint_index], "points": [point]})
    return sorted(groups, key=lambda group: len(group["items"]), reverse=True)


def average_point(points):
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def cluster_target(cluster, endpoints):
    points = [endpoints[index]["point"] for index in cluster]
    if len(points) == 1:
        return points[0], "single"

    groups = dominant_direction_groups(cluster, endpoints)
    centroid = average_point(points)

    if len(groups) == 1:
        direction = groups[0]["direction"]
        normal = (-direction[1], direction[0])
        projections = [point[0] * normal[0] + point[1] * normal[1] for point in points]
        if max(projections) - min(projections) > COLLINEAR_MAX_OFFSET:
            return None, "parallel-offset"
        return centroid, "collinear-gap"

    first = groups[0]
    second = groups[1]
    if angle_between_dirs(first["direction"], second["direction"]) <= 12.0:
        direction = first["direction"]
        normal = (-direction[1], direction[0])
        projections = [point[0] * normal[0] + point[1] * normal[1] for point in points]
        if max(projections) - min(projections) > COLLINEAR_MAX_OFFSET:
            return None, "parallel-offset"
        return centroid, "near-parallel"

    p1 = average_point(first["points"])
    p2 = average_point(second["points"])
    intersection = line_intersection(p1, first["direction"], p2, second["direction"])
    if intersection and distance(intersection, centroid) <= JUNCTION_MAX_INTERSECTION_SHIFT:
        return intersection, "intersection"
    return centroid, "junction-centroid"


def clean_segments(segments):
    endpoints = collect_endpoints(segments)
    clusters = cluster_endpoints(endpoints)
    replacements = {}
    nodes = []

    for cluster in clusters:
        target, reason = cluster_target(cluster, endpoints)
        if target is None:
            continue
        if len(cluster) > 1:
            nodes.append(
                {
                    "point": [round(target[0], 3), round(target[1], 3)],
                    "reason": reason,
                    "endpoints": len(cluster),
                }
            )
        for endpoint_index in cluster:
            replacements[endpoint_index] = target

    cleaned = []
    for segment_index, segment in enumerate(segments):
        a_index = segment_index * 2
        b_index = a_index + 1
        a = replacements.get(a_index, tuple(segment["a"]))
        b = replacements.get(b_index, tuple(segment["b"]))
        if distance(a, b) < MIN_OUTPUT_SEGMENT_LENGTH:
            continue
        item = dict(segment)
        item["a"] = [round(a[0], 3), round(a[1], 3)]
        item["b"] = [round(b[0], 3), round(b[1], 3)]
        cleaned.append(item)

    stats = {
        "input_segments": len(segments),
        "output_segments": len(cleaned),
        "endpoint_clusters": len(clusters),
        "junction_nodes": len(nodes),
    }
    return cleaned, nodes, stats


def project_point_to_segment(point, segment):
    ax, ay = segment["a"]
    bx, by = segment["b"]
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return None
    t = ((point[0] - ax) * dx + (point[1] - ay) * dy) / length_sq
    if t < 0.0 or t > 1.0:
        return None
    projected = (ax + dx * t, ay + dy * t)
    return t, projected, distance(point, projected)


def segment_unit(segment):
    ax, ay = segment["a"]
    bx, by = segment["b"]
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    if length == 0:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def join_dangling_endpoints_to_edges(segments):
    endpoint_counts = {}
    for segment in segments:
        endpoint_counts[node_key(segment["a"])] = endpoint_counts.get(node_key(segment["a"]), 0) + 1
        endpoint_counts[node_key(segment["b"])] = endpoint_counts.get(node_key(segment["b"]), 0) + 1

    replacements = {}
    joins = []
    for segment_index, segment in enumerate(segments):
        for endpoint_name in ("a", "b"):
            point = tuple(segment[endpoint_name])
            if endpoint_counts.get(node_key(point), 0) != 1:
                continue
            source_direction = endpoint_direction(segment, endpoint_name)
            best = None
            for target_index, target in enumerate(segments):
                if target_index == segment_index:
                    continue
                target_direction = segment_unit(target)
                if angle_between_dirs(source_direction, target_direction) < EDGE_JOIN_MIN_ANGLE_DEGREES:
                    continue
                projected = project_point_to_segment(point, target)
                if not projected:
                    continue
                t, projected_point, dist = projected
                if dist > EDGE_JOIN_DISTANCE:
                    continue
                if t < 0.02 or t > 0.98:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, projected_point, target_index)
            if best:
                replacements[(segment_index, endpoint_name)] = best[1]
                joins.append(
                    {
                        "segment_index": segment_index,
                        "endpoint": endpoint_name,
                        "target_segment_index": best[2],
                        "distance": round(best[0], 3),
                        "point": [round(best[1][0], 3), round(best[1][1], 3)],
                    }
                )

    result = []
    for segment_index, segment in enumerate(segments):
        item = dict(segment)
        for endpoint_name in ("a", "b"):
            replacement = replacements.get((segment_index, endpoint_name))
            if replacement:
                item[endpoint_name] = [round(replacement[0], 3), round(replacement[1], 3)]
        if segment_length(item) >= MIN_OUTPUT_SEGMENT_LENGTH:
            result.append(item)
    return result, joins


def segments_bbox(segments, fallback_box):
    if not segments:
        return fallback_box
    xs = []
    ys = []
    for segment in segments:
        xs.extend([segment["a"][0], segment["b"][0]])
        ys.extend([segment["a"][1], segment["b"][1]])
    return [min(xs), min(ys), max(xs), max(ys)]


def write_svg(path, bbox, segments):
    min_x, min_y, max_x, max_y = bbox
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


def write_png(path, bbox, segments):
    min_x, min_y, max_x, max_y = bbox
    height = PREVIEW_HEIGHT
    width = max(1, int(round(height * (max_x - min_x) / (max_y - min_y))))
    image = np.full((height, width, 3), 255, np.uint8)
    for segment in segments:
        x1, y1 = segment["a"]
        x2, y2 = segment["b"]
        p1 = (
            int(round((x1 - min_x) / (max_x - min_x) * (width - 1))),
            int(round((max_y - y1) / (max_y - min_y) * (height - 1))),
        )
        p2 = (
            int(round((x2 - min_x) / (max_x - min_x) * (width - 1))),
            int(round((max_y - y2) / (max_y - min_y) * (height - 1))),
        )
        cv2.line(image, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, image)


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def subtract(a, b):
    return (a[0] - b[0], a[1] - b[1])


def segment_intersection(seg_a, seg_b):
    p = tuple(seg_a["a"])
    r = subtract(tuple(seg_a["b"]), p)
    q = tuple(seg_b["a"])
    s = subtract(tuple(seg_b["b"]), q)
    denominator = cross(r, s)
    if abs(denominator) < INTERSECTION_EPSILON:
        return None

    qp = subtract(q, p)
    t = cross(qp, s) / denominator
    u = cross(qp, r) / denominator
    if -INTERSECTION_EPSILON <= t <= 1.0 + INTERSECTION_EPSILON and -INTERSECTION_EPSILON <= u <= 1.0 + INTERSECTION_EPSILON:
        return (
            max(0.0, min(1.0, t)),
            max(0.0, min(1.0, u)),
            (p[0] + r[0] * t, p[1] + r[1] * t),
        )
    return None


def point_at(segment, t):
    ax, ay = segment["a"]
    bx, by = segment["b"]
    return (ax + (bx - ax) * t, ay + (by - ay) * t)


def node_key(point):
    return (round(point[0], NODE_ROUND_DIGITS), round(point[1], NODE_ROUND_DIGITS))


def split_segments_at_intersections(segments):
    split_points = [[0.0, 1.0] for _ in segments]
    for i, seg_a in enumerate(segments):
        ax1, ay1 = seg_a["a"]
        ax2, ay2 = seg_a["b"]
        amin_x, amax_x = sorted((ax1, ax2))
        amin_y, amax_y = sorted((ay1, ay2))
        for j in range(i + 1, len(segments)):
            seg_b = segments[j]
            bx1, by1 = seg_b["a"]
            bx2, by2 = seg_b["b"]
            bmin_x, bmax_x = sorted((bx1, bx2))
            bmin_y, bmax_y = sorted((by1, by2))
            if amax_x < bmin_x or bmax_x < amin_x or amax_y < bmin_y or bmax_y < amin_y:
                continue
            hit = segment_intersection(seg_a, seg_b)
            if not hit:
                continue
            t, u, _ = hit
            split_points[i].append(t)
            split_points[j].append(u)

    edges = set()
    node_points = {}
    for segment, values in zip(segments, split_points):
        values = sorted(set(round(value, 9) for value in values))
        for t1, t2 in zip(values, values[1:]):
            if t2 - t1 < 1e-7:
                continue
            p1 = point_at(segment, t1)
            p2 = point_at(segment, t2)
            if distance(p1, p2) < MIN_OUTPUT_SEGMENT_LENGTH:
                continue
            k1 = node_key(p1)
            k2 = node_key(p2)
            if k1 == k2:
                continue
            node_points[k1] = p1
            node_points[k2] = p2
            edges.add(tuple(sorted((k1, k2))))
    return node_points, edges


def polygon_area(points):
    area = 0.0
    for i, point in enumerate(points):
        next_point = points[(i + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area * 0.5


def polygon_perimeter(points):
    return sum(distance(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))


def trace_faces(node_points, edges):
    adjacency = {key: [] for key in node_points}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    angle_maps = {}
    for node, neighbors in adjacency.items():
        px, py = node
        ordered = sorted(neighbors, key=lambda item: math.atan2(item[1] - py, item[0] - px))
        angle_maps[node] = ordered

    visited = set()
    faces = []
    for edge in edges:
        for start in edge:
            previous = edge[1] if start == edge[0] else edge[0]
            directed = (previous, start)
            if directed in visited:
                continue

            face = []
            current_directed = directed
            for _ in range(len(edges) * 2):
                if current_directed in visited:
                    break
                visited.add(current_directed)
                u, v = current_directed
                face.append(u)
                neighbors = angle_maps[v]
                try:
                    reverse_index = neighbors.index(u)
                except ValueError:
                    break
                # Clockwise turn from the incoming direction traces one side of
                # each edge. The opposite directed edge traces the adjacent side.
                next_node = neighbors[(reverse_index - 1) % len(neighbors)]
                current_directed = (v, next_node)
                if current_directed == directed:
                    break

            if len(face) < 3:
                continue
            points = [node_points[key] for key in face]
            area = polygon_area(points)
            if area <= 0:
                continue
            perimeter = polygon_perimeter(points)
            thinness = (perimeter * perimeter / area) if area else 999999.0
            if area < MIN_FACE_AREA or area > MAX_FACE_AREA:
                continue
            if thinness > MAX_FACE_THINNESS:
                continue
            faces.append(
                {
                    "area": round(area, 3),
                    "perimeter": round(perimeter, 3),
                    "thinness": round(thinness, 3),
                    "points": [[round(p[0], 3), round(p[1], 3)] for p in points],
                }
            )
    return faces


def write_faces_svg(path, bbox, faces):
    min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x
    height = max_y - min_y
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width:.3f} {height:.3f}" '
            f'width="{width:.3f}" height="{height:.3f}">\n'
        )
        f.write('<g fill="none" stroke="#ff0000" stroke-width="1">\n')
        for face in faces:
            parts = []
            for index, (x, y) in enumerate(face["points"]):
                sx = x - min_x
                sy = max_y - y
                parts.append(("M" if index == 0 else "L") + f" {sx:.3f} {sy:.3f}")
            f.write(f'<path d="{" ".join(parts)} Z"/>\n')
        f.write("</g>\n</svg>\n")


def write_faces_png(path, bbox, faces):
    min_x, min_y, max_x, max_y = bbox
    height = PREVIEW_HEIGHT
    width = max(1, int(round(height * (max_x - min_x) / (max_y - min_y))))
    image = np.full((height, width, 3), 255, np.uint8)
    for face in faces:
        pts = []
        for x, y in face["points"]:
            pts.append(
                [
                    int(round((x - min_x) / (max_x - min_x) * (width - 1))),
                    int(round((max_y - y) / (max_y - min_y) * (height - 1))),
                ]
            )
        arr = np.array(pts, np.int32)
        cv2.polylines(image, [arr], True, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, image)


def main():
    source, segments = load_segments(INPUT_JSON)
    cleaned, nodes, stats = clean_segments(segments)
    cleaned, edge_joins = join_dangling_endpoints_to_edges(cleaned)
    stats["edge_joins"] = len(edge_joins)
    bbox = segments_bbox(cleaned, source["media_box"])

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": INPUT_JSON,
                "media_box": source["media_box"],
                "segments_bbox": bbox,
                "stats": stats,
                "segments": cleaned,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(OUTPUT_NODES_JSON, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "nodes": nodes}, f, ensure_ascii=False, indent=2)

    write_svg(OUTPUT_SVG, bbox, cleaned)
    write_png(OUTPUT_PNG, bbox, cleaned)

    node_points, edges = split_segments_at_intersections(cleaned)
    faces = trace_faces(node_points, edges)
    with open(OUTPUT_FACES_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": OUTPUT_JSON,
                "segments_bbox": bbox,
                "stats": {
                    "graph_nodes": len(node_points),
                    "graph_edges": len(edges),
                    "faces": len(faces),
                },
                "faces": faces,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    write_faces_svg(OUTPUT_FACES_SVG, bbox, faces)
    write_faces_png(OUTPUT_FACES_PNG, bbox, faces)

    print(f"Input segments: {stats['input_segments']}")
    print(f"Output segments: {stats['output_segments']}")
    print(f"Endpoint clusters: {stats['endpoint_clusters']}")
    print(f"Junction nodes: {stats['junction_nodes']}")
    print(f"Edge joins: {stats['edge_joins']}")
    print(f"Graph nodes: {len(node_points)}")
    print(f"Graph edges: {len(edges)}")
    print(f"Faces: {len(faces)}")
    print(f"SVG saved: {OUTPUT_SVG}")
    print(f"PNG saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
