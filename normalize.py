
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize.py
  Приводит исходный набор полигонов к целевому формату и масштабу.

Поддержка:
- Исходник как список колец:  [ [ [x,y], ... ], [ ... ], ... ]
- Исходник как объект {"data":[{"coordinates":[[...]]}, ...]} — возьмём "coordinates"
- Автомасштаб под bbox образца (sample), либо ручной scale.
- Вывод — массив объектов со схемой (поля как в твоём примере).

CLI:
  python normalize.py --input detected_polygons.json --schema polygon_sample.json --out polygon.normalized.json --auto
  python normalize.py --input detected_polygons.json --schema polygon_sample.json --out polygon.normalized.json --scale 0.15

Опции:
  --status "sale35000"     # статус по умолчанию
  --start-number 1         # с какого номера начинать number/names
  --idtur-width 5          # ширина idtur с ведущими нулями
  --no-center              # не центрировать в целевой bbox (только масштабировать от исходного центра)
  --fit-padding 0.98       # запас при автоfit (0..1)
"""

import json
from pathlib import Path
import argparse
from typing import Any, Dict, List, Tuple

def read_input(path: Path) -> List[List[List[float]]]:
    """Читает исходник. Возвращает список колец (каждое кольцо — список [x,y])."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    # Вариант 1: список колец напрямую
    if isinstance(obj, list) and obj and isinstance(obj[0], list):
        # предполагаем [[ [x,y], ... ], ...] либо [ [x,y], ... ] (одно кольцо)
        # нормализуем к списку колец
        if obj and obj and obj and obj and obj and isinstance(obj[0][0], (list, tuple)) and len(obj[0][0]) == 2:
            # это список точек => одно кольцо
            if all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in obj):
                return [obj]
            # это список колец
            return obj
    # Вариант 2: объект с "data"
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        rings = []
        for item in obj["data"]:
            coords = item.get("coordinates")
            if isinstance(coords, list) and coords and isinstance(coords[0], list):
                # ожидаем [[points]] — берём первое кольцо
                first_ring = coords[0] if coords and isinstance(coords[0], list) else []
                rings.append(first_ring)
        return rings
    raise ValueError("Не распознал формат исходника. Ожидал список колец либо объект с data[].coordinates.")

def read_sample_bbox(sample_path: Path) -> Tuple[float, float, float, float]:
    """Читает bbox из образца: (min_x, min_y, max_x, max_y)."""
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    return bbox_of_dataset(sample)

def bbox_of_dataset(dataset: Dict[str, Any]) -> Tuple[float, float, float, float]:
    min_x = float("inf"); min_y = float("inf")
    max_x = float("-inf"); max_y = float("-inf")
    items = dataset.get("data", [])
    for it in items:
        for ring in it.get("coordinates", []):
            for pt in ring:
                x, y = pt
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y
    if min_x == float("inf"):
        raise ValueError("Не удалось вычислить bbox образца: пустые coordinates.")
    return (min_x, min_y, max_x, max_y)

def bbox_of_rings(rings: List[List[List[float]]]) -> Tuple[float, float, float, float]:
    min_x = float("inf"); min_y = float("inf")
    max_x = float("-inf"); max_y = float("-inf")
    for ring in rings:
        for x, y in ring:
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
    if min_x == float("inf"):
        raise ValueError("Не удалось вычислить bbox исходника: пустые rings.")
    return (min_x, min_y, max_x, max_y)

def transform_rings(rings: List[List[List[float]]],
                    scale: float,
                    src_center: Tuple[float, float],
                    dst_center: Tuple[float, float] = None,
                    center_to_dst: bool = True) -> List[List[List[float]]]:
    """Масштабирование вокруг src_center, затем опциональный перенос к dst_center."""
    sx, sy = src_center
    if dst_center is None:
        dst_center = src_center
    dx, dy = dst_center

    out: List[List[List[float]]] = []
    for ring in rings:
        new_ring = []
        for x, y in ring:
            x0 = (x - sx) * scale + (dx if center_to_dst else sx)
            y0 = (y - sy) * scale + (dy if center_to_dst else sy)
            new_ring.append([x0, y0])
        # закрыть кольцо, если нужно
        if new_ring and new_ring[0] != new_ring[-1]:
            new_ring.append(new_ring[0])
        out.append(new_ring)
    return out

def normalize_records(rings: List[List[List[float]]],
                      status: str,
                      start_number: int,
                      idtur_width: int) -> Dict[str, Any]:
    """Собирает выводной объект с нужной схемой полей."""
    key_order = [
        "id",
        "number",
        "names",
        "sizesotki",
        "pricesotka",
        "price",
        "kadastr",
        "kadastrurl",
        "idtur",
        "urltur",
        "images",
        "datebron",
        "status",
        "coordinates",
    ]
    out = {"inc": 0, "data": []}
    for i, ring in enumerate(rings, start=1):
        number_val = str(start_number + i - 1)
        record = {
            "id": i,
            "number": number_val,
            "names": number_val,
            "sizesotki": "0",
            "pricesotka": "0",
            "price": "0",
            "kadastr": 0,
            "kadastrurl": 0,
            "idtur": number_val.zfill(idtur_width),
            "urltur": 0,
            "images": 0,
            "datebron": 0,
            "status": status,
            "coordinates": [ring],
        }
        ordered = {k: record[k] for k in key_order}
        out["data"].append(ordered)
    out["inc"] = len(out["data"])
    return out

def main():
    ap = argparse.ArgumentParser(description="Нормализация и масштабирование полигонов под целевую схему.")
    ap.add_argument("--input", required=True, help="Путь к исходному JSON (список колец или объект с data[].coordinates)")
    ap.add_argument("--schema", required=True, help="Путь к образцу (polygon_sample.json) для автоfit bbox")
    ap.add_argument("--out", default="polygon.normalized.json", help="Путь вывода")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--auto", action="store_true", help="Автомасштаб под bbox образца (fit inside)")
    group.add_argument("--scale", type=float, help="Ручной scale (например, 0.15)")
    ap.add_argument("--fit-padding", type=float, default=0.98, help="Запас при автоfit (0..1), по умолчанию 0.98")
    ap.add_argument("--status", default="sale35000", help="Статус для всех объектов (по умолчанию sale35000)")
    ap.add_argument("--start-number", type=int, default=1, help="С какого номера начинать number/names (по умолчанию 1)")
    ap.add_argument("--idtur-width", type=int, default=5, help="Ширина idtur (ведущие нули), по умолчанию 5")
    ap.add_argument("--no-center", action="store_true", help="Не центрировать в целевой bbox (оставить центр исходника)")

    args = ap.parse_args()
    input_path = Path(args.input)
    schema_path = Path(args.schema)
    out_path = Path(args.out)

    rings = read_input(input_path)

    # BBox исходника
    d_min_x, d_min_y, d_max_x, d_max_y = bbox_of_rings(rings)
    d_w = d_max_x - d_min_x
    d_h = d_max_y - d_min_y
    d_cx = (d_min_x + d_max_x) / 2.0
    d_cy = (d_min_y + d_max_y) / 2.0

    # Определяем масштаб
    if args.scale is not None:
        scale = args.scale
        s_cx = d_cx
        s_cy = d_cy
    else:
        # Автофит в bbox образца
        s_min_x, s_min_y, s_max_x, s_max_y = read_sample_bbox(schema_path)
        s_w = s_max_x - s_min_x
        s_h = s_max_y - s_min_y
        if d_w == 0 or d_h == 0 or s_w == 0 or s_h == 0:
            raise ValueError("Нулевой размер bbox у исходника или образца — масштабирование невозможно.")
        scale = min(s_w / d_w, s_h / d_h) * args.fit_padding
        s_cx = (s_min_x + s_max_x) / 2.0
        s_cy = (s_min_y + s_max_y) / 2.0

    # Куда центрировать
    if args.no-center:
        dst_center = (d_cx, d_cy)  # центрируем на себя
    else:
        # если задан ручной scale — по умолчанию центрируем в центр образца
        if args.scale is not None:
            s_min_x, s_min_y, s_max_x, s_max_y = read_sample_bbox(schema_path)
            s_cx = (s_min_x + s_max_x) / 2.0
            s_cy = (s_min_y + s_max_y) / 2.0
        dst_center = (s_cx, s_cy)

    # Трансформация
    rings_tr = transform_rings(rings, scale=scale, src_center=(d_cx, d_cy), dst_center=dst_center, center_to_dst=True)

    # Нормализация структуры и полей
    result = normalize_records(rings_tr, status=args.status, start_number=args.start_number, idtur_width=args.idtur_width)

    # Сохранение
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Диагностика в консоль
    diag = {
        "src_bbox": [d_min_x, d_min_y, d_max_x, d_max_y],
        "dst_bbox_sample": [s_min_x, s_min_y, s_max_x, s_max_y] if 's_min_x' in locals() else None,
        "scale_used": scale,
        "center_dst": dst_center,
        "objects": len(result["data"]),
        "output": str(out_path)
    }
    print(json.dumps(diag, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
