# PDF vector and skeleton research

This folder keeps abandoned experimental code from the PDF-vector and
skeletonization investigation.

The main production path remains the raster detector in
`detect_polygons_from_image.py`.

## Why this was investigated

The raster detector extracts parcel contours from `page_1.png`. Its main
geometric limitation is that boundaries are detected along raster line edges,
not along the original centerline. Adjacent parcels may therefore have two close
parallel borders instead of one shared border.

Vertex and edge snapping improves topology, but it is not a fundamental fix:
the snap tolerance can be close to the size of real plan details. In that case
the algorithm can either leave small gaps or incorrectly collapse real narrow
spaces.

The PDF used in this project contains vector drawing commands, so a more exact
approach was tested:

1. Read PDF content streams directly.
2. Extract stroked vector segments.
3. Filter likely parcel boundaries by PDF stroke style.
4. Build a graph from vector segments.
5. Close small junction gaps.
6. Polygonize graph faces.

## What worked

`extract_pdf_vector_lines.py` successfully extracts vector stroke segments from
the PDF. A key PDF detail is that `/Contents` may be an array of streams. Those
streams must be parsed as one concatenated content stream, otherwise the current
transformation matrix can be lost between streams.

For the tested PDF, the useful parcel-boundary candidate layer was mostly:

```text
stroke = (0, 0, 0)
line_width = 0
min segment length = 20
```

This produced a much cleaner source than raster skeletonization and preserved
sharp PDF vertices where the source geometry was explicit.

`build_pdf_vector_graph.py` then tested:

- endpoint clustering;
- junction reconstruction through line intersections;
- protection against collapsing parallel offset road boundaries;
- dangling endpoint to nearby edge joining;
- segment splitting at intersections;
- half-edge face traversal.

The first graph pass found about 50 plausible closed faces on the tested plan.

## What did not justify further work

The approach is too dependent on how a particular PDF was authored.

Different source PDFs may use different:

- stroke colors;
- line widths;
- layers;
- clipping paths;
- hatch styles;
- road drawing conventions;
- text/vector glyph encodings;
- coordinate transforms;
- path fragmentation patterns.

Roads are especially problematic. In this PDF they are represented as pairs of
parallel vector lines. Removing both lines is wrong because road edges can also
serve as parcel boundaries. They should instead be classified as narrow corridor
faces after graph polygonization.

Some junctions are not single PDF vertices. They are drawn as small gaps,
facets, or short multi-segment turns. Fixing them robustly requires a dedicated
graph-normalization stage.

Overall, the improvement over the last committed raster version is not worth
the extra complexity for the current project. The raster output has small
line-width-related inaccuracies, but it is predictable, simpler, and works for
image-only sources too.

## If this is resumed later

A reasonable next version would be a separate optional backend, not a replacement
for the raster detector:

```text
input PDF
  -> vector extraction
  -> style/layer classification
  -> segment graph
  -> junction normalization
  -> face polygonization
  -> road/corridor face classification
  -> polygon.json export
```

The raster detector should remain the fallback for PNG/JPG sources and for PDFs
that contain only scanned images.

Useful files kept here:

- `extract_pdf_vector_lines.py` - PDF content stream and vector segment extractor.
- `build_pdf_vector_graph.py` - experimental graph builder and face tracer.
- `skeletonize_polygons_from_image.py` - earlier raster skeletonization attempt.

