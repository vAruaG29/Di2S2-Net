# Portal

A small web demo for the DINOv3 + HRDecoder segmentation pipeline.

Two modes:

1. **Browse results** — pick a dataset from the gallery, see its
   stitched satellite raster on a slippy map, toggle each predicted
   class (Built-Up Area, Road, Water Body, Utility, Bridge, Railway)
   on/off with opacity sliders. Cursor coordinates & evaluation
   metrics show in side panels.
2. **Live inference** — pick an existing test image, or drag-drop a
   new `.tif` / `.ecw`. The portal triggers the full pipeline
   (`convert_to_cog → tile_raster → run_pipeline → batch_stitched_to_gpkg`)
   as a subprocess and streams its per-step timing to the UI over SSE.
   When complete, you drop straight into Browse-results view for the
   new dataset.

## Layout

```
portal/
├── backend/                       # FastAPI app
│   ├── app.py                     # entrypoint, TiTiler mount, /api/config, /api/health
│   ├── settings.py                # reads paths from data_prep.yaml + train.yaml
│   ├── jobs.py                    # in-memory job registry + SSE pubsub
│   ├── pipeline_runner.py         # subprocesses + stdout → StepTimer event parser
│   ├── gpkg_reader.py             # GPKG layer → GeoJSON (EPSG:4326) with disk cache
│   └── routes/
│       ├── datasets.py            # GET /api/datasets, GET /api/datasets/{name}
│       ├── layers.py              # GET /api/datasets/{name}/gpkg/{layer}
│       ├── upload.py              # POST /api/uploads
│       └── inference.py           # POST /api/inference, GET /api/jobs/{id}/events (SSE)
└── frontend/                      # React + Vite + MapLibre
    ├── package.json
    ├── vite.config.ts             # proxies /api + /tiles → :8000
    └── src/
        ├── App.tsx                # layout, mode tabs, state
        ├── components/
        │   ├── TopBar.tsx
        │   ├── DatasetGallery.tsx
        │   ├── MapView.tsx        # MapLibre + TiTiler raster + GeoJSON layers
        │   ├── LayerPanel.tsx     # per-class toggle / opacity / metrics
        │   ├── UploadDropzone.tsx
        │   ├── ProgressTimeline.tsx
        │   └── InferencePanel.tsx
        └── api/
            ├── client.ts          # typed fetch wrappers
            └── sse.ts             # EventSource hook
```

## Quickstart

```bash
# (One-time) install Python + frontend deps:
bash setup_env.sh

# Launch both backend (uvicorn) and frontend (vite):
bash start_portal.sh
```

Then open <http://localhost:5173>.

## API surface

| Method | Path                                  | Use |
|--------|---------------------------------------|-----|
| GET    | `/api/health`                         | Liveness + titiler-installed flag |
| GET    | `/api/config`                         | Class colours + IDs (sourced from `data_prep.yaml`) |
| GET    | `/api/datasets`                       | List datasets (anything with `outputs/stitched/<NAME>_pred.tif`) |
| GET    | `/api/datasets/{name}`                | Detail: CRS, bounds (WGS84), layer list + feature counts, metrics rows |
| GET    | `/api/datasets/{name}/gpkg/{layer}`   | GeoJSON FeatureCollection in EPSG:4326 |
| POST   | `/api/uploads`                        | Multipart upload of a raster → `data/test/<UUID>_<safe>.tif` |
| POST   | `/api/inference`                      | Start a pipeline run (`{ source, dataset_name }`) |
| GET    | `/api/jobs/{id}`                      | Snapshot of events so far |
| GET    | `/api/jobs/{id}/events`               | **SSE** stream — phase + step events parsed from pipeline stdout |
| —      | `/tiles/cog/*`                        | TiTiler endpoints: `tilejson.json`, `{z}/{x}/{y}`, `preview`, `info` |

## How progress events are produced

The pipeline's `StepTimer` (see `dinov3_hrdecoder_pipeline/inference/_timing.py`)
prints lines like:

```
⏱  [09:14:38] START  inference / batch loop
⏱  [09:21:53] DONE   inference / batch loop  (435.12s)
⏱  [09:21:53] cumulative inference / forward pass per batch: 380.2s over 330 call(s)
```

`backend/pipeline_runner.py` runs each phase as `asyncio.create_subprocess_exec`,
tails stdout, regex-matches those lines, and pushes a structured event into the
Job's pubsub queue. The SSE endpoint subscribes and forwards events to the
browser; the `ProgressTimeline` component renders them as a timeline.

## Caching

- `outputs/gpkg_cache/<dataset>/<layer>.geojson` — built on first request,
  reused while mtime ≥ the source GPKG's mtime.
- TiTiler caches in-memory by COG path (default).
- React Query caches dataset list/detail on the frontend (5 s).

## What's out of scope (for v1)

- Auth, multi-user — local demo only.
- Persistent job registry — jobs live in-memory; restart wipes them.
- Vector tiles for huge GPKGs — full-layer GeoJSON for now.
- Cloud / Docker packaging.

## Troubleshooting

- **404 on `/tiles/cog/...`** → TiTiler isn't installed in the active env. Run `pip install titiler.core` or re-run `setup_env.sh`.
- **Empty dataset gallery** → no `_pred.tif` files in `outputs/stitched/`. Run a `run_pipeline` first.
- **`prepare_labels` non-zero** → expected when there are no shapefiles for an upload; runner treats it as soft-fail and continues.
