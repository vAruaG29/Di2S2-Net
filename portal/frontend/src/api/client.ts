/** Typed wrappers around the FastAPI endpoints. */

export interface DatasetSummary {
  name: string;
  /** Dataset name with any `@@<checkpoint-id>` suffix stripped. Equals
   *  `name` for ordinary (default-checkpoint) datasets. Use this as the
   *  display title. */
  base_name?: string;
  /** Checkpoint id this result was produced with, or null for the
   *  default checkpoint. Drives the "which model" badge. */
  checkpoint?: string | null;
  origin?: "portal" | "offline";
  /** Path to the stitched prediction raster (`<name>_pred.tif`). */
  stitched: string;
  /** Alias of `stitched` — explicit that this is the prediction COG. */
  prediction_cog?: string;
  prediction_filename?: string;
  /** True iff the GPKG file is on disk AND contains ≥ 1 vector layer.
   *  An empty-but-present GPKG counts as no GPKG (so the UI doesn't
   *  show a misleading badge). */
  has_gpkg: boolean;
  /** Number of vector layers actually inside the GPKG; 0 means no
   *  usable predictions, even if the file exists. */
  layer_count?: number;
  /** Unix mtime of the resolved GPKG, in seconds. Used as a cache
   *  buster in the map so a re-run on the same dataset name picks up
   *  the new features instead of the previous run's in-memory copy. */
  gpkg_mtime?: number | null;
  has_metrics: boolean;
  /** True iff this dataset originated from an upload. The Train /
   *  Test gallery filters uploads out — they only belong in the
   *  Upload tab's "Recent uploads" pane. */
  is_upload?: boolean;
}

export interface LayerInfo {
  name: string;
  feature_count: number;
  colour: string;
}

export interface MetricRow {
  source: string;
  metric: string;
  value: string;
}

export interface DatasetDetail extends DatasetSummary {
  origin?: "portal" | "offline";
  source_cog: string | null;       // path to the original aerial COG/raster
  /** Tile-URL template for the aerial COG, with literal {z}/{x}/{y}. */
  tiles_url_template?: string | null;
  /** Legacy alias of `tiles_url_template`. */
  tilejson_url?: string | null;
  /** Single-shot preview image (full-extent, low-res) — instant first paint. */
  preview_url?: string | null;
  crs: string | null;
  bounds_wgs84: [number, number, number, number] | null;
  bounds_error?: string;
  layers: LayerInfo[];
  metrics: MetricRow[];
  /** Class names with per-tile ground-truth masks available — empty
   *  means the dataset has no GT (typically a Test image). */
  gt_layers: string[];
}

export interface ConfigResponse {
  class_colors: Record<string, string>;
  feature_classes: Record<string, number>;
}

export interface InferenceStart {
  status: "queued" | "exists";
  job_id: string | null;
  dataset_name: string;
  existing?: PreflightInfo;
}

export interface PreflightInfo {
  dataset_name: string;
  job_root: string | null;
  has_stitched: boolean;
  stitched_path: string | null;
  has_gpkg: boolean;
  gpkg_path: string | null;
  has_metrics: boolean;
  metrics_path: string | null;
  outputs_exist: boolean;
}

/** One model checkpoint the portal can run (from `pretrained/`). */
export interface CheckpointInfo {
  id: string;          // stable slug; becomes the `@@<id>` dataset suffix
  label: string;       // human-ish label for the dropdown
  filename: string;    // the .ckpt filename
  path: string;
  is_default: boolean; // the checkpoint that keeps the bare base name
  decoder_type?: string | null; // sniffed architecture (hrdecoder/upernet/…)
  compatible?: boolean;          // matches the pipeline's decoder
}

/** A checkpoint hidden from the dropdown because its architecture doesn't
 *  match the pipeline (would predict garbage if loaded). */
export interface HiddenCheckpoint {
  id: string;
  filename: string;
  decoder_type: string | null;
}

export interface CheckpointsResponse {
  checkpoints: CheckpointInfo[];   // compatible only — safe to run
  hidden?: HiddenCheckpoint[];     // filtered-out, incompatible checkpoints
  expected_decoder?: string;       // the pipeline's decoder.type
  separator: string;   // the `@@` between base name and checkpoint id
}

export interface UploadResponse {
  dataset_name: string;
  saved_to: string;
  bytes: number;
}

export interface ServerFile {
  path: string;
  name: string;
  rel: string;
  size_bytes: number;
  size_mb: number;
  ext: string;
}

export interface ServerFilesResponse {
  roots: { root: string; label: string; count: number; files: ServerFile[] }[];
  allowed_extensions: string[];
}

export interface ServerPickResponse {
  dataset_name: string;
  saved_to: string;
  linked_to: string;
  symlink: boolean;
  bytes: number;
}

/** One row from `GET /api/uploads/list` — a past upload sitting in
 *  `portal_workspace/`, with enough state to render the UploadPane's
 *  "Recent uploads" list. */
export interface UploadRow {
  dataset_name: string;       // upload_<id>_<stem>
  display_name: string;       // original filename stem the user uploaded
  source: string;             // absolute path to source.<ext>
  bytes: number;
  has_outputs: boolean;       // true if the GeoPackage is on disk
  has_stitched: boolean;
  created_at: number;         // unix mtime of the source file
}

export interface UploadListResponse {
  uploads: UploadRow[];
}

async function jget<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} → ${r.status} ${r.statusText}`);
  return (await r.json()) as T;
}

async function jpost<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} → ${r.status} ${r.statusText}`);
  return (await r.json()) as T;
}

export const api = {
  config:      () => jget<ConfigResponse>("/api/config"),
  datasets:    () => jget<DatasetSummary[]>("/api/datasets"),
  dataset:     (name: string) => jget<DatasetDetail>(`/api/datasets/${encodeURIComponent(name)}`),
  layerGeoJSON: (name: string, layer: string) =>
    jget<GeoJSON.FeatureCollection>(
      `/api/datasets/${encodeURIComponent(name)}/gpkg/${encodeURIComponent(layer)}`,
    ),
  /** Ground-truth GeoJSON (per class). First call materialises the
   *  cache and can take 5-30 s — caller should show a loading hint. */
  gtLayerGeoJSON: (name: string, layer: string) =>
    jget<GeoJSON.FeatureCollection>(
      `/api/datasets/${encodeURIComponent(name)}/gt/${encodeURIComponent(layer)}`,
    ),
  /** Model checkpoints the portal can run. */
  checkpoints: () => jget<CheckpointsResponse>("/api/checkpoints"),
  /** Preflight for a (base dataset, checkpoint) pair. `checkpoint`
   *  omitted / the default id checks the bare base name. */
  preflight: (dataset_name: string, checkpoint?: string) => {
    const qs = checkpoint ? `?checkpoint=${encodeURIComponent(checkpoint)}` : "";
    return jget<PreflightInfo>(
      `/api/inference/preflight/${encodeURIComponent(dataset_name)}${qs}`,
    );
  },
  startInference: (
    source: "existing" | "upload",
    dataset_name: string,
    force = false,
    checkpoint?: string,
  ) =>
    jpost<InferenceStart>("/api/inference", { source, dataset_name, force, checkpoint }),
  cancelJob: (job_id: string) =>
    jpost<{ id: string; status: string; cancelled: boolean; killed_proc?: boolean }>(
      `/api/jobs/${encodeURIComponent(job_id)}/cancel`, {}),
  cleanupJob: (job_id: string, hard = false) =>
    jpost<{ id: string; removed: boolean; path: string; reason?: string }>(
      `/api/jobs/${encodeURIComponent(job_id)}/cleanup${hard ? "?hard=true" : ""}`, {}),
  uploadRaster: async (file: File): Promise<UploadResponse> => {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/uploads", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`upload → ${r.status} ${r.statusText}`);
    return (await r.json()) as UploadResponse;
  },
  serverFiles: () => jget<ServerFilesResponse>("/api/server-files"),
  pickServerFile: (server_path: string, dataset_name?: string) =>
    jpost<ServerPickResponse>("/api/server-pick", { server_path, dataset_name }),
  /** Everything currently in portal_workspace/ that started life as an
   *  upload (`upload_*` job folders). Used by UploadPane's recents
   *  list — newest first. */
  listUploads: () => jget<UploadListResponse>("/api/uploads/list"),
  deleteUpload: async (dataset_name: string) => {
    const r = await fetch(`/api/uploads/list/${encodeURIComponent(dataset_name)}`, {
      method: "DELETE",
    });
    if (!r.ok) throw new Error(`delete → ${r.status} ${r.statusText}`);
    return (await r.json()) as { removed: boolean; path: string };
  },
  /** TiTiler tilejson URL for a given COG path on the server. */
  tilejsonUrl: (cogPath: string) =>
    `/tiles/cog/tilejson.json?url=${encodeURIComponent(cogPath)}`,
  /** URL that streams the dataset's GeoPackage with a Save-As dialog. */
  gpkgDownloadUrl: (name: string) =>
    `/api/datasets/${encodeURIComponent(name)}/gpkg-download`,
};
