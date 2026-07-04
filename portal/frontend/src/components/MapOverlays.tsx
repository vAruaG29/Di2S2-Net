import {
  Plus, Minus, Moon, Sun, Map as MapIcon,
  SplitSquareHorizontal,
} from "lucide-react";
import type { MetricRow } from "../api/client";
import { BASE_STYLES } from "../constants";
import type { LoadStatus } from "./MapView";

// ── Legend (top-left) ──────────────────────────────────────────────────

export function Legend({ hasGroundTruth, showGt }: { hasGroundTruth: boolean; showGt: boolean }) {
  return (
    <div className="absolute top-3 left-3 px-3 py-2 bg-ink-800/90 border border-ink-700
                    text-[11px] flex items-center gap-3 font-medium">
      <span className="text-[10px] uppercase tracking-wider text-ink-400">Legend</span>
      <Swatch colour="#ff5600" label="Prediction" />
      {hasGroundTruth && showGt && <Swatch colour="#00B894" label="Ground truth" />}
    </div>
  );
}

function Swatch({ colour, label }: { colour: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-2.5 h-2.5 border" style={{ borderColor: colour }} />
      <span className="text-ink-300">{label}</span>
    </div>
  );
}

// ── Tool palette (top-right) — just the swipe toggle ────────────────────

export function ToolPalette({
  swipeOn, onToggleSwipe,
}: {
  swipeOn: boolean;
  onToggleSwipe: () => void;
}) {
  return (
    <div className="absolute top-3 right-3 bg-ink-800 border border-ink-700">
      <button
        title={swipeOn ? "Hide swipe slider" : "Compare imagery vs predictions"}
        onClick={onToggleSwipe}
        className={`w-8 h-8 grid place-items-center
                    ${swipeOn ? "bg-accent-500 text-white"
                              : "text-ink-400 hover:text-ink-300 hover:bg-ink-700/50"}`}
      >
        <SplitSquareHorizontal className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

// ── Zoom controls (bottom-right) ──────────────────────────────────────

export function ZoomControls({
  onZoomIn, onZoomOut,
}: {
  onZoomIn?: () => void;
  onZoomOut?: () => void;
}) {
  return (
    <div className="absolute bottom-3 right-3 bg-ink-800 border border-ink-700 flex flex-col">
      <button onClick={onZoomIn}
              className="w-8 h-8 grid place-items-center text-ink-300 hover:bg-ink-700/50">
        <Plus className="w-3 h-3" />
      </button>
      <div className="h-px bg-ink-700" />
      <button onClick={onZoomOut}
              className="w-8 h-8 grid place-items-center text-ink-300 hover:bg-ink-700/50">
        <Minus className="w-3 h-3" />
      </button>
    </div>
  );
}

// ── Basemap switcher (bottom-right, above zoom) ───────────────────────

export function BaseStyleSwitcher({
  style, setStyle,
}: {
  style: keyof typeof BASE_STYLES;
  setStyle: (s: keyof typeof BASE_STYLES) => void;
}) {
  return (
    <div className="absolute bottom-3 right-14 flex items-center gap-1 p-1
                    bg-ink-800/90 border border-ink-700">
      {(Object.keys(BASE_STYLES) as Array<keyof typeof BASE_STYLES>).map((k) => (
        <button
          key={k}
          onClick={() => setStyle(k)}
          title={k}
          className={`px-2 py-1 text-[10px] uppercase tracking-wider
                      ${style === k
                        ? "bg-accent-500/20 text-accent-500"
                        : "text-ink-400 hover:text-ink-300"}`}
        >
          {k === "dark" ? <Moon className="w-3 h-3 inline mr-1" />
           : k === "positron" ? <Sun className="w-3 h-3 inline mr-1" />
           : <MapIcon className="w-3 h-3 inline mr-1" />}
          {k}
        </button>
      ))}
    </div>
  );
}

// ── IoU panel (bottom-left, only when training + GT visible) ──────────
//
// Backing CSV format is long: `source,metric,value` with one row per
// metric. We surface the five aggregate rows shipped by
// stitch_and_evaluate.run_evaluate(): mIoU, mF1, mPrecision, mRecall,
// overall_accuracy. `pickMetric` also tolerates a wide-format row
// (e.g. summary_report.csv) where the metric names are column keys.

const IOU_KEYS       = ["mIoU", "iou", "mean_iou", "miou"];
const PRECISION_KEYS = ["mPrecision", "precision", "mean_precision", "mprecision"];
const ACCURACY_KEYS  = ["overall_accuracy", "accuracy", "overall accuracy"];

export function IoUPanel({ metrics }: { metrics: MetricRow[] }) {
  const iou  = pickMetric(metrics, IOU_KEYS);
  const prec = pickMetric(metrics, PRECISION_KEYS);
  const acc  = pickMetric(metrics, ACCURACY_KEYS);
  if (iou == null && prec == null && acc == null) return null;
  return (
    <div className="absolute bottom-3 left-3 w-64 bg-ink-800/90 border border-ink-700 p-3">
      <div className="text-[10px] uppercase tracking-wider text-ink-400 font-medium mb-2">
        Prediction vs Ground Truth
      </div>
      <div className="grid grid-cols-3 gap-2">
        <Metric k="IoU"       v={iou}  color="#00B894" />
        <Metric k="Precision" v={prec} />
        <Metric k="Accuracy"  v={acc}  color="#F1C40F" />
      </div>
    </div>
  );
}

function Metric({ k, v, color }: { k: string; v: number | null; color?: string }) {
  return (
    <div>
      <div className="text-[9px] uppercase tracking-wider text-ink-400 font-medium">{k}</div>
      <div
        className="text-base font-semibold font-mono tabular-nums mt-0.5"
        style={{ color: color ?? "#ebebeb" }}
      >
        {v == null ? "—" : v.toFixed(2)}
      </div>
    </div>
  );
}

function pickMetric(
  metrics: MetricRow[],
  keys: string[],
): number | null {
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const needles = new Set(keys.map(norm));
  for (const m of metrics) {
    // Long format: row carries the metric name in `metric`.
    const metricKey = (m as any).metric;
    if (typeof metricKey === "string" && needles.has(norm(metricKey))) {
      const n = Number((m as any).value);
      if (!isNaN(n)) return n;
    }
    // Wide format: metric names are the column keys themselves (e.g.
    // summary_report.csv → {dataset, mIoU, mF1, …}).
    for (const [k, v] of Object.entries(m)) {
      if (needles.has(norm(k))) {
        const n = Number(v);
        if (!isNaN(n)) return n;
      }
    }
  }
  return null;
}

// ── Loading bar (top centre) ──────────────────────────────────────────

export function LoadingBar({ status }: { status: LoadStatus }) {
  const active = status.imageryLoading || status.tilesInFlight > 0 || status.layersInFlight > 0;
  if (!active) return null;

  const parts: string[] = [];
  if (status.imageryLoading || status.tilesInFlight > 0) {
    parts.push(status.tilesInFlight > 0
      ? `aerial COG · ${status.tilesInFlight} tile${status.tilesInFlight === 1 ? "" : "s"}`
      : "aerial COG");
  }
  if (status.layersInFlight > 0) {
    parts.push(`${status.layersInFlight} / ${Math.max(status.layersTotal, status.layersInFlight)} layer${status.layersTotal === 1 ? "" : "s"}`);
  }

  return (
    <div className="pointer-events-none absolute top-0 left-0 right-0 z-10">
      <div className="h-[3px] w-full overflow-hidden bg-accent-500/15">
        <div className="h-full w-1/3 bg-accent-500 animate-[shimmer_1.2s_ease-in-out_infinite]" />
      </div>
      <div className="mx-auto mt-2 inline-flex items-center gap-2 px-2.5 py-1 bg-ink-800/95
                       border border-ink-700 text-[10px] font-mono text-ink-300">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent-500 animate-pulse" />
        loading {parts.join(" + ")}
      </div>
    </div>
  );
}
