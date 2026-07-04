import { useState } from "react";
import { Eye, EyeOff, ChevronDown, Minus, ChevronLeft } from "lucide-react";

import { DatasetDetail, LayerInfo, MetricRow } from "../api/client";
import { inferGeom, LAYER_STYLES } from "../constants";
import type { LayerState } from "../types";

interface Props {
  dataset: DatasetDetail;
  layerState: Record<string, LayerState>;
  onLayerStateChange: (next: Record<string, LayerState>) => void;
  predictionsOn: boolean;
}

/**
 * Right-side floating card — shown whenever a dataset is selected.
 * Replaces the old MapTopBar + the in-left-panel Classes section.
 *
 * Stacks vertically:
 *   1. Header   — dataset name + LIB badge + close button
 *   2. Actions  — Run inference / Download features
 *   3. Classes  — per-class layer toggles, each expandable to show
 *                 IoU / Precision / F1 / Recall pulled from the
 *                 per-dataset metrics CSV.
 */
export function ClassesFloat({
  dataset, layerState, onLayerStateChange, predictionsOn,
}: Props) {
  const layers = dataset.layers ?? [];
  const [minimized, setMinimized] = useState(false);

  // ── Minimised state: collapse to a thin vertical tab pinned to the
  //    right edge. Click anywhere on the tab to expand again.
  if (minimized) {
    return (
      <button
        onClick={() => setMinimized(false)}
        // Lives inside the App-level right-side wrapper (which is
        // pointer-events:none and inset 12 px from the viewport edge).
        // `-right-3` pushes the tab back out to hug the viewport edge
        // and `pointer-events-auto` re-enables clicks.
        className="absolute top-0 -right-3 z-20 pointer-events-auto
                   flex flex-col items-center gap-2 px-1.5 py-3
                   bg-ink-800/95 border border-ink-700 border-r-0 backdrop-blur shadow-lg
                   text-ink-300 hover:text-accent-500"
        title="Expand classes panel"
      >
        <ChevronLeft className="w-4 h-4" />
        <span
          className="text-[10px] tracking-wider uppercase font-medium"
          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
        >
          Classes ({layers.length})
        </span>
      </button>
    );
  }

  return (
    <aside
      // Positioning is now owned by the parent flex-column wrapper in
      // App.tsx — this card just lays out its own content. `min-h-0`
      // lets the flex container distribute vertical space; the
      // internal scroll area below uses `flex-1 overflow-y-auto` to
      // soak up whatever room it has.
      className="w-full min-h-0 bg-ink-800/95 border border-ink-700
                 backdrop-blur flex flex-col overflow-hidden shadow-lg
                 pointer-events-auto"
    >
      <div className="px-3 py-2 border-b border-ink-700 flex items-center gap-1 flex-shrink-0">
        <div className="text-[10px] uppercase tracking-wider text-ink-400 font-medium flex-1">
          Classes ({layers.length})
        </div>
        <button
          onClick={() => setMinimized(true)}
          className="w-5 h-5 grid place-items-center text-ink-400 hover:text-ink-300 flex-shrink-0"
          title="Minimise to right edge"
        >
          <Minus className="w-3.5 h-3.5" />
        </button>
      </div>

      {layers.length > 0 ? (
        <div className="flex-1 overflow-y-auto px-3 py-2 min-h-0">
          <ClassRows
            layers={layers}
            state={layerState}
            onChange={onLayerStateChange}
            dimmed={!predictionsOn}
            metrics={dataset.metrics ?? []}
          />
        </div>
      ) : (
        <div className="px-4 py-6 text-[11px] text-ink-400">
          No GeoPackage layers for this dataset.
        </div>
      )}
    </aside>
  );
}

// ── per-class rows (moved out of the old LeftPanel) ────────────────────

interface ClassMetricSet {
  iou:  number | null;
  f1:   number | null;
  prec: number | null;
  rec:  number | null;
}

function pickClassMetrics(metrics: MetricRow[], layerName: string): ClassMetricSet {
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const target = norm(layerName.replace(/_(type|poly|line|point)$/i, ""));

  const out: ClassMetricSet = { iou: null, f1: null, prec: null, rec: null };
  for (const m of metrics) {
    const key = (m as any).metric as string | undefined;
    if (!key || !key.includes("/")) continue;
    const [t, cls] = key.split("/");
    const candidate = norm(cls);
    if (!candidate || (candidate !== target && !candidate.includes(target) && !target.includes(candidate))) continue;
    const v = Number((m as any).value);
    if (isNaN(v)) continue;
    const lt = t.toLowerCase();
    if (lt === "iou")            out.iou  = v;
    else if (lt === "f1")        out.f1   = v;
    else if (lt === "precision") out.prec = v;
    else if (lt === "recall")    out.rec  = v;
  }
  return out;
}

function ClassRows({
  layers, state, onChange, dimmed, metrics,
}: {
  layers: LayerInfo[];
  state: Record<string, LayerState>;
  onChange: (next: Record<string, LayerState>) => void;
  dimmed: boolean;
  metrics: MetricRow[];
}) {
  const [expanded, setExpanded] = useState<string | null>(null);

  const toggle = (name: string) => {
    onChange({
      ...state,
      [name]: {
        visible: !(state[name]?.visible ?? false),
        opacity: state[name]?.opacity ?? 0.7,
      },
    });
  };
  const setOpacity = (name: string, v: number) => {
    onChange({
      ...state,
      [name]: {
        visible: state[name]?.visible ?? true,
        opacity: v,
      },
    });
  };

  return (
    <div className={`space-y-1 ${dimmed ? "opacity-50 pointer-events-none" : ""}`}>
      {layers.map((l) => {
        const style = LAYER_STYLES[l.name];
        const colour = style?.colour ?? l.colour ?? "#ff5600";
        const label = style?.label ?? l.name;
        const geom = style?.geom ?? inferGeom(l.name);
        const s = state[l.name] ?? { visible: false, opacity: 0.7 };
        const isOpen = expanded === l.name;
        const mset = isOpen ? pickClassMetrics(metrics, l.name) : null;

        return (
          <div key={l.name} className={`group ${s.visible ? "" : "opacity-60"}`}>
            <div className="flex items-center h-7 gap-1.5">
              <button
                onClick={() => setExpanded(isOpen ? null : l.name)}
                className="w-3.5 h-3.5 grid place-items-center text-ink-400 hover:text-ink-300"
                title={isOpen ? "Hide metrics" : "Show class metrics"}
              >
                <ChevronDown className={`w-3 h-3 transition-transform ${isOpen ? "rotate-0" : "-rotate-90"}`} />
              </button>
              <div className="w-2.5 h-2.5 flex-shrink-0"
                   style={{ background: colour }} />
              <button
                onClick={() => setExpanded(isOpen ? null : l.name)}
                className="flex-1 text-left text-xs text-ink-300 truncate hover:text-accent-500"
              >
                {label}
              </button>
              <span
                className="text-[10px] text-ink-400 font-mono tabular-nums"
                title={`${l.feature_count.toLocaleString()} feature${l.feature_count === 1 ? "" : "s"} predicted in this class`}
              >
                {l.feature_count.toLocaleString()}
              </span>
              <span className="text-[9px] uppercase tracking-wider text-ink-400 px-1
                                bg-ink-700/60">{geom}</span>
              <button
                onClick={() => toggle(l.name)}
                className="w-4 h-4 grid place-items-center"
                title={s.visible ? "Hide layer on map" : "Show layer on map"}
              >
                {s.visible
                  ? <Eye    className="w-3 h-3 text-ink-300" />
                  : <EyeOff className="w-3 h-3 text-ink-500" />}
              </button>
            </div>

            {s.visible && (
              <div className="pl-6 pr-1 pb-1 flex items-center gap-2">
                <span className="text-[10px] text-ink-400 w-10">opacity</span>
                <input
                  type="range"
                  min={0} max={1} step={0.05}
                  value={s.opacity}
                  onChange={(e) => setOpacity(l.name, parseFloat(e.target.value))}
                  className="flex-1"
                />
                <span className="text-[10px] text-ink-400 font-mono tabular-nums w-8 text-right">
                  {Math.round(s.opacity * 100)}%
                </span>
              </div>
            )}

            {isOpen && mset && <ClassMetricsCard mset={mset} />}
          </div>
        );
      })}
    </div>
  );
}

function ClassMetricsCard({ mset }: { mset: ClassMetricSet }) {
  const cells: Array<[string, number | null]> = [
    ["IoU",       mset.iou],
    ["Precision", mset.prec],
    ["F1",        mset.f1],
    ["Recall",    mset.rec],
  ];
  const empty = cells.every(([, v]) => v == null);
  if (empty) {
    return (
      <div className="ml-6 mt-1 mb-2 px-2 py-1.5 bg-ink-900 border border-ink-700 text-[10px] text-ink-400 italic">
        no per-class metrics for this dataset
      </div>
    );
  }
  return (
    <div className="ml-6 mt-1 mb-2 px-1.5 py-1.5 bg-ink-900 border border-ink-700 grid grid-cols-2 gap-x-1.5 gap-y-1">
      {cells.map(([label, v]) => (
        <div key={label} className="flex items-baseline justify-between gap-1 min-w-0">
          <span className="text-[10px] uppercase tracking-tight text-ink-400 font-medium truncate">
            {label}
          </span>
          <span className="text-[11px] font-semibold font-mono tabular-nums text-ink-300 flex-shrink-0">
            {v == null ? "—" : v.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}
