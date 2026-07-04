import { useState } from "react";
import { Eye, EyeOff, Minus, ChevronLeft } from "lucide-react";

import { LAYER_STYLES, inferGeom } from "../constants";
import type { LayerState } from "../types";

interface Props {
  /** Class names that actually have GT pixels for the current
   *  dataset — comes straight from `DatasetDetail.gt_layers`. */
  gtLayers: string[];
  /** Per-class visibility / opacity state for GT (mirror of the
   *  predictions `layerState`). */
  state: Record<string, LayerState>;
  onChange: (next: Record<string, LayerState>) => void;
  /** Master toggle from the LeftFloat's "Ground Truth" row — when
   *  false the per-class rows are dimmed + non-interactive. */
  masterOn: boolean;
}

/**
 * Right-bottom floating card listing the dataset's ground-truth
 * classes. Mirrors the prediction `ClassesFloat` on the top-right,
 * but rendered in the GT-green palette so the user can clearly see
 * which panel governs which overlay.
 *
 * Like ClassesFloat:
 *   - has its own minimise → vertical right-edge tab (sits below the
 *     predictions panel's tab so they don't overlap)
 *   - per-row eye toggle controls map visibility
 *   - per-row opacity slider appears when the row is visible
 *
 * Differs from ClassesFloat:
 *   - no per-class metrics block (those metrics live on the
 *     predictions side — there's no separate "GT metric")
 *   - uses the green palette
 */
export function GtClassesFloat({
  gtLayers, state, onChange, masterOn,
}: Props) {
  const [minimized, setMinimized] = useState(false);

  const toggle = (name: string) => {
    onChange({
      ...state,
      [name]: {
        visible: !(state[name]?.visible ?? true),
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

  // Minimised state — vertical tab pinned to the right edge, positioned
  // BELOW the predictions panel's minimise tab (top-[100px]) so the two
  // don't overlap if both are collapsed simultaneously.
  if (minimized) {
    return (
      <button
        onClick={() => setMinimized(false)}
        // Lives inside the right-side flex-column wrapper. `bottom-0`
        // anchors it to the bottom of the wrapper (right above the
        // footer), `-right-3` pushes it back out to viewport's right
        // edge, `pointer-events-auto` re-enables clicks through the
        // wrapper's `pointer-events:none`.
        className="absolute bottom-0 -right-3 z-20 pointer-events-auto
                   flex flex-col items-center gap-2 px-1.5 py-3
                   bg-ink-800/95 border border-emerald-700/40 border-r-0 backdrop-blur shadow-lg
                   text-gt hover:text-emerald-300"
        title="Expand ground-truth classes panel"
      >
        <ChevronLeft className="w-4 h-4" />
        <span
          className="text-[10px] tracking-wider uppercase font-medium"
          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
        >
          GT ({gtLayers.length})
        </span>
      </button>
    );
  }

  return (
    <aside
      // Positioning owned by the parent flex-column wrapper in App.tsx.
      className={`w-full min-h-0 bg-ink-800/95 border border-emerald-700/40
                  backdrop-blur flex flex-col overflow-hidden shadow-lg
                  pointer-events-auto
                  ${masterOn ? "" : "opacity-60"}`}
    >
      <div className="px-3 py-2 border-b border-ink-700 flex items-center gap-1 flex-shrink-0">
        <span className="w-2 h-2 bg-gt rounded-sm flex-shrink-0" />
        <div className="text-[10px] uppercase tracking-wider text-gt font-medium flex-1">
          Ground Truth ({gtLayers.length})
        </div>
        <button
          onClick={() => setMinimized(true)}
          className="w-5 h-5 grid place-items-center text-ink-400 hover:text-ink-300 flex-shrink-0"
          title="Minimise to right edge"
        >
          <Minus className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-2 min-h-0">
        {gtLayers.length === 0 ? (
          <div className="text-[11px] text-ink-400 italic">
            No ground-truth classes present in this dataset's masks.
          </div>
        ) : (
          <div className={`space-y-1 ${masterOn ? "" : "pointer-events-none"}`}>
            {gtLayers.map((name) => {
              const style = LAYER_STYLES[name];
              const label = style?.label ?? name;
              const geom = style?.geom ?? inferGeom(name);
              const s = state[name] ?? { visible: true, opacity: 0.7 };
              return (
                <div key={name} className={`group ${s.visible ? "" : "opacity-60"}`}>
                  <div className="flex items-center h-7 gap-1.5">
                    <div className="w-2.5 h-2.5 flex-shrink-0 bg-gt" />
                    <span className="flex-1 text-xs text-ink-300 truncate">
                      {label}
                    </span>
                    <span className="text-[9px] uppercase tracking-wider text-ink-400 px-1
                                      bg-ink-700/60">
                      {geom}
                    </span>
                    <button
                      onClick={() => toggle(name)}
                      className="w-4 h-4 grid place-items-center"
                      title={s.visible ? "Hide GT class on map" : "Show GT class on map"}
                    >
                      {s.visible
                        ? <Eye    className="w-3 h-3 text-ink-300" />
                        : <EyeOff className="w-3 h-3 text-ink-500" />}
                    </button>
                  </div>
                  {s.visible && (
                    <div className="pl-4 pr-1 pb-1 flex items-center gap-2">
                      <span className="text-[10px] text-ink-400 w-10">opacity</span>
                      <input
                        type="range"
                        min={0} max={1} step={0.05}
                        value={s.opacity}
                        onChange={(e) => setOpacity(name, parseFloat(e.target.value))}
                        className="flex-1"
                      />
                      <span className="text-[10px] text-ink-400 font-mono tabular-nums w-8 text-right">
                        {Math.round(s.opacity * 100)}%
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}
