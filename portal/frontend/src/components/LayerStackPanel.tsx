import { useState } from "react";
import { Eye, EyeOff, Minus, ChevronLeft } from "lucide-react";

import type { StackState } from "../types";

interface Props {
  stack: StackState;
  onStackChange: (next: StackState) => void;
  imageryLabel?: string | null;
  hasGroundTruth?: boolean;
  predLayerCount: number;
}

/**
 * The 4-row layer stack (Predictions / Ground Truth / Image / Basemap)
 * lifted out of the old LeftFloat. Lives on the right side now,
 * underneath ClassesFloat — only mounts when a dataset is selected.
 *
 * Minimises to a thin tab on the right edge (mirrors ClassesFloat).
 */
export function LayerStackPanel({
  stack, onStackChange, imageryLabel, hasGroundTruth, predLayerCount,
}: Props) {
  const [minimized, setMinimized] = useState(false);

  if (minimized) {
    return (
      <button
        onClick={() => setMinimized(false)}
        className="absolute top-1/2 -translate-y-1/2 -right-3 z-20 pointer-events-auto
                   flex flex-col items-center gap-2 px-1.5 py-3
                   bg-ink-800/95 border border-ink-700 border-r-0 backdrop-blur shadow-lg
                   text-ink-300 hover:text-accent-500"
        title="Expand layers panel"
      >
        <ChevronLeft className="w-4 h-4" />
        <span
          className="text-[10px] tracking-wider uppercase font-medium"
          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
        >
          Layers
        </span>
      </button>
    );
  }

  return (
    <aside
      className="w-full min-h-0 bg-ink-800/95 border border-ink-700 backdrop-blur
                 flex flex-col overflow-hidden shadow-lg pointer-events-auto"
    >
      <div className="px-3 py-2 border-b border-ink-700 flex items-center gap-1 flex-shrink-0">
        <div className="text-[10px] uppercase tracking-wider text-ink-400 font-medium flex-1">
          Layers
        </div>
        <span className="text-[9px] text-ink-500 font-mono">top → bottom</span>
        <button
          onClick={() => setMinimized(true)}
          className="w-5 h-5 grid place-items-center text-ink-400 hover:text-ink-300 ml-1"
          title="Minimise"
        >
          <Minus className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="px-3 py-2 flex flex-col">
        <LayerRow
          tag="04"
          name="Predictions"
          sub={`${predLayerCount} class layer${predLayerCount === 1 ? "" : "s"}`}
          color="#ff5600"
          preview="vector"
          on={stack.predictions}
          onToggle={() => onStackChange({ ...stack, predictions: !stack.predictions })}
        />
        <LayerRow
          tag="03"
          name="Ground Truth"
          sub={hasGroundTruth ? "Verified labels" : "no GT for this dataset"}
          color="#00B894"
          preview="vector-gt"
          disabled={!hasGroundTruth}
          on={stack.groundTruth && !!hasGroundTruth}
          onToggle={() => onStackChange({ ...stack, groundTruth: !stack.groundTruth })}
        />
        <LayerRow
          tag="02"
          name="Image (RGB)"
          sub={imageryLabel ?? "—"}
          color="#888888"
          preview="tiff"
          on={stack.imagery}
          onToggle={() => onStackChange({ ...stack, imagery: !stack.imagery })}
        />
        <LayerRow
          tag="01"
          name="Basemap"
          sub="OSM Dark Matter"
          color="#444444"
          preview="base"
          on={stack.basemap}
          onToggle={() => onStackChange({ ...stack, basemap: !stack.basemap })}
          last
        />
      </div>
    </aside>
  );
}

function LayerRow({
  tag, name, sub, color, preview, on, onToggle, last, disabled,
}: {
  tag: string;
  name: string;
  sub: string;
  color: string;
  preview: "tiff" | "base" | "vector" | "vector-gt";
  on: boolean;
  onToggle: () => void;
  last?: boolean;
  disabled?: boolean;
}) {
  return (
    <div
      className={`flex items-center gap-2 py-1.5
                  ${last ? "" : "border-b border-ink-700"}
                  ${on ? "opacity-100" : "opacity-50"}
                  ${disabled ? "opacity-30" : ""}`}
    >
      <LayerThumb type={preview} color={color} />
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-1.5">
          <span className="text-[9px] text-ink-400 font-mono tracking-wider">{tag}</span>
          <span className="text-xs font-semibold text-ink-300">{name}</span>
        </div>
        <div className="text-[10px] text-ink-400 font-mono truncate">{sub}</div>
      </div>
      <button
        onClick={disabled ? undefined : onToggle}
        disabled={disabled}
        className={`w-6 h-6 grid place-items-center
                    ${on ? "text-ink-300" : "text-ink-500"}
                    ${disabled ? "cursor-not-allowed" : "hover:text-accent-500"}`}
      >
        {on ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
      </button>
    </div>
  );
}

function LayerThumb({ type, color }: { type: string; color: string }) {
  const wrap = "border border-ink-700 flex-shrink-0";
  if (type === "tiff") {
    return (
      <svg width="24" height="24" viewBox="0 0 28 28" className={wrap}>
        <rect width="28" height="28" fill="#2a2520" />
        <rect x="2" y="2" width="7" height="5" fill="#7a3a26" />
        <rect x="10" y="2" width="5" height="5" fill="#8a4a30" />
        <rect x="17" y="2" width="9" height="7" fill="#6e3520" />
        <rect x="2" y="9" width="9" height="7" fill="#a05a3d" />
        <rect x="13" y="11" width="7" height="5" fill="#5c4a3a" />
        <rect x="2" y="18" width="6" height="8" fill="#264a20" />
        <rect x="10" y="18" width="10" height="8" fill="#7a3a26" />
        <rect x="22" y="11" width="4" height="15" fill="#264a20" />
      </svg>
    );
  }
  if (type === "base") {
    return (
      <svg width="24" height="24" viewBox="0 0 28 28" className={wrap}>
        <rect width="28" height="28" fill="#1c1c1c" />
        <line x1="0" y1="9"  x2="28" y2="9"  stroke="#2a2a2a" strokeWidth="2" />
        <line x1="0" y1="19" x2="28" y2="19" stroke="#2a2a2a" strokeWidth="2" />
        <line x1="12" y1="0" x2="12" y2="28" stroke="#2a2a2a" strokeWidth="2" />
        <rect x="2"  y="2"  width="8"  height="5" fill="#262626" />
        <rect x="14" y="11" width="12" height="6" fill="#262626" />
        <rect x="2"  y="21" width="8"  height="5" fill="#262626" />
      </svg>
    );
  }
  if (type === "vector-gt") {
    return (
      <svg width="24" height="24" viewBox="0 0 28 28" className={wrap}>
        <rect width="28" height="28" fill="#0e0e0e" />
        {[
          [3, 3, 7, 5], [12, 3, 5, 5], [19, 3, 6, 6],
          [3, 11, 9, 6], [14, 13, 8, 5],
          [3, 20, 6, 5], [11, 20, 10, 5],
        ].map(([x, y, w, h], i) => (
          <rect key={i} x={x} y={y} width={w} height={h}
                fill="#00B89422" stroke="#00B894" strokeWidth="1" />
        ))}
      </svg>
    );
  }
  return (
    <svg width="24" height="24" viewBox="0 0 28 28" className={wrap}>
      <rect width="28" height="28" fill="#0e0e0e" />
      {[
        [3, 3, 7, 5], [12, 3, 5, 5], [19, 3, 6, 6],
        [3, 11, 9, 6], [14, 13, 8, 5],
        [3, 20, 6, 5], [11, 20, 10, 5],
      ].map(([x, y, w, h], i) => (
        <rect key={i} x={x} y={y} width={w} height={h}
              fill="none" stroke={color} strokeWidth="1" />
      ))}
    </svg>
  );
}
