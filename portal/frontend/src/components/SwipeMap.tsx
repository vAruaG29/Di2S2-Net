import { useCallback, useEffect, useRef, useState } from "react";
import type { Map as MLMap } from "maplibre-gl";

import { MapView, type CameraInfo, type LoadStatus } from "./MapView";
import type { LayerState } from "../types";
import { BASE_STYLES } from "../constants";
import { DatasetDetail } from "../api/client";
import type { StackState } from "../types";

/** Which overlay the user wants on the right side of the swipe slider.
 *  Left side is always imagery-only by design. */
export type SwipeRightLayer = "predictions" | "groundTruth";

interface Props {
  dataset: DatasetDetail | null;
  layerState: Record<string, LayerState>;
  /** Per-class GT visibility/opacity, keyed by class name. */
  gtLayerState: Record<string, LayerState>;
  baseStyle: keyof typeof BASE_STYLES;
  stack: StackState;
  swipeEnabled: boolean;
  /** Which overlay sits on the right side when swipe is on. Ignored
   *  when swipe is off (both predictions + GT then follow the
   *  layer-stack master toggles). */
  swipeRight: SwipeRightLayer;
  onSwipeRightChange: (v: SwipeRightLayer) => void;
  onCameraChange?: (c: CameraInfo) => void;
  onLoadingChange?: (s: LoadStatus) => void;
  onVectorLoadingChange?: (inFlight: number, total: number) => void;
  onPrimaryMap?: (m: MLMap | null) => void;
}

/**
 * Swipe-compare view that *looks* like a single MapLibre instance.
 *
 * Under the hood we mount two maps:
 *
 *   - **Primary (full layers)** — the only interactive map. Owns the
 *     camera; receives every wheel, drag, click, etc.
 *   - **Mirror (imagery only)** — `interactive:false` + a wrapper with
 *     `pointer-events:none`, so it can never receive an event. It just
 *     mirrors the primary's camera on every `move` (which MapLibre
 *     fires on every interpolation frame), so the two views are always
 *     pixel-aligned. It's clipped to the LEFT of the swipe handle, so
 *     pre-handle you see imagery only; post-handle you see imagery +
 *     predictions + (optional) ground truth.
 *
 * When `swipeEnabled === false` only the primary is rendered, so the
 * common case stays single-instance and there's no tile pipeline
 * duplication.
 */
export function SwipeMap({
  dataset, layerState, gtLayerState, baseStyle, stack, swipeEnabled,
  swipeRight, onSwipeRightChange,
  onCameraChange, onLoadingChange, onVectorLoadingChange, onPrimaryMap,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  // We hold the maps in *state* (not refs) so the sync effect actually
  // re-runs when MapLibre finishes mounting and hands us the instance.
  const [primaryMap, setPrimaryMap] = useState<MLMap | null>(null);
  const [mirrorMap,  setMirrorMap]  = useState<MLMap | null>(null);

  const [split, setSplit] = useState(0.5);
  const draggingRef = useRef(false);

  // Slider can only do something useful when an overlay is BOTH
  // available for the current dataset AND enabled in the right-side
  // Layers panel. Otherwise the right side would show the same
  // thing as the left (imagery only) and the user would see no
  // visible difference. `swipeEnabled` stays a global preference;
  // we just suppress the rendered slider when there's nothing to
  // compare against.
  const hasPreds      = !!dataset?.has_gpkg;
  const hasGt         = (dataset?.gt_layers?.length ?? 0) > 0;
  const predsOnInPanel = stack.predictions && hasPreds;
  const gtOnInPanel    = stack.groundTruth && hasGt;
  const swipeActive    = swipeEnabled && !!dataset && (predsOnInPanel || gtOnInPanel);

  // The Predictions / Ground Truth chooser pills only make sense if
  // BOTH masters are on — when only one is, the slider implicitly
  // uses that one (no chooser).
  const showChooser    = predsOnInPanel && gtOnInPanel;
  const effectiveRight: SwipeRightLayer =
    showChooser ? swipeRight
    : predsOnInPanel ? "predictions"
    : "groundTruth";

  // ── Handle-drag (window-level so the cursor doesn't escape) ─────────
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!draggingRef.current || !containerRef.current) return;
      const r = containerRef.current.getBoundingClientRect();
      const v = (e.clientX - r.left) / r.width;
      setSplit(Math.max(0.04, Math.min(0.96, v)));
    };
    const onUp = () => {
      draggingRef.current = false;
      document.body.style.cursor = "";
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  // ── One-direction camera sync: primary → mirror.
  //    Because the mirror is non-interactive AND covered by a
  //    pointer-events:none wrapper, it never produces its own move
  //    events, so a uni-directional handler is enough and can't race.
  const syncMirror = useCallback(() => {
    if (!primaryMap || !mirrorMap) return;
    mirrorMap.jumpTo({
      center:  primaryMap.getCenter(),
      zoom:    primaryMap.getZoom(),
      bearing: primaryMap.getBearing(),
      pitch:   primaryMap.getPitch(),
    });
  }, [primaryMap, mirrorMap]);

  // When swipe is turned off, drop the mirror reference so we never
  // call methods on a MapLibre instance that's already been .remove()'d
  // by its component's unmount cleanup.
  useEffect(() => {
    if (!swipeActive) setMirrorMap(null);
  }, [swipeActive]);

  useEffect(() => {
    if (!swipeActive || !primaryMap || !mirrorMap) return;

    // 1) initial align
    syncMirror();

    // 2) follow every camera change. MapLibre fires `move` on every
    //    pan / zoom / wheel frame, so this single listener covers
    //    every interaction.
    primaryMap.on("move", syncMirror);

    // 3) re-align after a resize, just in case
    const ro = new ResizeObserver(() => syncMirror());
    if (containerRef.current) ro.observe(containerRef.current);

    return () => {
      primaryMap.off("move", syncMirror);
      ro.disconnect();
    };
  }, [swipeActive, primaryMap, mirrorMap, syncMirror]);

  // Effective overlay flags per side.
  //
  // When the swipe slider is ACTIVE (master on AND at least one
  // overlay enabled in the right panel), the primary renders
  // imagery + whichever overlay corresponds to `effectiveRight`.
  // When the slider is OFF the master toggles win — predictions
  // and GT each stack on the primary per the layer-stack panel.
  const primaryShowsPreds = swipeActive
    ? effectiveRight === "predictions"
    : stack.predictions;
  const primaryShowsGT = swipeActive
    ? effectiveRight === "groundTruth"
    : stack.groundTruth;

  return (
    <div ref={containerRef} className="absolute inset-0 overflow-hidden bg-ink-950">
      {/* Swipe semantics:
          ─ off:  primary stacks everything (basemap + image +
                  predictions + optional GT) per the layer-stack
                  master toggles.
          ─ on:   LEFT side (mirror) = imagery-only;
                  RIGHT side (primary) = imagery + ONE overlay
                  (predictions or GT, chosen via `swipeRight`).
          See the const block above the return for the effective flags. */}
      {/* PRIMARY — interactive, owns the camera. */}
      <div className="absolute inset-0">
        <MapView
          dataset={dataset}
          layerState={layerState}
          gtLayerState={gtLayerState}
          baseStyle={baseStyle}
          showBasemap={stack.basemap}
          showImagery={stack.imagery}
          showPredictions={primaryShowsPreds}
          showGroundTruth={primaryShowsGT}
          onMapReady={(m) => { setPrimaryMap(m); onPrimaryMap?.(m); }}
          onCameraChange={onCameraChange}
          onLoadingChange={onLoadingChange}
          onVectorLoadingChange={onVectorLoadingChange}
        />
      </div>

      {/* MIRROR — imagery-only on the LEFT side of the handle.
          pointer-events:none so events fall through to the primary. */}
      {swipeActive && (
        <div
          className="absolute inset-0 pointer-events-none"
          style={{ clipPath: `inset(0 ${(1 - split) * 100}% 0 0)` }}
        >
          <MapView
            dataset={dataset}
            layerState={layerState}
            gtLayerState={gtLayerState}
            baseStyle={baseStyle}
            showBasemap={stack.basemap}
            showImagery={stack.imagery}
            showPredictions={false}
            showGroundTruth={false}
            noControls
            interactive={false}
            onMapReady={setMirrorMap}
          />
        </div>
      )}

      {/* Swipe handle (interactive — its container has pointer-events
          enabled by default). Sits above both maps. */}
      {swipeActive && (
        <>
          <div
            className="absolute top-0 bottom-0 w-0.5 -translate-x-px bg-accent-500 pointer-events-none"
            style={{ left: `${split * 100}%` }}
          />
          <button
            onMouseDown={(e) => {
              e.preventDefault();
              draggingRef.current = true;
              document.body.style.cursor = "ew-resize";
            }}
            className="absolute top-1/2 w-9 h-9 -translate-x-1/2 -translate-y-1/2
                       bg-accent-500 text-white grid place-items-center cursor-ew-resize
                       hover:bg-accent-600 transition-colors z-10"
            style={{ left: `${split * 100}%` }}
            aria-label="Swipe between imagery and predictions"
          >
            <svg width="18" height="18" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M6 4 L3 8 L6 12" />
              <path d="M10 4 L13 8 L10 12" />
            </svg>
          </button>

          {/* LEFT label — always "imagery" while swipe is on. */}
          <div
            className="absolute top-3 px-2 py-1 bg-ink-800/90 border border-ink-700
                       text-[10px] font-mono tracking-wider text-ink-400 pointer-events-none
                       -translate-x-full"
            style={{ left: `calc(${split * 100}% - 12px)` }}
          >
            IMAGERY
          </div>

          {/* RIGHT-side label.
              - When both Predictions and Ground Truth are enabled in
                the master panel, render the chooser pills so the user
                picks which one sits on the right.
              - When only one is on, just show its name as a static
                label — no pointless chooser. */}
          {showChooser ? (
            <div
              className="absolute top-3 flex items-stretch border bg-ink-800/95 backdrop-blur
                         text-[10px] font-mono tracking-wider pointer-events-auto"
              style={{
                left: `calc(${split * 100}% + 12px)`,
                borderColor: effectiveRight === "groundTruth" ? "#00B894" : "#ff5600",
              }}
              title="What sits on the right side of the slider"
            >
              <button
                onClick={() => onSwipeRightChange("predictions")}
                className={`px-2 py-1 ${
                  effectiveRight === "predictions"
                    ? "bg-accent-500/15 text-accent-500"
                    : "text-ink-400 hover:text-ink-300"
                }`}
                title="Show predictions on the right"
              >
                PREDICTIONS
              </button>
              <button
                onClick={() => onSwipeRightChange("groundTruth")}
                className={`px-2 py-1 border-l border-ink-700 ${
                  effectiveRight === "groundTruth"
                    ? "bg-emerald-500/15 text-gt"
                    : "text-ink-400 hover:text-ink-300"
                }`}
                title="Show ground truth on the right"
              >
                GROUND TRUTH
              </button>
            </div>
          ) : (
            <div
              className={`absolute top-3 px-2 py-1 bg-ink-800/90 border
                          text-[10px] font-mono tracking-wider pointer-events-none
                          ${effectiveRight === "groundTruth"
                            ? "border-emerald-500/60 text-gt"
                            : "border-accent-500 text-accent-500"}`}
              style={{ left: `calc(${split * 100}% + 12px)` }}
            >
              {effectiveRight === "groundTruth" ? "GROUND TRUTH" : "PREDICTIONS"}
            </div>
          )}

          {/* Percent readout */}
          <div
            className="absolute top-[calc(50%+28px)] px-2 py-0.5 bg-accent-500 text-white
                       text-[10px] font-mono font-semibold pointer-events-none -translate-x-1/2 whitespace-nowrap"
            style={{ left: `${split * 100}%` }}
          >
            {Math.round(split * 100)}%  ←  →  {Math.round((1 - split) * 100)}%
          </div>
        </>
      )}
    </div>
  );
}
