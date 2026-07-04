import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { Map as MLMap } from "maplibre-gl";
import { Layers } from "lucide-react";

import { api, DatasetDetail } from "./api/client";
import { BASE_STYLES } from "./constants";
import { useJobStream } from "./api/sse";

import { LeftRail } from "./components/LeftRail";
import { FilesPanel } from "./components/FilesPanel";
import { ClassesFloat } from "./components/ClassesFloat";
import { GtClassesFloat } from "./components/GtClassesFloat";
import { LayerStackPanel } from "./components/LayerStackPanel";
import { MapTopBar } from "./components/MapTopBar";
import { SwipeMap, type SwipeRightLayer } from "./components/SwipeMap";
import { FooterBar } from "./components/FooterBar";
import type { CameraInfo, LoadStatus } from "./components/MapView";
import type { LayerState, LeftTab, StackState } from "./types";
import {
  ZoomControls, BaseStyleSwitcher, IoUPanel, LoadingBar,
} from "./components/MapOverlays";
import { JobOverlay, StartInferenceDialog } from "./components/JobOverlay";

// Default stack on a fresh dataset click: only basemap + image visible.
// The user reaches for the right-side Layers panel to enable
// predictions / ground truth. We also re-apply this on every dataset
// switch (see the effect below) so the user gets the same starting
// state no matter which image they opened previously.
const DEFAULT_STACK: StackState = {
  basemap: true,
  imagery: true,
  predictions: false,
  groundTruth: false,
};

export function App() {
  // ── Selection state ──────────────────────────────────────────────────
  const [tab, setTab] = useState<LeftTab | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  // FilesPanel is **always** mounted: either as a thin left-edge tab
  // (minimised) or expanded next to the LeftFloat. Defaulting to
  // minimised means a fresh portal load shows the tab on the left
  // edge, prompting the user to expand it and pick Train/Test/Upload.
  const [filesMinimized, setFilesMinimized] = useState(true);

  /** Tab toggler used by LeftFloat. Clicking any tab always expands
   *  the files panel, even if it was previously minimised. */
  const handleTabChange = (t: LeftTab | null) => {
    setTab(t);
    if (t) setFilesMinimized(false);
  };

  // ── Layer + map state ────────────────────────────────────────────────
  const [stack, setStack] = useState<StackState>(DEFAULT_STACK);
  const [layerState, setLayerState] = useState<Record<string, LayerState>>({});
  // Parallel per-class state for the ground-truth overlay.  Same shape
  // as `layerState`, but keyed by GT class names (which come from the
  // dataset's `gt_layers`). All classes default to visible so the GT
  // master toggle in LeftFloat does what the user expects on first
  // enable.
  const [gtLayerState, setGtLayerState] = useState<Record<string, LayerState>>({});
  const [baseStyle, setBaseStyle] = useState<keyof typeof BASE_STYLES>("dark");
  // Off at first launch — the user explicitly clicks the slider
  // button in the top bar to activate compare mode. Toggling any
  // master layer on its own does NOT activate it; the slider just
  // *uses* whatever masters are on once the user opts in.
  const [swipeEnabled, setSwipeEnabled] = useState(false);
  // What sits on the right side of the swipe slider — only meaningful
  // while `swipeEnabled` is true. Defaults to predictions; user picks
  // "groundTruth" if they want a GT-vs-imagery diff instead.
  const [swipeRight, setSwipeRight] = useState<SwipeRightLayer>("predictions");
  const [cam, setCam] = useState<CameraInfo | null>(null);
  const [loadStatus, setLoadStatus] = useState<LoadStatus>({
    imageryLoading: false, tilesInFlight: 0, layersInFlight: 0, layersTotal: 0,
  });
  // Prediction-vector fetch progress reported by MapView/VectorLayers.
  // Merged into `loadStatus` for the LoadingBar so the user can see
  // "loading aerial COG + N/M layers" rather than guessing whether
  // predictions are coming.
  const [vectorLoading, setVectorLoading] = useState<{ inFlight: number; total: number }>({
    inFlight: 0, total: 0,
  });
  const mergedLoadStatus: LoadStatus = {
    imageryLoading: loadStatus.imageryLoading,
    tilesInFlight: loadStatus.tilesInFlight,
    layersInFlight: vectorLoading.inFlight,
    layersTotal: vectorLoading.total,
  };
  const mapRef = useRef<MLMap | null>(null);

  // ── Inference state ──────────────────────────────────────────────────
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobDataset, setJobDataset] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  // BASE dataset name the Run dialog is open for (null = closed). The
  // dialog handles checkpoint selection + the "outputs already exist"
  // branch internally.
  const [confirmingStart, setConfirmingStart] = useState<string | null>(null);

  const { events, status } = useJobStream(jobId);
  const queryClient = useQueryClient();

  // ── Dataset summaries (tab counts) ──────────────────────────────────
  const dsList = useQuery({
    queryKey: ["datasets"],
    queryFn: api.datasets,
    refetchInterval: 10_000,
  });
  const trainCount = (dsList.data ?? []).filter((d) => d.has_metrics).length;
  const testCount  = (dsList.data ?? []).filter((d) => !d.has_metrics).length;

  // ── Selected dataset detail ─────────────────────────────────────────
  const dsQuery = useQuery<DatasetDetail | null>({
    queryKey: ["dataset", selected],
    queryFn: () => (selected ? api.dataset(selected) : Promise.resolve(null)),
    enabled: !!selected,
  });
  const dataset = dsQuery.data ?? null;
  // Ground truth is real-only when there are per-tile masks on disk for
  // this dataset; backend reports that as `gt_layers` (a list of class
  // names that actually have mask pixels). Empty → no GT toggle.
  const hasGroundTruth = (dataset?.gt_layers?.length ?? 0) > 0;

  // When the user switches to a different dataset, reset the
  // overlay master toggles (predictions / ground truth) to off so
  // they have to deliberately enable what they want. Basemap +
  // image follow whatever the user had on for the previous
  // dataset, so leave those alone. `swipeEnabled` is also untouched
  // — that's a global preference per the brief.
  useEffect(() => {
    if (!dataset?.name) return;
    setStack((s) => ({ ...s, predictions: false, groundTruth: false }));
  }, [dataset?.name]);

  // Auto-correct the swipe right-side chooser when the current
  // dataset doesn't offer the previously-selected overlay.  E.g.
  // user had "predictions" selected on a Test dataset, switches to
  // a Train image that only has GT (no inference yet) — flip
  // `swipeRight` to "groundTruth" so the slider actually shows
  // something on the right. Inverse case mirrors.
  useEffect(() => {
    if (!dataset) return;
    const hasPreds = !!dataset.has_gpkg;
    const hasGtForDs = (dataset.gt_layers?.length ?? 0) > 0;
    if (swipeRight === "predictions" && !hasPreds && hasGtForDs) {
      setSwipeRight("groundTruth");
    } else if (swipeRight === "groundTruth" && !hasGtForDs && hasPreds) {
      setSwipeRight("predictions");
    }
  }, [dataset?.name, dataset?.has_gpkg, dataset?.gt_layers?.length]);   // eslint-disable-line

  // Seed per-class layer toggles when the dataset changes OR when
  // new layers appear for the same dataset (which happens when an
  // upload completes inference and `dataset.layers` grows from [] to
  // the prediction set — without re-seeding, ClassesFloat eyes would
  // show closed even though `VectorLayers`' fallback rendered the
  // preferred 6 on the map).
  //
  // Implementation: track the dataset we last fully-seeded for; if
  // it's a new dataset, do a full reset; if the same dataset has new
  // layer names, ADD them (preserving any user toggles on existing
  // entries).
  const lastSeededRef = useRef<string | null>(null);
  useEffect(() => {
    if (!dataset) {
      setLayerState({});
      lastSeededRef.current = null;
      return;
    }
    const preferred = new Set([
      "Built_Up_Area_type", "Road", "Water_Body",
      "Bridge", "Railway", "Utility_Poly",
    ]);
    const isNewDataset = lastSeededRef.current !== dataset.name;

    if (isNewDataset) {
      const seed: Record<string, LayerState> = {};
      for (const l of dataset.layers) {
        seed[l.name] = { visible: preferred.has(l.name), opacity: 0.7 };
      }
      setLayerState(seed);
      lastSeededRef.current = dataset.name;
    } else {
      setLayerState((prev) => {
        const next = { ...prev };
        let changed = false;
        for (const l of dataset.layers) {
          if (!(l.name in next)) {
            next[l.name] = { visible: preferred.has(l.name), opacity: 0.7 };
            changed = true;
          }
        }
        return changed ? next : prev;
      });
    }
  }, [dataset?.name, dataset?.layers?.length]);              // eslint-disable-line

  // Same dance for the GT per-class state — re-seed when gt_layers
  // grows for the same dataset (so the GtClassesFloat row eyes match
  // what's rendered by GroundTruthLayers).
  const lastGtSeededRef = useRef<string | null>(null);
  useEffect(() => {
    const gts = dataset?.gt_layers ?? [];
    if (!dataset || gts.length === 0) {
      setGtLayerState({});
      lastGtSeededRef.current = null;
      return;
    }
    const isNewDataset = lastGtSeededRef.current !== dataset.name;
    if (isNewDataset) {
      const seed: Record<string, LayerState> = {};
      for (const name of gts) {
        seed[name] = { visible: true, opacity: 0.7 };
      }
      setGtLayerState(seed);
      lastGtSeededRef.current = dataset.name;
    } else {
      setGtLayerState((prev) => {
        const next = { ...prev };
        let changed = false;
        for (const name of gts) {
          if (!(name in next)) {
            next[name] = { visible: true, opacity: 0.7 };
            changed = true;
          }
        }
        return changed ? next : prev;
      });
    }
  }, [dataset?.name, dataset?.gt_layers?.length]);           // eslint-disable-line

  useEffect(() => {
    if (status === "done" && jobDataset) {
      // The dataset existed BEFORE inference completed (we already
      // surface source-only uploads on the map), so react-query has a
      // cached `DatasetDetail` with `layers=[]` and `has_gpkg=false`.
      // Invalidate both the per-dataset cache and the gallery list so
      // the new prediction layers + GPKG flag show up the moment the
      // user clicks Inspect.
      queryClient.invalidateQueries({ queryKey: ["dataset", jobDataset] });
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      queryClient.invalidateQueries({ queryKey: ["uploads", "list"] });
      setSelected(jobDataset);
    }
  }, [status, jobDataset, queryClient]);

  // ── Inference flow ──────────────────────────────────────────────────
  // Clicking ▶ just opens the Run dialog for the BASE dataset name; the
  // dialog picks the checkpoint and decides (via preflight) whether to
  // offer "Start", "Re-run" or "Use existing".
  const askInference = (name: string) => setConfirmingStart(name);

  // Launch a run for (base dataset, checkpoint). The server returns the
  // *effective* dataset name (`<base>@@<id>` for a non-default
  // checkpoint) — we track that so the overlay + auto-load target the
  // right per-checkpoint result.
  const runInference = async (base: string, checkpointId: string, force: boolean) => {
    setStarting(true);
    try {
      const source = tab === "upload" ? "upload" : "existing";
      const resp = await api.startInference(source, base, force, checkpointId);
      setConfirmingStart(null);
      if (resp.status === "exists") {
        // Server says it's already computed (race with the dialog's own
        // preflight) — just load it.
        setSelected(resp.dataset_name);
        return;
      }
      setJobId(resp.job_id);
      setJobDataset(resp.dataset_name);
    } catch (e) {
      console.error("start inference", e);
    } finally {
      setStarting(false);
    }
  };

  const useExistingResults = (effectiveName: string) => {
    setSelected(effectiveName);
    setConfirmingStart(null);
  };
  const dismissJob = () => { setJobId(null); setJobDataset(null); };

  // ── Layout: fixed left column + top bar + main map + full-width footer
  return (
    <div className="h-screen w-screen flex flex-col bg-ink-900 text-ink-300 overflow-hidden font-sans">
      <div className="flex flex-1 min-h-0">
        {/* Thin vertical Train/Test/Upload rail */}
        <LeftRail
          tab={tab}
          onTabChange={handleTabChange}
          trainCount={trainCount}
          testCount={testCount}
        />

        {/* Right column = top header + map */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0">
          <MapTopBar
            dataset={dataset}
            stack={stack}
            swipeOn={swipeEnabled}
            onToggleSwipe={() => setSwipeEnabled(!swipeEnabled)}
            onRunInference={selected ? () => askInference(selected) : undefined}
          />

          <div className="flex-1 relative min-h-0">
            <SwipeMap
              dataset={dataset}
              layerState={layerState}
              gtLayerState={gtLayerState}
              baseStyle={baseStyle}
              stack={stack}
              swipeEnabled={swipeEnabled}
              swipeRight={swipeRight}
              onSwipeRightChange={setSwipeRight}
              onCameraChange={setCam}
              onLoadingChange={setLoadStatus}
              onVectorLoadingChange={(inFlight, total) =>
                setVectorLoading({ inFlight, total })
              }
              onPrimaryMap={(m) => { mapRef.current = m; }}
            />

            {/* FilesPanel — always mounted. Either expanded next to the
                LeftFloat, or collapsed into a thin tab on the left
                edge of the map area. */}
            <FilesPanel
              tab={tab}
              selected={selected}
              onSelect={setSelected}
              onRunInference={askInference}
              onIngested={setSelected}
              minimized={filesMinimized}
              onMinimize={() => setFilesMinimized(true)}
              onExpand={() => setFilesMinimized(false)}
            />

            {/* ClassesFloat — floats top-right when a dataset is selected */}
            {/* Right-side panels stack in a flex column so the GT
                panel always sits directly below the predictions
                panel (no gap, no overlap, no matter how tall either
                gets). Each panel handles its own minimised tab via
                its internal state — those tabs use abs positioning
                and break out of this flex flow. */}
            {dataset && (
              <div
                className="absolute top-3 right-3 bottom-12 w-[240px] z-20
                           flex flex-col gap-2 pointer-events-none"
              >
                <ClassesFloat
                  dataset={dataset}
                  layerState={layerState}
                  onLayerStateChange={setLayerState}
                  predictionsOn={stack.predictions}
                />
                {/* 4-row layer master toggles, lifted out of the old
                    LeftFloat — sits right under the classes panel, so
                    "below the classes layers" per the brief. */}
                <LayerStackPanel
                  stack={stack}
                  onStackChange={setStack}
                  imageryLabel={dataset.source_cog?.split("/").pop() ?? null}
                  hasGroundTruth={hasGroundTruth}
                  predLayerCount={dataset.layers?.length ?? 0}
                />
                {hasGroundTruth && stack.groundTruth && (
                  <GtClassesFloat
                    gtLayers={dataset.gt_layers ?? []}
                    state={gtLayerState}
                    onChange={setGtLayerState}
                    masterOn={stack.groundTruth}
                  />
                )}
              </div>
            )}

            {/* Map chrome — all positioned inside the map area, so they
                naturally sit above the footer's top edge. */}
            <BaseStyleSwitcher style={baseStyle} setStyle={setBaseStyle} />
            <ZoomControls
              onZoomIn={() => mapRef.current?.zoomIn({ duration: 250 })}
              onZoomOut={() => mapRef.current?.zoomOut({ duration: 250 })}
            />
            {dataset && hasGroundTruth && stack.groundTruth && stack.predictions && (
              <IoUPanel metrics={dataset.metrics} />
            )}
            <LoadingBar status={mergedLoadStatus} />

            {!selected && <EmptyMapHint />}

            {/* Modal-ish dialogs */}
            {confirmingStart && (
              <StartInferenceDialog
                datasetName={confirmingStart}
                onStart={(ckptId, force) => runInference(confirmingStart, ckptId, force)}
                onUseExisting={useExistingResults}
                onCancel={() => setConfirmingStart(null)}
                starting={starting}
              />
            )}
            {jobId && jobDataset && (
              <JobOverlay
                jobId={jobId}
                datasetName={jobDataset}
                events={events}
                status={status}
                onDone={() => {
                  // Force a refetch so the dataset detail reflects the
                  // freshly-written GPKG + metrics + gt_layers.
                  queryClient.invalidateQueries({ queryKey: ["dataset", jobDataset] });
                  queryClient.invalidateQueries({ queryKey: ["datasets"] });
                  setSelected(jobDataset);
                  dismissJob();
                }}
                onDismiss={dismissJob}
              />
            )}
          </div>
        </div>
      </div>

      {/* Full-width footer along the bottom (spans left column + map) */}
      <FooterBar
        crs={dataset?.crs}
        lat={cam?.lat}
        lng={cam?.lng}
        zoom={cam?.zoom}
      />
    </div>
  );
}

function EmptyMapHint() {
  // Centred empty-state card. Borrows the same surface as the other
  // floating panels (`bg-ink-800/95 border-ink-700 backdrop-blur`) and
  // adds a small accent-tinted icon block + eyebrow / title / body
  // hierarchy so it reads as part of the portal rather than a debug
  // overlay. Sits above the basemap (z-30) but below any modal (z-50).
  // `pointer-events-none` on the wrapper keeps the map pannable
  // through it.
  return (
    <div className="pointer-events-none absolute inset-0 grid place-items-center z-30">
      <div className="text-center w-[320px] px-6 py-5 bg-ink-800/40 border border-ink-700/60 backdrop-blur-lg shadow-lg">
        <div className="w-10 h-10 mx-auto mb-3 bg-accent-500/10 border border-accent-500/30 grid place-items-center">
          <Layers className="w-5 h-5 text-accent-500" />
        </div>
        <div className="text-[10px] uppercase tracking-wider text-ink-400 font-medium mb-1.5">
          Get started
        </div>
        <div className="text-sm font-semibold text-ink-300 mb-1.5">
          Pick a dataset to begin
        </div>
        <p className="text-[11px] text-ink-400 leading-relaxed">
          Open{" "}
          <span className="text-accent-500 font-medium">Train</span>,{" "}
          <span className="text-accent-500 font-medium">Test</span>, or{" "}
          <span className="text-accent-500 font-medium">Upload</span>{" "}
          from the left panel to choose an image.
        </p>
      </div>
    </div>
  );
}
