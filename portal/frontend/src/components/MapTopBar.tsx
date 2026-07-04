import { useState } from "react";
import { Download, Play, SplitSquareHorizontal } from "lucide-react";
import { api, DatasetDetail } from "../api/client";
import type { StackState } from "../types";

interface Props {
  dataset: DatasetDetail | null;
  /** Layer-stack master toggles — used to decide whether the swipe
   *  button is enabled (slider needs at least one overlay master on
   *  AND that overlay actually present in the dataset). */
  stack: StackState;
  swipeOn: boolean;
  onToggleSwipe: () => void;
  onRunInference?: () => void;
}

/**
 * 48-px header bar above the map. Holds the dataset title + LIB badge,
 * the swipe-compare toggle, and the primary Run / Download actions.
 * The actions auto-hide when no dataset is selected.
 */
export function MapTopBar({
  dataset, stack, swipeOn, onToggleSwipe, onRunInference,
}: Props) {
  // Track whether the user has confirmed the download. Until they
  // do, we just show a confirmation popup — we don't kick off the
  // browser's save-as dialog without a deliberate click.
  const [confirmDownload, setConfirmDownload] = useState(false);

  const title = dataset?.base_name ?? dataset?.name ?? "No dataset selected";
  const subtitle = dataset?.layers?.length
    ? `${dataset.layers.length} layer${dataset.layers.length === 1 ? "" : "s"} · ${dataset.crs ?? "EPSG:4326"}`
    : dataset
      ? "no outputs"
      : "open a tab on the left to pick a file";

  // The swipe slider needs at least one overlay that's BOTH present
  // on the dataset AND enabled in the right-side Layers panel — if
  // nothing is toggled on the slider can't show anything meaningful
  // on its right side.
  const hasPreds       = !!dataset?.has_gpkg;
  const hasGt          = (dataset?.gt_layers?.length ?? 0) > 0;
  const predsOnInPanel = stack.predictions && hasPreds;
  const gtOnInPanel    = stack.groundTruth && hasGt;
  const swipeAvailable = predsOnInPanel || gtOnInPanel;

  /** Triggers the browser Save-As dialog for the dataset's GPKG. */
  const triggerDownload = () => {
    if (!dataset?.has_gpkg) return;
    const a = document.createElement("a");
    a.href = api.gpkgDownloadUrl(dataset.name);
    a.download = `${dataset.name}_pred.gpkg`;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  return (
    <div className="h-12 bg-ink-800 border-b border-ink-700 flex items-center px-4 gap-3 flex-shrink-0">
      <div className="leading-tight min-w-0 flex items-center gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span
              title={dataset?.name}
              className="text-sm font-semibold text-ink-300 truncate font-mono"
            >
              {title}
            </span>
            {dataset?.checkpoint && (
              <span
                title={`Model checkpoint: ${dataset.checkpoint}`}
                className="text-[9px] text-accent-500 border border-accent-500/40 px-1 tracking-wider font-medium flex-shrink-0"
              >
                {dataset.checkpoint}
              </span>
            )}
            {dataset?.origin === "offline" && (
              <span
                title="Read-only library result from the bundled CLI run"
                className="text-[9px] text-ink-400 border border-ink-600 px-1 tracking-wider font-medium"
              >
                LIB
              </span>
            )}
          </div>
          <div className="text-[10px] text-ink-400 font-mono truncate">{subtitle}</div>
        </div>
      </div>

      <div className="flex-1" />

      {dataset && (
        <button
          onClick={swipeAvailable ? onToggleSwipe : undefined}
          disabled={!swipeAvailable}
          title={swipeAvailable
            ? (swipeOn ? "Hide swipe slider"
                       : `Compare imagery vs ${predsOnInPanel ? "predictions" : "ground truth"}`)
            : "Enable predictions or ground truth in the Layers panel first"}
          className={`h-7 w-7 grid place-items-center
                      ${swipeOn && swipeAvailable
                        ? "bg-accent-500 text-white"
                        : "text-ink-400 hover:text-ink-300 border border-ink-700 hover:bg-ink-700/40"}
                      ${!swipeAvailable ? "opacity-40 cursor-not-allowed" : ""}`}
        >
          <SplitSquareHorizontal className="w-3.5 h-3.5" />
        </button>
      )}

      {onRunInference && (
        <button
          onClick={onRunInference}
          className="h-7 px-3 inline-flex items-center gap-1.5
                      bg-accent-500 text-ink-900 font-semibold text-[11px] hover:bg-accent-600 transition-colors"
        >
          <Play className="w-3 h-3" /> Run inference
        </button>
      )}

      <button
        onClick={() => dataset?.has_gpkg && setConfirmDownload(true)}
        disabled={!dataset?.has_gpkg}
        title={dataset?.has_gpkg
          ? "Download GeoPackage (.gpkg) to your device"
          : "No GeoPackage available for this dataset"}
        className="h-7 px-3 inline-flex items-center gap-1.5
                    border border-ink-700 text-[11px] text-ink-300 hover:bg-ink-700/40
                    disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <Download className="w-3 h-3" /> Download features
      </button>

      {/* Download confirmation dialog — same visual language as the
          Start / Existing dialogs over the map area.  Anchored to the
          viewport so the modal centers properly even though MapTopBar
          itself is just a horizontal strip. */}
      {confirmDownload && dataset && (
        <div className="fixed inset-0 grid place-items-center bg-ink-950/70 backdrop-blur-sm z-50">
          <div className="w-[420px] bg-ink-800 border border-accent-500/40 p-4">
            <div className="flex items-start gap-3 mb-3">
              <div className="w-8 h-8 bg-accent-500/15 grid place-items-center flex-shrink-0">
                <Download className="w-4 h-4 text-accent-500" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-semibold text-ink-300 mb-1">
                  Download features?
                </div>
                <p className="text-[11px] text-ink-400 leading-relaxed">
                  Save the GeoPackage (
                  <span className="font-mono text-ink-300">{dataset.name}_pred.gpkg</span>
                  ) to your device. The browser will ask where to put it.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              <button
                onClick={() => { triggerDownload(); setConfirmDownload(false); }}
                className="inline-flex items-center gap-1.5 px-3 h-8 text-xs font-semibold
                           bg-accent-500 text-ink-900 hover:bg-accent-600 transition-colors"
              >
                <Download className="w-3 h-3" /> Download
              </button>
              <button
                onClick={() => setConfirmDownload(false)}
                className="px-3 h-8 text-[11px] text-ink-400 hover:text-ink-300
                           border border-ink-700 hover:bg-ink-700/40"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
