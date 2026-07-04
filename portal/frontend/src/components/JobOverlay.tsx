import { Ban, RotateCw, Trash2, X, ArrowRight, History, CheckCircle2, Play, Cpu, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ProgressTimeline } from "./ProgressTimeline";
import type { JobEvent } from "../api/sse";
import { api, PreflightInfo } from "../api/client";

interface Props {
  jobId: string;
  datasetName: string;
  events: JobEvent[];
  status: "idle" | "open" | "done" | "failed" | "error" | "cancelled";
  onDone: () => void;
  onDismiss: () => void;
}

/**
 * Floating panel docked to the right edge of the map area, showing
 * the progress timeline of an in-flight inference job. On
 * completion the body is replaced with a simple "all done" message
 * pointing the user at the right-side Layers panel — no separate
 * "Inspect" step (the dataset is already loaded; predictions just
 * need to be toggled on).
 *
 * z-50 keeps it above the floating right-side panels (LayerStack,
 * ClassesFloat, FilesPanel — all z-20) so the timeline never hides
 * behind them.
 */
export function JobOverlay({
  jobId, datasetName, events, status, onDone, onDismiss,
}: Props) {
  const [busy, setBusy] = useState<"cancel" | "cleanup" | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const cancel = async () => {
    setBusy("cancel");
    try {
      const r = await api.cancelJob(jobId);
      setToast(r.killed_proc ? "Termination signal sent." : `Already ${r.status}.`);
    } catch (e) { setToast(`Cancel failed: ${e}`); }
    finally { setBusy(null); }
  };

  const cleanup = async () => {
    if (!confirm("Delete all generated files for this run?")) return;
    setBusy("cleanup");
    try {
      const r = await api.cleanupJob(jobId, status === "open");
      setToast(r.removed ? "Files removed." : `Skipped: ${r.reason}`);
    } catch (e) { setToast(`Cleanup failed: ${e}`); }
    finally { setBusy(null); }
  };

  const running = status === "open";
  const ready = status === "done";
  const terminal = status === "done" || status === "failed"
                 || status === "cancelled" || status === "error";

  return (
    <div className="absolute top-3 right-14 w-[380px] max-h-[calc(100%-32px)]
                    bg-ink-800/95 border border-ink-700 flex flex-col overflow-hidden
                    backdrop-blur shadow-xl z-50">
      <div className="flex items-center px-3 py-2 border-b border-ink-700 flex-shrink-0">
        <span className={`w-1.5 h-1.5 rounded-full mr-2
                          ${status === "done" ? "bg-gt"
                          : status === "failed" || status === "error" ? "bg-red-400"
                          : status === "cancelled" ? "bg-amber-400"
                          : "bg-accent-500 animate-pulse"}`} />
        <div className="text-[11px] uppercase tracking-wider text-ink-400">
          {running ? "Running inference" : ready ? "Inference complete" : "Inference"}
        </div>
        <div className="flex-1" />
        <span className="text-[10px] font-mono text-ink-400 truncate">{datasetName}</span>
        <button
          onClick={onDismiss}
          className="ml-2 w-5 h-5 grid place-items-center text-ink-400 hover:text-ink-300"
        >
          <X className="w-3 h-3" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {ready ? (
          // Simple completion popup — no timeline, no "Inspect" step.
          // The user is already on this dataset (selection happened on
          // upload / when ▶ was clicked) and our cache-invalidate hook
          // has refreshed `DatasetDetail` with the new layers. All
          // that's left is for the user to toggle Predictions on in
          // the right-side Layers panel.
          <div className="px-2 py-6 text-center">
            <div className="w-12 h-12 mx-auto mb-3 bg-emerald-500/15 border border-emerald-500/40
                            grid place-items-center">
              <CheckCircle2 className="w-6 h-6 text-gt" />
            </div>
            <div className="text-sm font-semibold text-ink-300 mb-2">
              Inference complete
            </div>
            <p className="text-[11px] text-ink-400 leading-relaxed">
              You can now enable{" "}
              <span className="text-accent-500 font-medium">Predictions</span>
              {" "}from the Layers panel on the right side of the map.
            </p>
          </div>
        ) : (
          <ProgressTimeline events={events} status={status} />
        )}

        {toast && (
          <div className="mt-2 text-[11px] text-ink-300 bg-ink-900 border border-ink-700 p-2 font-mono">
            {toast}
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 p-3 border-t border-ink-700 flex-shrink-0 flex-wrap">
        {running && (
          <button
            onClick={cancel}
            disabled={busy === "cancel"}
            className="inline-flex items-center gap-1.5 px-2.5 h-7 text-[11px] font-medium
                       text-amber-400 border border-amber-500/30 hover:bg-amber-500/10 disabled:opacity-50"
          >
            <Ban className="w-3 h-3" />
            {busy === "cancel" ? "cancelling…" : "Cancel"}
          </button>
        )}
        {terminal && !ready && (
          <button
            onClick={cleanup}
            disabled={busy === "cleanup"}
            className="inline-flex items-center gap-1.5 px-2.5 h-7 text-[11px] font-medium
                       text-red-300 border border-red-500/30 hover:bg-red-500/10 disabled:opacity-50"
          >
            <Trash2 className="w-3 h-3" />
            {busy === "cleanup" ? "removing…" : "Clean up"}
          </button>
        )}
        <div className="flex-1" />
        {ready && (
          <button
            onClick={onDone}
            className="inline-flex items-center gap-1.5 px-4 h-8 text-xs font-semibold
                       bg-accent-500 text-ink-900 hover:bg-accent-600 transition-colors"
          >
            Got it
          </button>
        )}
        {terminal && !ready && (
          <button
            onClick={onDismiss}
            className="inline-flex items-center gap-1.5 px-2.5 h-7 text-[11px] font-medium
                       text-ink-400 border border-ink-700 hover:text-ink-300"
          >
            <RotateCw className="w-3 h-3" /> Close
          </button>
        )}
      </div>
    </div>
  );
}

interface ExistingProps {
  info: PreflightInfo;
  onUse: () => void;
  onRerun: () => void;
  onCancel: () => void;
  rerunning: boolean;
}

/** Modal-ish card centred over the map when outputs already exist. */
export function ExistingResultsDialog({ info, onUse, onRerun, onCancel, rerunning }: ExistingProps) {
  return (
    <div className="absolute inset-0 grid place-items-center bg-ink-950/70 backdrop-blur-sm z-50">
      <div className="w-[420px] bg-ink-800 border border-amber-500/30 p-4">
        <div className="flex items-start gap-3 mb-3">
          <div className="w-8 h-8 bg-amber-500/15 grid place-items-center flex-shrink-0">
            <History className="w-4 h-4 text-amber-400" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold text-ink-300 mb-1">
              Outputs already exist for{" "}
              <span className="font-mono text-amber-400">{info.dataset_name}</span>
            </div>
            <ul className="text-[11px] text-ink-400 space-y-0.5">
              <CheckLi ok={info.has_gpkg}        label="GeoPackage" />
              <CheckLi ok={info.has_metrics}     label="metrics CSV" />
              {info.has_stitched && (
                <CheckLi ok={true}                label="stitched raster" />
              )}
            </ul>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={onUse}
            className="inline-flex items-center gap-1.5 px-3 h-8 text-xs font-semibold
                       bg-accent-500 text-ink-900 hover:bg-accent-600 transition-colors"
          >
            <ArrowRight className="w-3 h-3" /> Use existing
          </button>
          <button
            onClick={onRerun}
            disabled={rerunning}
            className="inline-flex items-center gap-1.5 px-3 h-8 text-xs font-medium
                       text-amber-400 border border-amber-500/30 hover:bg-amber-500/10 disabled:opacity-50"
          >
            <RotateCw className="w-3 h-3" />
            {rerunning ? "Starting…" : "Re-run from scratch"}
          </button>
          <button
            onClick={onCancel}
            className="px-2 h-8 text-[11px] text-ink-400 hover:text-ink-300"
          >
            cancel
          </button>
        </div>
      </div>
    </div>
  );
}

interface StartProps {
  /** BASE dataset name (no checkpoint suffix). */
  datasetName: string;
  /** Start a run with the chosen checkpoint. `force` re-runs even when
   *  outputs for this (image, checkpoint) pair already exist. */
  onStart: (checkpointId: string, force: boolean) => void;
  /** Load the already-computed result for (image, checkpoint). */
  onUseExisting: (effectiveName: string) => void;
  onCancel: () => void;
  starting?: boolean;
}

/**
 * Modal-ish card centred over the map for launching inference. Shown
 * when the ▶ button is clicked on any dataset. Lets the user pick which
 * model checkpoint to run; results from non-default checkpoints are
 * stored under a separate `<name>@@<id>` dataset so they coexist with
 * the default checkpoint's predictions.
 *
 * The dialog preflights the chosen (image, checkpoint) pair live: if a
 * result already exists it offers "Use existing" / "Re-run", otherwise
 * a plain "Start inference".
 */
export function StartInferenceDialog({
  datasetName, onStart, onUseExisting, onCancel, starting,
}: StartProps) {
  const cksQ = useQuery({ queryKey: ["checkpoints"], queryFn: api.checkpoints });
  const checkpoints = cksQ.data?.checkpoints ?? [];
  const hidden = cksQ.data?.hidden ?? [];
  const expectedDecoder = cksQ.data?.expected_decoder ?? "hrdecoder";
  const defaultId = checkpoints.find((c) => c.is_default)?.id ?? checkpoints[0]?.id ?? "";

  const [ckpt, setCkpt] = useState<string>("");
  // Snap to the default checkpoint once the list arrives.
  useEffect(() => {
    if (!ckpt && defaultId) setCkpt(defaultId);
  }, [defaultId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Preflight the chosen (dataset, checkpoint) pair. Re-runs whenever the
  // checkpoint changes so the buttons reflect that exact combination.
  const pfQ = useQuery({
    queryKey: ["preflight", datasetName, ckpt],
    queryFn: () => api.preflight(datasetName, ckpt || undefined),
    enabled: !!ckpt,
  });
  const exists = pfQ.data?.outputs_exist ?? false;
  const effName = pfQ.data?.dataset_name ?? datasetName;
  const isDefault = ckpt === defaultId;

  return (
    <div className="absolute inset-0 grid place-items-center bg-ink-950/70 backdrop-blur-sm z-50">
      <div className="w-[440px] bg-ink-800 border border-accent-500/40 p-4">
        <div className="flex items-start gap-3 mb-3">
          <div className="w-8 h-8 bg-accent-500/15 grid place-items-center flex-shrink-0">
            <Play className="w-4 h-4 text-accent-500" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold text-ink-300 mb-1">
              Run inference on{" "}
              <span className="font-mono text-accent-500">{datasetName}</span>
            </div>
            <p className="text-[11px] text-ink-400 leading-relaxed">
              Pick a model checkpoint and generate predicted features. Each
              non-default checkpoint is saved as its own result so you can
              compare them on the map.
            </p>
          </div>
        </div>

        {/* Checkpoint dropdown */}
        <label className="block mb-3">
          <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-ink-400 font-medium mb-1">
            <Cpu className="w-3 h-3" /> Model checkpoint
          </span>
          <select
            value={ckpt}
            onChange={(e) => setCkpt(e.target.value)}
            disabled={cksQ.isLoading || checkpoints.length === 0}
            className="w-full h-8 bg-ink-900 border border-ink-700 text-[11px] text-ink-300
                       font-mono px-2 focus:outline-none focus:border-accent-500
                       disabled:opacity-50"
          >
            {cksQ.isLoading && <option>loading…</option>}
            {!cksQ.isLoading && checkpoints.length === 0 && (
              <option value="">no checkpoints found in pretrained/</option>
            )}
            {checkpoints.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}{c.is_default ? "  (default)" : ""}
              </option>
            ))}
          </select>
          {hidden.length > 0 && (
            <span className="block mt-1 text-[10px] text-ink-500">
              {hidden.length} checkpoint{hidden.length === 1 ? "" : "s"} hidden —
              incompatible decoder (need {expectedDecoder}):{" "}
              <span className="text-ink-400">
                {hidden.map((h) => `${h.filename}${h.decoder_type ? ` [${h.decoder_type}]` : ""}`).join(", ")}
              </span>
            </span>
          )}
        </label>

        {/* Live status for the chosen (image, checkpoint) pair */}
        <div className="text-[11px] mb-3 min-h-[16px]">
          {pfQ.isFetching ? (
            <span className="inline-flex items-center gap-1.5 text-ink-400">
              <Loader2 className="w-3 h-3 animate-spin" /> checking for existing result…
            </span>
          ) : exists ? (
            <span className="inline-flex items-center gap-1.5 text-gt">
              <CheckCircle2 className="w-3 h-3" />
              Result already exists for this model
              {!isDefault && (
                <span className="font-mono text-ink-400">({effName})</span>
              )}
            </span>
          ) : (
            <span className="text-ink-500">No result yet for this model.</span>
          )}
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {exists ? (
            <>
              <button
                onClick={() => onUseExisting(effName)}
                className="inline-flex items-center gap-1.5 px-3 h-8 text-xs font-semibold
                           bg-accent-500 text-ink-900 hover:bg-accent-600 transition-colors"
              >
                <ArrowRight className="w-3 h-3" /> Use existing
              </button>
              <button
                onClick={() => onStart(ckpt, true)}
                disabled={starting || !ckpt}
                className="inline-flex items-center gap-1.5 px-3 h-8 text-xs font-medium
                           text-amber-400 border border-amber-500/30 hover:bg-amber-500/10
                           disabled:opacity-50"
              >
                <RotateCw className="w-3 h-3" />
                {starting ? "Starting…" : "Re-run"}
              </button>
            </>
          ) : (
            <button
              onClick={() => onStart(ckpt, false)}
              disabled={starting || !ckpt}
              className="inline-flex items-center gap-1.5 px-3 h-8 text-xs font-semibold
                         bg-accent-500 text-ink-900 hover:bg-accent-600 transition-colors
                         disabled:opacity-50"
            >
              <Play className="w-3 h-3" />
              {starting ? "Starting…" : "Start inference"}
            </button>
          )}
          <button
            onClick={onCancel}
            className="px-3 h-8 text-[11px] text-ink-400 hover:text-ink-300
                       border border-ink-700 hover:bg-ink-700/40"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

function CheckLi({ ok, label }: { ok: boolean; label: string }) {
  return (
    <li className="flex items-center gap-1.5">
      <CheckCircle2 className={`w-3 h-3 ${ok ? "text-gt" : "text-ink-500"}`} />
      {label}: <span className={ok ? "text-ink-300" : "text-ink-500"}>{ok ? "present" : "missing"}</span>
    </li>
  );
}
