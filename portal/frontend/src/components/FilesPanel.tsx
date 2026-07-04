import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Crosshair, Search, X, Minus, ChevronRight,
  Play, FolderOpen, Upload as UploadIcon,
  Loader2, Folder, Eye, Trash2,
} from "lucide-react";

import {
  api, DatasetSummary, ServerFile, UploadResponse, UploadRow,
} from "../api/client";
import { UploadDropzone } from "./UploadDropzone";
import type { LeftTab } from "../types";

interface Props {
  tab: LeftTab | null;
  selected: string | null;
  onSelect: (name: string) => void;
  onRunInference: (name: string) => void;
  /** Called with the name once an upload/server-pick has produced a dataset. */
  onIngested?: (name: string) => void;
  /** Whether the panel is collapsed into the left-edge tab. */
  minimized: boolean;
  onMinimize: () => void;
  onExpand: () => void;
}

/**
 * Floating files panel — sits beside the LeftFloat. Can be minimised
 * into a thin tab pinned to the left edge of the map area; the tab
 * shows the active eyebrow ("Training Set" / "Test Set" / "Upload") or
 * a "Choose Train / Test / Upload" hint if the user collapsed all
 * tabs while the panel was minimised.
 */
export function FilesPanel({
  tab, selected, onSelect, onRunInference, onIngested,
  minimized, onMinimize, onExpand,
}: Props) {
  // ── Minimised: thin vertical tab pinned to the left edge ──────────
  if (minimized) {
    const label = tab === "train"  ? "Training Set"
                : tab === "test"   ? "Test Set"
                : tab === "upload" ? "Upload"
                :                    "Choose Train / Test / Upload";
    return (
      <button
        onClick={onExpand}
        className="absolute top-3 left-0 z-20 flex flex-col items-center gap-2 px-1.5 py-3
                   bg-ink-800/95 border border-ink-700 border-l-0 backdrop-blur shadow-lg
                   text-ink-300 hover:text-accent-500"
        title="Expand files panel"
      >
        <ChevronRight className="w-4 h-4" />
        <span
          className="text-[10px] tracking-wider uppercase font-medium whitespace-nowrap"
          style={{ writingMode: "vertical-rl" }}
        >
          {label}
        </span>
      </button>
    );
  }

  return (
    <aside
      className="absolute top-3 left-3 w-[300px] max-h-[60vh]
                 bg-ink-800/95 border border-ink-700 backdrop-blur
                 flex flex-col overflow-hidden shadow-lg z-20"
    >
      {/* No tab selected: nudge the user to pick one. */}
      {!tab && (
        <ChooseTabHint onMinimize={onMinimize} />
      )}

      {tab && (
        <div
          key={tab}
          className="flex flex-col flex-1 overflow-hidden animate-slide-in-right min-h-0"
        >
          {tab === "upload"
            ? <UploadPane
                selected={selected}
                onSelect={onSelect}
                onIngested={onIngested}
                onRun={onRunInference}
                onMinimize={onMinimize}
              />
            : <DatasetsPane
                tab={tab}
                selected={selected}
                onSelect={onSelect}
                onRunInference={onRunInference}
                onMinimize={onMinimize}
              />
          }
        </div>
      )}
    </aside>
  );
}

function ChooseTabHint({ onMinimize }: { onMinimize: () => void }) {
  return (
    <>
      <div className="px-4 py-3 border-b border-ink-700 flex items-center flex-shrink-0">
        <div className="text-[10px] uppercase tracking-wider text-ink-400 font-medium flex-1">
          Files
        </div>
        <button
          onClick={onMinimize}
          className="w-7 h-7 grid place-items-center text-ink-400 hover:text-ink-300"
          title="Minimise to left edge"
        >
          <Minus className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="px-4 py-6 text-center text-[11px] text-ink-400">
        <UploadIcon className="w-5 h-5 mx-auto mb-2 text-ink-500" />
        Choose one of{" "}
        <span className="text-accent-500 font-medium">Train</span>,{" "}
        <span className="text-accent-500 font-medium">Test</span>, or{" "}
        <span className="text-accent-500 font-medium">Upload</span>{" "}
        from the left panel to see its files.
      </div>
    </>
  );
}

// ─── Train / Test panes ────────────────────────────────────────────────

function DatasetsPane({
  tab, selected, onSelect, onRunInference, onMinimize,
}: {
  tab: "train" | "test";
  selected: string | null;
  onSelect: (name: string) => void;
  onRunInference: (name: string) => void;
  onMinimize?: () => void;
}) {
  const q = useQuery({
    queryKey: ["datasets"],
    queryFn: api.datasets,
    refetchInterval: 10_000,
  });

  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const all = q.data ?? [];
    // Upload-origin datasets never belong in the Train / Test
    // galleries — they live exclusively in the Upload tab's
    // "Recent uploads" panel. Without this filter, the Test tab
    // ends up with every uploaded file (uploads have `has_metrics`
    // false because there are no GT shapefiles for them).
    const visible = all.filter((d) => !d.is_upload);
    const byTab = tab === "train"
      ? visible.filter((d) => d.has_metrics)
      : visible.filter((d) => !d.has_metrics);
    const needle = query.trim().toLowerCase();
    if (!needle) return byTab;
    return byTab.filter((d) => d.name.toLowerCase().includes(needle));
  }, [q.data, tab, query]);

  return (
    <>
      <PaneHeader
        eyebrow={tab === "train" ? "Training Set" : "Test Set"}
        title={`${filtered.length} image${filtered.length === 1 ? "" : "s"}`}
        search={query}
        onSearch={setQuery}
        onMinimize={onMinimize}
      />

      <div className="flex-1 overflow-y-auto">
        {q.isLoading && <SkeletonRows />}
        {q.error && <ErrorBlock msg={String(q.error)} />}
        {!q.isLoading && filtered.length === 0 && (
          <Empty msg={query
            ? `No matches for "${query}"`
            : `No ${tab} datasets yet.`} />
        )}

        {filtered.map((d) => (
          <DatasetRow
            key={d.name}
            ds={d}
            selected={selected === d.name}
            onClick={() => onSelect(d.name)}
            onRun={() => onRunInference(d.name)}
            showGtBadge={tab === "train"}
          />
        ))}
      </div>

    </>
  );
}

function DatasetRow({
  ds, selected, onClick, onRun, showGtBadge,
}: {
  ds: DatasetSummary;
  selected: boolean;
  onClick: () => void;
  onRun: () => void;
  showGtBadge: boolean;
}) {
  return (
    <div
      onClick={onClick}
      className={`flex items-center gap-2.5 px-3 py-2.5 border-b border-ink-700 cursor-pointer
                  ${selected
                    ? "bg-accent-500/10 border-l-2 border-l-accent-500 pl-[10px]"
                    : "border-l-2 border-l-transparent hover:bg-ink-800/60"}`}
    >
      <FileThumb seed={ds.name} highlighted={selected} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span
            title={ds.name}
            className="text-xs font-semibold text-ink-300 font-mono truncate"
          >
            {ds.base_name ?? ds.name}
          </span>
          {showGtBadge && (
            <span className="text-[9px] text-gt border border-emerald-900 px-1 tracking-wider font-medium">
              GT
            </span>
          )}
          {ds.checkpoint && (
            <span
              title={`Model checkpoint: ${ds.checkpoint}`}
              className="text-[9px] text-accent-500 border border-accent-500/40 px-1
                         tracking-wider font-medium truncate max-w-[90px] flex-shrink-0"
            >
              {ds.checkpoint}
            </span>
          )}
          {ds.origin === "offline" && (
            <span
              title="Read-only library result from the bundled CLI run"
              className="text-[9px] text-ink-400 border border-ink-600 px-1 tracking-wider font-medium"
            >
              LIB
            </span>
          )}
        </div>
        <div className="text-[10px] text-ink-400 font-mono mt-0.5 flex items-center gap-1.5 flex-wrap">
          {ds.has_gpkg && (
            <span>
              GPKG
              {ds.layer_count != null && ds.layer_count > 0 && ` · ${ds.layer_count}`}
            </span>
          )}
          {ds.has_metrics && <span>· metrics</span>}
          {!ds.has_gpkg && !ds.has_metrics && <span className="text-ink-500">no outputs yet</span>}
        </div>
      </div>

      <div className="flex items-center gap-1">
        <button
          onClick={(e) => { e.stopPropagation(); onRun(); }}
          title={ds.has_gpkg ? "Re-run inference" : "Run inference"}
          className="w-7 h-7 grid place-items-center text-ink-400 hover:text-accent-500
                     hover:bg-accent-500/10 transition-colors"
        >
          <Play className="w-3.5 h-3.5" />
        </button>
        {selected && <Crosshair className="w-3.5 h-3.5 text-accent-500" />}
      </div>
    </div>
  );
}

// ─── Upload pane ───────────────────────────────────────────────────────

function UploadPane({
  selected, onSelect, onIngested, onRun, onMinimize,
}: {
  selected: string | null;
  onSelect: (name: string) => void;
  onIngested?: (name: string) => void;
  onRun: (name: string) => void;
  onMinimize?: () => void;
}) {
  const [staged, setStaged] = useState<UploadResponse | null>(null);

  // Past uploads sitting in portal_workspace. Refetched every 5 s so
  // the list reflects "Run inference" → "Has outputs" transitions
  // without the user needing to manually refresh.
  const uploadsQ = useQuery({
    queryKey: ["uploads", "list"],
    queryFn: api.listUploads,
    refetchInterval: 5_000,
  });
  const uploads = uploadsQ.data?.uploads ?? [];

  const onUploaded = (resp: UploadResponse) => {
    setStaged(resp);
    onIngested?.(resp.dataset_name);
    uploadsQ.refetch();
  };

  const onServerPicked = (name: string) => {
    setStaged({ dataset_name: name, saved_to: "", bytes: 0 });
    onIngested?.(name);
    uploadsQ.refetch();
  };

  const deleteUpload = async (name: string) => {
    if (!confirm(`Delete ${name}? This removes the entire portal_workspace folder for it.`)) return;
    try {
      await api.deleteUpload(name);
    } catch (e) {
      alert(`Delete failed: ${e}`);
    } finally {
      uploadsQ.refetch();
    }
  };

  return (
    <>
      <PaneHeader eyebrow="Upload" title="New imagery" onMinimize={onMinimize} />

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {/* Drop zone — chunked upload */}
        <div className="bg-ink-800 border border-ink-700 p-3">
          <UploadDropzone onUploaded={onUploaded} />
        </div>

        {/* Or pick from server */}
        <ServerFilesQuickPicker onPicked={onServerPicked} />

        {/* Staged file card — appears the moment the upload finishes */}
        {staged && (
          <div className="border border-accent-500/40 bg-ink-800">
            <div className="p-3 flex gap-2.5 items-center border-b border-ink-700">
              <FileThumb seed={staged.dataset_name} highlighted={false} />
              <div className="flex-1 min-w-0">
                <div className="text-xs font-semibold text-ink-300 font-mono truncate">
                  {staged.dataset_name}
                </div>
                <div className="text-[10px] text-ink-400 font-mono mt-0.5">
                  {staged.bytes
                    ? `${(staged.bytes / (1024 ** 3)).toFixed(2)} GB`
                    : "linked from server"} · ready to analyse
                </div>
              </div>
            </div>
            <div className="p-3 flex flex-col gap-2">
              <button
                onClick={() => onRun(staged.dataset_name)}
                className="h-9 w-full bg-accent-500 text-ink-900 font-semibold text-xs
                            inline-flex items-center justify-center gap-2 hover:bg-accent-600
                            transition-colors"
              >
                <Play className="w-3.5 h-3.5" /> Run inference
              </button>
            </div>
          </div>
        )}

        {/* Recent uploads list — every upload_* job in portal_workspace */}
        <div>
          <div className="text-[10px] uppercase tracking-wider text-ink-400 font-medium mb-2 flex items-center gap-2">
            <span>Recent uploads</span>
            <span className="text-[10px] text-ink-500 font-mono tabular-nums">
              {uploads.length}
            </span>
            {uploadsQ.isFetching && (
              <Loader2 className="w-3 h-3 animate-spin text-ink-500" />
            )}
          </div>

          {uploads.length === 0 ? (
            <div className="text-[11px] text-ink-500 italic px-1">
              Nothing uploaded yet — drop a raster above or pick one from the server.
            </div>
          ) : (
            <div className="space-y-1">
              {uploads.map((u) => (
                <UploadRowCard
                  key={u.dataset_name}
                  row={u}
                  selected={selected === u.dataset_name}
                  onSelect={() => onSelect(u.dataset_name)}
                  onRun={() => onRun(u.dataset_name)}
                  onDelete={() => deleteUpload(u.dataset_name)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

/** One row in the Recent uploads list. Clicking the body selects the
 *  dataset (loads it onto the map); the trailing action button is
 *  context-sensitive — "View" if outputs already exist, "Run" if not. */
function UploadRowCard({
  row, selected, onSelect, onRun, onDelete,
}: {
  row: UploadRow;
  selected: boolean;
  onSelect: () => void;
  onRun: () => void;
  onDelete: () => void;
}) {
  const sizeStr = row.bytes
    ? row.bytes > 1024 ** 3
      ? `${(row.bytes / (1024 ** 3)).toFixed(2)} GB`
      : `${(row.bytes / (1024 ** 2)).toFixed(0)} MB`
    : "—";
  const ageStr = row.created_at
    ? formatRelative(Date.now() / 1000 - row.created_at)
    : "";

  return (
    <div
      onClick={onSelect}
      className={`flex items-center gap-2 px-2 py-2 border cursor-pointer
                  ${selected
                    ? "bg-accent-500/10 border-accent-500/50"
                    : "border-ink-700 hover:bg-ink-800/60"}`}
    >
      <FileThumb seed={row.dataset_name} highlighted={selected} />
      <div className="flex-1 min-w-0">
        <div className="text-[11px] font-semibold text-ink-300 font-mono truncate">
          {row.display_name}
        </div>
        <div className="text-[10px] text-ink-400 font-mono mt-0.5 flex items-center gap-1.5 flex-wrap">
          <span>{sizeStr}</span>
          {ageStr && (<><span className="text-ink-600">·</span><span>{ageStr}</span></>)}
          {row.has_outputs
            ? <span className="text-gt">· ready</span>
            : <span className="text-ink-500">· no outputs</span>}
        </div>
      </div>
      {row.has_outputs ? (
        <button
          onClick={(e) => { e.stopPropagation(); onSelect(); }}
          title="Load results on the map"
          className="w-7 h-7 grid place-items-center text-gt hover:bg-emerald-500/10"
        >
          <Eye className="w-3.5 h-3.5" />
        </button>
      ) : (
        <button
          onClick={(e) => { e.stopPropagation(); onRun(); }}
          title="Run inference on this upload"
          className="w-7 h-7 grid place-items-center text-accent-500 hover:bg-accent-500/10"
        >
          <Play className="w-3.5 h-3.5" />
        </button>
      )}
      <button
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        title="Delete this upload + its outputs"
        className="w-7 h-7 grid place-items-center text-ink-400 hover:text-red-400 hover:bg-red-500/10"
      >
        <Trash2 className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

function formatRelative(seconds: number): string {
  if (seconds < 0) return "now";
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

/** Inline server-files quick picker — collapsible list. */
function ServerFilesQuickPicker({
  onPicked,
}: {
  onPicked: (name: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const q = useQuery({
    queryKey: ["server-files"],
    queryFn: api.serverFiles,
    enabled: open,
  });
  const [busy, setBusy] = useState<string | null>(null);

  const pick = async (f: ServerFile) => {
    setBusy(f.path);
    try {
      const r = await api.pickServerFile(f.path);
      onPicked(r.dataset_name);
    } catch (e) {
      console.warn("pick failed", e);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="border border-ink-700 bg-ink-800">
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-3 py-2 text-left flex items-center gap-2 text-[11px] text-ink-300
                   hover:bg-ink-800/60"
      >
        <FolderOpen className="w-3.5 h-3.5 text-ink-400" />
        <span>Pick a file already on the server</span>
        <span className="flex-1" />
        <span className="text-ink-400 text-[10px]">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="border-t border-ink-700 max-h-64 overflow-y-auto">
          {q.isLoading && <div className="p-3 text-[10px] text-ink-400">loading…</div>}
          {q.data?.roots.map((r) => (
            <div key={r.root}>
              <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider text-ink-400 bg-ink-900">
                {r.label} · {r.count}
              </div>
              {r.files.map((f) => (
                <button
                  key={f.path}
                  onClick={() => pick(f)}
                  disabled={busy === f.path}
                  className="w-full px-3 py-1.5 flex items-center gap-2 text-left
                              hover:bg-ink-700/40 disabled:opacity-50"
                >
                  <Folder className="w-3 h-3 text-ink-400 flex-shrink-0" />
                  <span className="text-[10px] text-ink-300 font-mono truncate flex-1">
                    {f.rel}
                  </span>
                  <span className="text-[10px] text-ink-400 font-mono tabular-nums">
                    {f.size_mb.toFixed(0)} MB
                  </span>
                  {busy === f.path && <Loader2 className="w-3 h-3 animate-spin text-accent-500" />}
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── shared bits ──────────────────────────────────────────────────────

function PaneHeader({
  eyebrow, title, search, onSearch, onMinimize,
}: {
  eyebrow: string;
  title: string;
  search?: string;
  onSearch?: (s: string) => void;
  onMinimize?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const showInput = !!onSearch && (open || !!search);

  return (
    <div className="px-4 py-3 border-b border-ink-700 flex items-center gap-2 flex-shrink-0">
      {!showInput && (
        <>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-400 font-medium">
              {eyebrow}
            </div>
            <div className="text-sm font-semibold text-ink-300 mt-0.5">{title}</div>
          </div>
          <div className="flex-1" />
        </>
      )}
      {showInput && (
        <div className="flex-1 relative">
          <Search className="w-3 h-3 text-ink-400 absolute left-2 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            autoFocus
            value={search ?? ""}
            onChange={(e) => onSearch?.(e.target.value)}
            placeholder="Search datasets…"
            className="w-full h-7 pl-7 pr-7 bg-ink-800 border border-ink-700
                       text-[11px] text-ink-300 placeholder:text-ink-500
                       focus:outline-none focus:border-accent-500"
          />
          {(search ?? "") !== "" && (
            <button
              onClick={() => onSearch?.("")}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 w-4 h-4 grid place-items-center text-ink-400 hover:text-ink-300"
              aria-label="Clear search"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      )}
      {onSearch && (
        <button
          onClick={() => setOpen(!open)}
          className={`w-7 h-7 grid place-items-center
                      ${showInput ? "text-accent-500" : "text-ink-400 hover:text-ink-300"}`}
          title={showInput ? "Close search" : "Search"}
        >
          {showInput ? <X className="w-3.5 h-3.5" /> : <Search className="w-3.5 h-3.5" />}
        </button>
      )}
      {onMinimize && (
        <button
          onClick={onMinimize}
          className="w-7 h-7 grid place-items-center text-ink-400 hover:text-ink-300 border-l border-ink-700 ml-1"
          title="Minimise to left edge"
        >
          <Minus className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
}

function SkeletonRows() {
  return (
    <div className="p-3 space-y-2">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="h-14 bg-ink-800 border border-ink-700 animate-pulse" />
      ))}
    </div>
  );
}

function ErrorBlock({ msg }: { msg: string }) {
  return (
    <div className="p-3 text-[11px] text-red-400 font-mono break-words">
      {msg}
    </div>
  );
}

function Empty({ msg }: { msg: string }) {
  return (
    <div className="p-6 text-center text-[11px] text-ink-400">
      <UploadIcon className="w-5 h-5 mx-auto mb-2 text-ink-500" />
      {msg}
    </div>
  );
}

/**
 * 48×48 deterministic micro-aerial preview keyed off the dataset name —
 * gives each row a visually distinct thumb without needing a real
 * preview image.
 */
function FileThumb({ seed, highlighted }: { seed: string; highlighted: boolean }) {
  let s = 0;
  for (let i = 0; i < seed.length; i++) s = (s * 31 + seed.charCodeAt(i)) % 1000;
  const rand = () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
  const tones = [
    "#7a3a26", "#8a4a30", "#a05a3d", "#6e3520", "#5c4a3a",
    "#264a20", "#1f3a1d", "#2a2520",
  ];
  const cells = Array.from({ length: 24 }, () => tones[Math.floor(rand() * tones.length)]);
  return (
    <div
      className="w-12 h-12 grid grid-cols-6 grid-rows-4 flex-shrink-0 border"
      style={{ borderColor: highlighted ? "#ff5600" : "#2a2a2a" }}
    >
      {cells.map((c, i) => (
        <div key={i} style={{ background: c }} />
      ))}
    </div>
  );
}
