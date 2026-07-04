import { useRef, useState } from "react";
import { CloudUpload, FileImage, Loader2, X } from "lucide-react";
import { UploadResponse } from "../api/client";
import { chunkedUpload, ChunkedProgress } from "../api/chunked_upload";

interface Props {
  onUploaded: (resp: UploadResponse) => void;
}

export function UploadDropzone({ onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<ChunkedProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [last, setLast] = useState<UploadResponse | null>(null);

  const upload = async (file: File) => {
    setBusy(true);
    setError(null);
    setProgress(null);
    abortRef.current = new AbortController();
    try {
      const out = await chunkedUpload(file, {
        chunkBytes:  32 * 1024 * 1024,   // 32 MiB — fewer HTTP round-trips
        parallelism: 12,                  // HTTP/2 fans out; HTTP/1.1 queues
        retries:     3,
        signal:      abortRef.current.signal,
        onProgress:  setProgress,
      });
      const resp: UploadResponse = {
        dataset_name: out.dataset_name,
        saved_to:     out.saved_to,
        bytes:        out.bytes,
      };
      setLast(resp);
      onUploaded(resp);
    } catch (e: any) {
      if (e?.name === "AbortError") {
        setError("Upload cancelled.");
      } else {
        setError(String(e?.message ?? e));
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  const cancel = () => {
    abortRef.current?.abort();
  };

  return (
    <div className="space-y-3">
      <label
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f && !busy) upload(f);
        }}
        className={`flex flex-col items-center justify-center w-full h-44 rounded-lg
                    border-2 border-dashed transition-colors
                    ${dragging
                      ? "border-accent-500 bg-accent-500/5"
                      : "border-ink-600 hover:border-ink-500 bg-ink-800/50"}
                    ${busy ? "cursor-not-allowed" : "cursor-pointer"}`}
      >
        {busy ? (
          <>
            <Loader2 className="w-7 h-7 text-accent-500 animate-spin mb-2" />
            <span className="text-xs text-ink-400">
              {progress
                ? `${(progress.uploaded_bytes / (1024 * 1024 * 1024)).toFixed(2)} / `
                  + `${(progress.total_bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
                : "Preparing…"}
            </span>
          </>
        ) : (
          <>
            <CloudUpload className="w-7 h-7 text-ink-400 mb-2" />
            <span className="text-sm text-ink-300">Drag a raster here, or click to browse</span>
            <span className="text-[11px] text-ink-400 mt-1">
              .tif · .tiff · .ecw  —  chunked + parallel, handles multi-GB
            </span>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".tif,.tiff,.ecw"
          disabled={busy}
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) upload(f);
            if (inputRef.current) inputRef.current.value = "";
          }}
        />
      </label>

      {busy && progress && (
        <UploadProgressBar p={progress} onCancel={cancel} />
      )}

      {error && (
        <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded p-2">
          {error}
        </div>
      )}

      {last && !busy && (
        <div className="text-[11px] bg-ink-800 border border-ink-700 rounded-md p-2.5 flex items-center gap-2">
          <FileImage className="w-3.5 h-3.5 text-accent-500 flex-shrink-0" />
          <div className="min-w-0 flex-1">
            <div className="text-ink-300 truncate font-mono">{last.dataset_name}</div>
            <div className="text-ink-400">
              {(last.bytes / (1024 * 1024 * 1024)).toFixed(2)} GB uploaded
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


function UploadProgressBar({
  p, onCancel,
}: { p: ChunkedProgress; onCancel: () => void }) {
  const pct = p.total_bytes > 0
    ? Math.min(100, (p.uploaded_bytes / p.total_bytes) * 100)
    : 0;
  const mbps = p.bytes_per_sec / (1024 * 1024);
  const eta = formatEta(p.eta_seconds);

  return (
    <div className="rounded-md bg-ink-800 border border-ink-700 p-2.5 space-y-2">
      <div className="flex items-center justify-between text-[11px] text-ink-400 font-mono tabular-nums">
        <span>
          <span className="text-ink-300">{pct.toFixed(1)}%</span>
          {" · "}
          {(p.uploaded_bytes / (1024 * 1024)).toFixed(0)} MB
          {" / "}
          {(p.total_bytes / (1024 * 1024)).toFixed(0)} MB
        </span>
        <span>
          {mbps > 0 ? `${mbps.toFixed(1)} MB/s` : "—"}
          {p.eta_seconds > 0 ? `  ·  ETA ${eta}` : ""}
        </span>
        <button
          onClick={onCancel}
          className="flex items-center gap-1 px-2 py-0.5 rounded
                     text-red-300 hover:text-red-200 hover:bg-red-500/10 transition-colors"
        >
          <X className="w-3 h-3" /> cancel
        </button>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded bg-ink-700">
        <div
          className="h-full bg-accent-500 transition-[width] duration-150"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="text-[10px] text-ink-400 font-mono">
        chunks: {p.received_chunks} / {p.total_chunks}
      </div>
    </div>
  );
}


function formatEta(seconds: number): string {
  if (!isFinite(seconds) || seconds <= 0) return "—";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
  return `${(seconds / 3600).toFixed(1)}h`;
}
