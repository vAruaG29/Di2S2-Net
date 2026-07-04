/**
 * Chunked, parallel, resumable upload of a single large File.
 *
 * Why this exists: the browser's `fetch(... body: file)` opens one TCP
 * stream — TCP's slow-start + a single window cap rarely saturate a
 * fast link, and a mid-flight drop forces restart from byte 0. By
 * slicing the file and uploading N chunks in parallel we (a) hit line
 * rate sooner, (b) only re-send the failed chunk on a retry, and
 * (c) can pick up where we left off if the browser tab is closed.
 *
 * The backend exposes:
 *   POST   /api/uploads/init                  → { upload_id, total_chunks }
 *   GET    /api/uploads/<id>/status           → { received_chunks[], ... }
 *   PUT    /api/uploads/<id>/chunk/<index>    body: raw bytes
 *   POST   /api/uploads/<id>/finish           → { dataset_name, saved_to, bytes }
 *   DELETE /api/uploads/<id>                  → cancel + free disk
 */

export interface ChunkedProgress {
  uploaded_bytes: number;
  total_bytes: number;
  received_chunks: number;
  total_chunks: number;
  bytes_per_sec: number;
  eta_seconds: number;
}

export interface ChunkedResult {
  dataset_name: string;
  saved_to: string;
  bytes: number;
}

export interface ChunkedUploadOptions {
  /** chunk size in bytes (default 16 MiB) */
  chunkBytes?: number;
  /** how many chunks to upload in parallel (default 6 — browsers cap at 6/host on HTTP/1.1) */
  parallelism?: number;
  /** per-chunk retry attempts on transient error (default 3) */
  retries?: number;
  /** AbortSignal — abort all in-flight chunk uploads */
  signal?: AbortSignal;
  /** called whenever progress changes meaningfully */
  onProgress?: (p: ChunkedProgress) => void;
}

const DEFAULTS = {
  // 32 MiB chunks halve the HTTP round-trip count on multi-GB
  // uploads. Memory cost is bounded by `parallelism × chunkBytes`
  // (~384 MiB peak), which is fine in modern browsers.
  chunkBytes:  32 * 1024 * 1024,
  // 12 parallel chunks: HTTP/2 portals (TLS) actually fan out wide;
  // HTTP/1.1 browsers cap at 6/host so the extras just queue — never
  // slower than 6, often faster.
  parallelism: 12,
  retries:     3,
};

/** Tiny helper: sleep N ms. */
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Upload `file` to the chunked-upload backend. Returns the same
 * shape /api/uploads returns (dataset_name, saved_to, bytes), so the
 * caller can treat it identically to a single-shot upload.
 */
export async function chunkedUpload(
  file: File,
  opts: ChunkedUploadOptions = {},
): Promise<ChunkedResult> {
  const chunkBytes  = opts.chunkBytes  ?? DEFAULTS.chunkBytes;
  const parallelism = opts.parallelism ?? DEFAULTS.parallelism;
  const retries     = opts.retries     ?? DEFAULTS.retries;

  // ── 1. init session ───────────────────────────────────────────────
  const initR = await fetch("/api/uploads/init", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      filename:   file.name,
      total_size: file.size,
      chunk_size: chunkBytes,
    }),
    signal: opts.signal,
  });
  if (!initR.ok) {
    throw new Error(`init → ${initR.status} ${initR.statusText}`);
  }
  const { upload_id, total_chunks, received_chunks } = await initR.json() as {
    upload_id: string;
    total_chunks: number;
    received_chunks: number[];
  };

  // ── 2. plan: which chunks to send (skip the ones already on disk
  //    in case this is a resume of a previous session — though for a
  //    fresh init it's always empty).
  const already = new Set(received_chunks);
  const queue: number[] = [];
  for (let i = 0; i < total_chunks; i++) if (!already.has(i)) queue.push(i);

  let uploadedBytes = file.size > 0
    ? already.size * chunkBytes        // approximate — last chunk could be smaller
    : 0;
  const startTime = performance.now();

  const emit = () => {
    if (!opts.onProgress) return;
    const elapsed = (performance.now() - startTime) / 1000;
    const newBytes = Math.min(uploadedBytes, file.size);
    const bps = elapsed > 0.1 ? newBytes / elapsed : 0;
    const remaining = Math.max(0, file.size - newBytes);
    const eta = bps > 0 ? remaining / bps : 0;
    opts.onProgress({
      uploaded_bytes:   newBytes,
      total_bytes:      file.size,
      received_chunks:  already.size,
      total_chunks,
      bytes_per_sec:    bps,
      eta_seconds:      eta,
    });
  };
  emit();

  // ── 3. parallel worker pool — N concurrent uploads.
  let nextIdx = 0;
  let failure: any = null;

  async function uploadOne(idx: number): Promise<void> {
    const start = idx * chunkBytes;
    const end   = Math.min(start + chunkBytes, file.size);
    const blob  = file.slice(start, end);
    const sz    = end - start;

    let attempt = 0;
    while (true) {
      try {
        const r = await fetch(
          `/api/uploads/${encodeURIComponent(upload_id)}/chunk/${idx}`,
          {
            method: "PUT",
            headers: { "content-type": "application/octet-stream" },
            body:    blob,
            signal:  opts.signal,
          },
        );
        if (!r.ok) throw new Error(`chunk ${idx} → ${r.status} ${r.statusText}`);
        // No need to read body — backend's response is small JSON we don't use.
        already.add(idx);
        uploadedBytes += sz;
        emit();
        return;
      } catch (e: any) {
        if (opts.signal?.aborted) throw new DOMException("aborted", "AbortError");
        attempt += 1;
        if (attempt > retries) throw e;
        // Exponential backoff, capped.
        await sleep(Math.min(2000, 200 * 2 ** attempt));
      }
    }
  }

  async function worker() {
    while (!failure) {
      const myIdx = nextIdx++;
      if (myIdx >= queue.length) return;
      const chunkIdx = queue[myIdx];
      try {
        await uploadOne(chunkIdx);
      } catch (e) {
        failure = e;
        return;
      }
    }
  }

  const workers = Array.from({ length: Math.min(parallelism, queue.length) },
                             () => worker());
  await Promise.all(workers);
  if (failure) {
    // Caller might want to keep the session around for a future
    // resume; we DON'T DELETE on transient failure. Caller can call
    // cancelUpload(upload_id) explicitly.
    throw failure;
  }

  // ── 4. finalise — backend concatenates chunks + creates portal job.
  const finR = await fetch(
    `/api/uploads/${encodeURIComponent(upload_id)}/finish`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body:    JSON.stringify({}),
      signal:  opts.signal,
    },
  );
  if (!finR.ok) {
    throw new Error(`finish → ${finR.status} ${finR.statusText}`);
  }
  const out = await finR.json() as ChunkedResult;
  emit();   // final tick
  return out;
}


/** Cancel a partial session and free its disk on the server. */
export async function cancelChunkedUpload(upload_id: string): Promise<void> {
  await fetch(`/api/uploads/${encodeURIComponent(upload_id)}`, {
    method: "DELETE",
  });
}
