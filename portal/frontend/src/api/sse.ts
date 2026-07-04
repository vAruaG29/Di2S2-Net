import { useEffect, useRef, useState } from "react";

/** Shape of an event we expect from /api/jobs/:id/events. */
export type JobEvent =
  | { type: "phase";     index: number; total: number; status: "start"|"done"|"failed"|"skipped"; label: string; t_iso: string }
  | { type: "step";      action: "START"|"DONE"|"cumulative"; label: string; t_iso: string; elapsed_s: number | null; raw: string }
  | { type: "log";       line: string; t_iso: string }
  | { type: "complete";  dataset_name: string; job_root?: string; t_iso: string }
  | { type: "cancelled"; dataset_name: string; job_root?: string; t_iso: string }
  | { type: "error";     message: string; t_iso: string };

/**
 * Subscribe to a job's SSE stream. Re-runs when jobId changes.
 * Returns the accumulated event list + the terminal status.
 */
export function useJobStream(jobId: string | null) {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [status, setStatus] = useState<"idle"|"open"|"done"|"failed"|"error"|"cancelled">("idle");
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!jobId) {
      setEvents([]);
      setStatus("idle");
      return;
    }

    setEvents([]);
    setStatus("open");

    const es = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
    esRef.current = es;

    const onMessage = (ev: MessageEvent) => {
      try {
        const data = JSON.parse(ev.data) as JobEvent;
        setEvents((xs) => [...xs, data]);
        if (data.type === "complete")  setStatus("done");
        if (data.type === "error")     setStatus("failed");
        if (data.type === "cancelled") setStatus("cancelled");
      } catch {
        // ignore malformed
      }
    };

    // Same handler for every named event type the backend emits.
    const types = ["phase", "step", "log", "complete", "cancelled", "error", "ping", "message"];
    for (const t of types) es.addEventListener(t, onMessage as EventListener);

    es.onerror = () => {
      setStatus((s) => (s === "done" || s === "failed" || s === "cancelled" ? s : "error"));
    };

    return () => {
      for (const t of types) es.removeEventListener(t, onMessage as EventListener);
      es.close();
      esRef.current = null;
    };
  }, [jobId]);

  return { events, status };
}
