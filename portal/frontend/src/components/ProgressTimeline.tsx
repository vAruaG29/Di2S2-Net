import { useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, ChevronDown, Circle, Loader2, Ban } from "lucide-react";
import { JobEvent } from "../api/sse";

interface Props {
  events: JobEvent[];
  status: "idle" | "open" | "done" | "failed" | "error" | "cancelled";
}

interface Phase {
  index: number;
  total: number;
  label: string;
  status: "pending" | "start" | "done" | "failed" | "skipped";
  startTs?: string;
  endTs?: string;
  steps: { action: string; label: string; t_iso: string; elapsed_s: number | null }[];
}

export function ProgressTimeline({ events, status }: Props) {
  const phases = useMemo(() => buildPhases(events), [events]);
  const logLines = useMemo(
    () => events.filter((e) => e.type === "log").slice(-200),
    [events],
  );
  const errorEv = events.find((e) => e.type === "error") as (JobEvent & { type: "error" }) | undefined;

  return (
    <div className="space-y-3">
      <ol className="space-y-2">
        {phases.map((p) => <PhaseRow key={p.index} phase={p} />)}
      </ol>

      {errorEv && (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">
          <div className="flex items-center gap-1.5 font-semibold mb-1">
            <AlertTriangle className="w-3.5 h-3.5" /> pipeline failed
          </div>
          <pre className="font-mono whitespace-pre-wrap text-red-200/90">{errorEv.message}</pre>
        </div>
      )}

      {logLines.length > 0 && (
        <details className="rounded-md border border-ink-700 bg-ink-800/50">
          <summary className="px-3 py-2 cursor-pointer text-[11px] uppercase tracking-wider text-ink-400 flex items-center gap-1 hover:text-ink-300">
            <ChevronDown className="w-3 h-3" /> stdout ({logLines.length} lines)
          </summary>
          <pre className="px-3 pb-3 text-[10px] font-mono leading-snug text-ink-400 overflow-x-auto max-h-64">
            {logLines.map((l, i) => (l.type === "log" ? l.line : "") + "\n").join("")}
          </pre>
        </details>
      )}

      <div className="text-[11px] text-ink-400 flex items-center gap-2">
        <span className={`w-1.5 h-1.5 rounded-full
                          ${status === "done" ? "bg-emerald-400"
                          : status === "failed" || status === "error" ? "bg-red-400"
                          : status === "cancelled" ? "bg-amber-400"
                          : "bg-accent-500 animate-pulse"}`} />
        status: <span className="text-ink-300">{status}</span>
        {status === "cancelled" && <Ban className="w-3 h-3 text-amber-400" />}
      </div>
    </div>
  );
}

function PhaseRow({ phase }: { phase: Phase }) {
  const [open, setOpen] = useState(false);
  const icon = {
    pending: <Circle className="w-4 h-4 text-ink-500" />,
    start:   <Loader2 className="w-4 h-4 text-accent-500 animate-spin" />,
    done:    <CheckCircle2 className="w-4 h-4 text-emerald-400" />,
    failed:  <AlertTriangle className="w-4 h-4 text-red-400" />,
    skipped: <CheckCircle2 className="w-4 h-4 text-ink-400" />,
  }[phase.status];

  const elapsed =
    phase.startTs && phase.endTs ? elapsedFromIso(phase.startTs, phase.endTs) : null;

  return (
    <li className={`rounded-md border transition-colors
                    ${phase.status === "start"
                      ? "bg-accent-500/5 border-accent-500/30"
                      : phase.status === "done"
                        ? "bg-emerald-500/5 border-emerald-500/20"
                        : phase.status === "failed"
                          ? "bg-red-500/5 border-red-500/30"
                          : "bg-ink-800 border-ink-700"}`}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-3 py-2.5 text-left"
      >
        <span>{icon}</span>
        <span className="flex-1 text-sm text-ink-300">{phase.label}</span>
        {elapsed && (
          <span className="text-[11px] text-ink-400 font-mono tabular-nums">{elapsed}</span>
        )}
        {phase.steps.length > 0 && (
          <ChevronDown className={`w-3 h-3 text-ink-400 transition-transform ${open ? "rotate-180" : ""}`} />
        )}
      </button>
      {open && phase.steps.length > 0 && (
        <ul className="px-3 pb-3 pl-10 space-y-1 text-[11px] font-mono">
          {phase.steps.map((s, i) => (
            <li key={i} className="flex items-center gap-2 text-ink-400">
              <span className={`px-1 py-0.5 rounded
                                ${s.action === "DONE" ? "bg-emerald-500/15 text-emerald-300"
                                : s.action === "START" ? "bg-accent-500/15 text-accent-500"
                                : "bg-ink-700 text-ink-300"}`}>
                {s.action}
              </span>
              <span className="flex-1 truncate">{s.label}</span>
              {s.elapsed_s != null && (
                <span className="text-ink-300 tabular-nums">{s.elapsed_s.toFixed(2)}s</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

const PHASE_TOTAL = 5;
const PLACEHOLDER_LABELS = [
  "1/5 Convert to COG",
  "2/5 Tile rasters",
  "3/5 Prepare labels",
  "4/5 Inference + stitch + eval",
  "5/5 Vectorise to GeoPackage",
];

function buildPhases(events: JobEvent[]): Phase[] {
  // Initialise five placeholder rows so the UI feels solid even before
  // the first event lands.
  const phases: Phase[] = PLACEHOLDER_LABELS.map((label, i) => ({
    index: i, total: PHASE_TOTAL, label, status: "pending", steps: [],
  }));

  let cursor = -1;
  for (const ev of events) {
    if (ev.type === "phase") {
      const p = phases[ev.index];
      if (!p) continue;
      p.label = ev.label;
      if (ev.status === "start") {
        p.status = "start";
        p.startTs = ev.t_iso;
        cursor = ev.index;
      } else {
        p.status = ev.status as Phase["status"];
        p.endTs = ev.t_iso;
      }
    } else if (ev.type === "step" && cursor >= 0) {
      phases[cursor].steps.push({
        action: ev.action,
        label: ev.label,
        t_iso: ev.t_iso,
        elapsed_s: ev.elapsed_s,
      });
    }
  }
  return phases;
}

function elapsedFromIso(a: string, b: string): string {
  const [ah, am, as] = a.split(":").map(Number);
  const [bh, bm, bs] = b.split(":").map(Number);
  let s = (bh * 3600 + bm * 60 + bs) - (ah * 3600 + am * 60 + as);
  if (s < 0) s += 86400;
  const m = Math.floor(s / 60);
  s = s % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}
