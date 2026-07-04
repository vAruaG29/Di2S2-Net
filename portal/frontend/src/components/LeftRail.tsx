import { Brain, Crosshair, Upload } from "lucide-react";
import type { ReactNode } from "react";

import { EsLogo } from "./EsLogo";
import type { LeftTab } from "../types";

interface Props {
  /** Which Train/Test/Upload tab is open. `null` = files panel closed. */
  tab: LeftTab | null;
  onTabChange: (t: LeftTab | null) => void;
  trainCount: number;
  testCount: number;
}

/**
 * Thin vertical rail anchored to the left edge of the portal — the
 * ONLY navigation on this side. Three primary buttons (Train, Test,
 * Upload) stacked vertically; clicking one toggles the FilesPanel.
 *
 * The 4-row layer stack (basemap / image / GT / predictions) no
 * longer lives here — it's been moved to the right side under
 * ClassesFloat, so this rail stays small and uncluttered.
 */
export function LeftRail({
  tab, onTabChange, trainCount, testCount,
}: Props) {
  return (
    <aside
      className="w-16 h-full bg-ink-800 border-r border-ink-700
                 flex flex-col items-stretch flex-shrink-0"
    >
      <a
        href="https://earthsenselabs.com"
        target="_blank"
        rel="noreferrer"
        className="h-20 grid place-items-center border-b border-ink-700
                   text-accent-500 hover:text-accent-400 transition-colors"
        title="EarthSense Labs"
      >
        <EsLogo size={44} />
      </a>

      <RailTab
        active={tab === "train"}
        onClick={() => onTabChange(tab === "train" ? null : "train")}
        icon={<Brain className="w-5 h-5" />}
        label="Train"
        count={trainCount}
      />
      <RailTab
        active={tab === "test"}
        onClick={() => onTabChange(tab === "test" ? null : "test")}
        icon={<Crosshair className="w-5 h-5" />}
        label="Test"
        count={testCount}
      />
      <RailTab
        active={tab === "upload"}
        onClick={() => onTabChange(tab === "upload" ? null : "upload")}
        icon={<Upload className="w-5 h-5" />}
        label="Upload"
      />
    </aside>
  );
}

function RailTab({
  active, onClick, icon, label, count,
}: {
  active: boolean;
  onClick: () => void;
  icon: ReactNode;
  label: string;
  count?: number;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex flex-col items-center justify-center gap-1 py-3
                  border-b border-ink-700 text-[10px] font-medium uppercase tracking-wider
                  transition-colors
                  ${active
                    ? "bg-ink-900 text-accent-500 border-l-2 border-l-accent-500"
                    : "text-ink-400 hover:text-ink-300 border-l-2 border-l-transparent"}`}
    >
      {icon}
      <div className="flex items-center gap-1">
        <span>{label}</span>
        {count != null && (
          <span className="text-[9px] text-ink-400 font-mono tabular-nums normal-case tracking-normal">
            {count}
          </span>
        )}
      </div>
    </button>
  );
}
