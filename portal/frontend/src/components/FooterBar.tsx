import { EsLogo } from "./EsLogo";

interface Props {
  crs?: string | null;
  lat?: number | null;
  lng?: number | null;
  zoom?: number | null;
  mPerPx?: number | null;
}

/**
 * Full-width 28-px footer running along the very bottom of the
 * viewport. Spans both the left column and the map area.
 *
 *  Left :  EPSG · lat/lng · zoom · m/px (cursor-tracking)
 *  Right:  "Powered by EarthSense Labs" + logo (anchored bottom-right)
 *
 * The map's overlay chrome (IoU panel, zoom buttons, base-style
 * switcher) is positioned WITHIN the map area, so it naturally sits
 * above the top edge of this footer.
 */
export function FooterBar({ crs, lat, lng, zoom, mPerPx }: Props) {
  return (
    <div className="h-7 flex items-center px-4 gap-3 text-[11px] text-ink-400 font-mono
                    bg-ink-800 border-t border-ink-700 flex-shrink-0">
      <span>{crs ?? "EPSG:4326"}</span>
      <Dot />
      <span className="tabular-nums">
        {lat != null && lng != null
          ? `${lat.toFixed(4)}° ${lat >= 0 ? "N" : "S"}, ${lng.toFixed(4)}° ${lng >= 0 ? "E" : "W"}`
          : "—"}
      </span>
      <Dot />
      <span className="tabular-nums">
        {zoom != null ? `z ${zoom.toFixed(2)}` : "z —"}
        {mPerPx != null && `  ·  ${mPerPx.toFixed(2)} m/px`}
      </span>

      <div className="flex-1" />

      <span className="font-sans text-ink-400">Powered by</span>
      <span className="font-sans font-semibold text-ink-300">EarthSense Labs</span>
      <span className="text-accent-500"><EsLogo size={14} /></span>
    </div>
  );
}

function Dot() {
  return <span className="text-ink-600">·</span>;
}
