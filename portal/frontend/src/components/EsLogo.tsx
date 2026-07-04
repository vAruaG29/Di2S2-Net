import { useState } from "react";

interface Props {
  size?: number;
  className?: string;
}

/**
 * EarthSense Labs mark. Renders `public/earthsense-logo.png` (the
 * official brand asset) and falls back to a geometric SVG if the file
 * is missing, so the UI never breaks during a fresh checkout.
 */
export function EsLogo({ size = 18, className }: Props) {
  const [broken, setBroken] = useState(false);

  if (!broken) {
    return (
      <img
        src="/earthsense_logo.png"
        alt="EarthSense Labs"
        width={size}
        height={size}
        className={className}
        onError={() => setBroken(true)}
        style={{ objectFit: "contain" }}
      />
    );
  }

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className={className}
      aria-label="EarthSense Labs"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.6" />
      <path d="M 3 12 Q 12 5 21 12" stroke="currentColor" strokeWidth="1.4" fill="none" />
      <path d="M 3 12 Q 12 19 21 12" stroke="currentColor" strokeWidth="1.4" fill="none" />
      <circle cx="12" cy="12" r="2.2" fill="currentColor" />
    </svg>
  );
}
