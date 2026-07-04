/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // ES Dark palette
        ink: {
          950: "#0e0e0e",
          900: "#151515",
          800: "#1a1a1a",
          700: "#2a2a2a",
          600: "#3a3a3a",
          500: "#555555",
          400: "#888888",
          300: "#ebebeb",
          200: "#f5f5f5",
        },
        accent: {
          400: "#ff8c42",
          500: "#ff5600",
          600: "#e64a00",
        },
        gt:     "#00B894",
        water:  "#2D98DA",
        info:   "#F1C40F",
      },
      fontFamily: {
        sans: ['"Segoe UI"', "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"Segoe UI Mono"', "ui-monospace", "Menlo", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(255,86,0,0.4), 0 0 18px rgba(255,86,0,0.18)",
      },
      keyframes: {
        slideInRight: {
          from: { transform: "translateX(24px)", opacity: "0" },
          to:   { transform: "translateX(0)",    opacity: "1" },
        },
      },
      animation: {
        "slide-in-right": "slideInRight 0.26s cubic-bezier(0.2,0.7,0.2,1)",
      },
    },
  },
  plugins: [],
};
