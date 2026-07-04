import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite dev proxy: the React app on :5173 forwards /api/* and /tiles/*
// straight to FastAPI on :8000 so we avoid any CORS dance during dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api":   { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/tiles": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
