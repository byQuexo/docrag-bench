import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev: `npm run dev` serves the UI on :5173 and forwards API calls to the
// FastAPI backend on :8000. Prod: `npm run build` -> dist/, served by FastAPI.
const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(
      ["/query", "/threads", "/pages", "/results", "/documents", "/health"].map((p) => [p, api]),
    ),
  },
});
