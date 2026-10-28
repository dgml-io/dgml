import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the API runs separately (scripts/start.sh); proxy /api (and the
// OpenAPI document the APIs page reads) to it so the browser sees one origin.
// `scripts/start.sh --build` serves the built app from the API itself instead.
const apiPort = process.env.DGML_SAMPLE_API_PORT ?? "8700";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    proxy: {
      // "/api/", not "/api": the bare prefix would also capture the app's own /apis page.
      "/api/": `http://127.0.0.1:${apiPort}`,
      "/openapi.json": `http://127.0.0.1:${apiPort}`,
    },
  },
});
