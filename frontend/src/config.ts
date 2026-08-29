const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

function resolveApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL;
  if (!configured || configured.trim() === "") {
    return DEFAULT_API_BASE_URL;
  }
  return configured.replace(/\/+$/, "");
}

export const API_BASE_URL = resolveApiBaseUrl();
