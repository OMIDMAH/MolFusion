import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, getAgents, getHealth } from "../client";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

function mockFetchOnce(body: { ok: boolean; status: number; statusText: string; json: () => Promise<unknown> }) {
  globalThis.fetch = vi.fn().mockResolvedValue(body) as unknown as typeof fetch;
}

describe("api client error handling", () => {
  it("throws a clear ApiError when the backend is unreachable", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch")) as unknown as typeof fetch;

    await expect(getHealth()).rejects.toBeInstanceOf(ApiError);
    await expect(getHealth()).rejects.toThrow(/could not reach the molfusion backend/i);
  });

  it("surfaces the backend's detail message on an HTTP error response", async () => {
    mockFetchOnce({
      ok: false,
      status: 400,
      statusText: "Bad Request",
      json: async () => ({ detail: "Unknown agent id 'nope'" }),
    });

    await expect(getAgents()).rejects.toThrow(/unknown agent id/i);
  });

  it("joins FastAPI 422 validation error messages", async () => {
    mockFetchOnce({
      ok: false,
      status: 422,
      statusText: "Unprocessable Entity",
      json: async () => ({
        detail: [{ loc: ["body", "smiles"], msg: "List should have at least 1 item", type: "too_short" }],
      }),
    });

    await expect(getAgents()).rejects.toThrow(/at least 1 item/i);
  });

  it("throws a clear error when the response body is not valid JSON", async () => {
    mockFetchOnce({
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => {
        throw new Error("not json");
      },
    });

    await expect(getHealth()).rejects.toThrow(/not valid json/i);
  });

  it("throws a clear error when the response shape is malformed", async () => {
    mockFetchOnce({
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => ({ unexpected: true }),
    });

    await expect(getAgents()).rejects.toThrow(/malformed/i);
  });
});
