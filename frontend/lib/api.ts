import {
  DecisionsListResponse,
  DecisionsStatsResponse,
  InvestigationResponse,
  ModelComparisonResponse,
  ModelMetadataResponse,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function fetchDecisions(
  params: { decision?: string; limit?: number; beforeId?: number } = {}
): Promise<DecisionsListResponse> {
  const search = new URLSearchParams();
  if (params.decision) search.set("decision", params.decision);
  if (params.limit) search.set("limit", String(params.limit));
  if (params.beforeId) search.set("before_id", String(params.beforeId));
  const qs = search.toString();
  return apiGet(`/decisions${qs ? `?${qs}` : ""}`);
}

export function fetchDecisionsStats(sinceMinutes?: number): Promise<DecisionsStatsResponse> {
  const qs = sinceMinutes !== undefined ? `?since_minutes=${sinceMinutes}` : "";
  return apiGet(`/decisions/stats${qs}`);
}

export function fetchModelMetadata(): Promise<ModelMetadataResponse> {
  return apiGet("/model/metadata");
}

export function fetchModelComparison(): Promise<ModelComparisonResponse> {
  return apiGet("/model/comparison");
}

export function fetchInvestigation(transactionId: string): Promise<InvestigationResponse> {
  return apiGet(`/investigations/${encodeURIComponent(transactionId)}`);
}
