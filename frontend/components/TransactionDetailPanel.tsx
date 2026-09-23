"use client";

import { useEffect, useState } from "react";
import { fetchInvestigation } from "@/lib/api";
import { InvestigationResponse } from "@/lib/types";
import { DecisionBadge } from "./DecisionBadge";

interface TransactionDetailPanelProps {
  transactionId: string | null;
  tReview: number;
  tBlock: number;
}

function riskLabel(score: number, tReview: number, tBlock: number): string {
  if (score >= tBlock) return "High risk — requires immediate attention";
  if (score >= tReview) return "Medium risk — flagged for review";
  return "Low risk — no action needed";
}

export function TransactionDetailPanel({
  transactionId,
  tReview,
  tBlock,
}: TransactionDetailPanelProps) {
  const [data, setData] = useState<InvestigationResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!transactionId) {
      setData(null);
      return;
    }
    setLoading(true);
    fetchInvestigation(transactionId)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [transactionId]);

  if (!transactionId) {
    return (
      <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 text-sm text-neutral-400 dark:text-neutral-500">
        Select a transaction to see its full risk assessment.
      </div>
    );
  }

  if (loading || !data) {
    return (
      <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 text-sm text-neutral-400 dark:text-neutral-500">
        Loading&hellip;
      </div>
    );
  }

  const pct = Math.round(data.model_score * 100);
  const barColor =
    data.decision === "block" ? "bg-red-500" : data.decision === "review" ? "bg-amber-500" : "bg-emerald-500";
  const bgTint =
    data.decision === "block"
      ? "bg-red-50 dark:bg-red-500/10"
      : data.decision === "review"
      ? "bg-amber-50 dark:bg-amber-500/10"
      : "bg-emerald-50 dark:bg-emerald-500/10";

  const topFactors = Object.entries(data.shap_top_features)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 5);

  const fr = data.feature_row ?? {};

  return (
    <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-5 space-y-5">
      <div className="flex items-center gap-2 text-sm font-medium text-neutral-900 dark:text-neutral-50">
        Transaction analysis
      </div>

      <div className={`rounded-lg p-4 ${bgTint}`}>
        <div className="flex items-center justify-between">
          <span className="text-sm text-neutral-600 dark:text-neutral-300">Risk assessment</span>
          <span className="text-xl font-mono font-semibold text-neutral-900 dark:text-neutral-50">
            {pct}%
          </span>
        </div>
        <div className="w-full h-1.5 bg-neutral-200 dark:bg-neutral-700 rounded-full overflow-hidden mt-2">
          <div className={`h-full ${barColor}`} style={{ width: `${pct}%` }} />
        </div>
        <div className="text-xs text-neutral-500 dark:text-neutral-400 mt-2">
          {riskLabel(data.model_score, tReview, tBlock)}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <div className="text-xs text-neutral-400 dark:text-neutral-500">Transaction ID</div>
          <div className="font-mono text-neutral-800 dark:text-neutral-100">
            {data.transaction_id}
          </div>
        </div>
        <div>
          <div className="text-xs text-neutral-400 dark:text-neutral-500">Amount</div>
          <div className="font-mono text-neutral-800 dark:text-neutral-100">
            {typeof fr.amount === "number" ? `$${fr.amount.toFixed(2)}` : "—"}
          </div>
        </div>
        <div>
          <div className="text-xs text-neutral-400 dark:text-neutral-500">Merchant</div>
          <div className="font-mono text-neutral-800 dark:text-neutral-100">
            {fr.merchant_id ?? "—"}
          </div>
        </div>
        <div>
          <div className="text-xs text-neutral-400 dark:text-neutral-500">Customer</div>
          <div className="font-mono text-neutral-800 dark:text-neutral-100">
            {fr.customer_id ?? "—"}
          </div>
        </div>
        <div>
          <div className="text-xs text-neutral-400 dark:text-neutral-500">Payment method</div>
          <div className="text-neutral-800 dark:text-neutral-100">
            {typeof fr.payment_method === "string" ? fr.payment_method : "—"}
          </div>
        </div>
        <div>
          <div className="text-xs text-neutral-400 dark:text-neutral-500">Scored via</div>
          <div className="text-neutral-800 dark:text-neutral-100">{data.decision_source}</div>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <DecisionBadge decision={data.decision} />
        <span className="text-xs text-neutral-400 dark:text-neutral-500">
          {new Date(data.created_at).toLocaleString()}
        </span>
      </div>

      {data.triggered_rules.length > 0 && (
        <div>
          <div className="text-xs text-neutral-400 dark:text-neutral-500 mb-1.5">Risk factors</div>
          <div className="flex flex-wrap gap-1.5">
            {data.triggered_rules.map((rule) => (
              <span
                key={rule}
                className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300"
              >
                {rule}
              </span>
            ))}
          </div>
        </div>
      )}

      {topFactors.length > 0 && (
        <div>
          <div className="text-xs text-neutral-400 dark:text-neutral-500 mb-1.5">
            Top model contributors
          </div>
          <ul className="space-y-1 text-sm">
            {topFactors.map(([feature, value]) => (
              <li key={feature} className="flex items-center justify-between">
                <span className="text-neutral-700 dark:text-neutral-200">{feature}</span>
                <span
                  className={`font-mono ${value > 0 ? "text-red-500" : "text-emerald-500"}`}
                >
                  {value > 0 ? "+" : ""}
                  {value.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
