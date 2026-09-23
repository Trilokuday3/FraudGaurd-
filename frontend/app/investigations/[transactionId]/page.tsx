import { fetchInvestigation, fetchModelMetadata } from "@/lib/api";
import { DecisionBadge } from "@/components/DecisionBadge";
import { ScoreGauge } from "@/components/ScoreGauge";
import { SHAPBarChart } from "@/components/SHAPBarChart";
import { PageHeader } from "@/components/PageHeader";

export default async function InvestigationDetailPage({
  params,
}: {
  params: { transactionId: string };
}) {
  const [investigation, metadata] = await Promise.all([
    fetchInvestigation(params.transactionId).catch(() => null),
    fetchModelMetadata(),
  ]);

  if (!investigation) {
    return (
      <main className="p-6">
        <p className="text-sm text-neutral-500 dark:text-neutral-400">
          No decision found for transaction{" "}
          <span className="font-mono">{params.transactionId}</span>.
        </p>
      </main>
    );
  }

  const shapData = Object.entries(investigation.shap_top_features).map(([feature, value]) => ({
    feature,
    value,
  }));
  const fr = investigation.feature_row ?? {};

  return (
    <main className="p-6 space-y-6">
      <PageHeader title={investigation.transaction_id} subtitle="Transaction investigation" />
      <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-5 space-y-5">
        <div className="flex items-center gap-3">
          <DecisionBadge decision={investigation.decision} />
          <span className="text-sm text-neutral-500 dark:text-neutral-400">
            via {investigation.decision_source}
          </span>
        </div>
        <ScoreGauge
          score={investigation.model_score}
          tReview={metadata.t_review}
          tBlock={metadata.t_block}
        />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
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
        </div>
        {investigation.triggered_rules.length > 0 && (
          <div>
            <div className="text-xs text-neutral-400 dark:text-neutral-500 mb-1.5">
              Triggered rules
            </div>
            <div className="flex flex-wrap gap-1.5">
              {investigation.triggered_rules.map((r) => (
                <span
                  key={r}
                  className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300"
                >
                  {r}
                </span>
              ))}
            </div>
          </div>
        )}
        <div>
          <div className="text-xs text-neutral-400 dark:text-neutral-500 mb-2">
            Top SHAP contributions
          </div>
          <SHAPBarChart data={shapData} />
        </div>
      </div>
    </main>
  );
}
