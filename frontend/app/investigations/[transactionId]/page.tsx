import { fetchInvestigation, fetchModelMetadata } from "@/lib/api";
import { DecisionBadge } from "@/components/DecisionBadge";
import { ScoreGauge } from "@/components/ScoreGauge";
import { SHAPBarChart } from "@/components/SHAPBarChart";

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
        <p className="text-sm text-neutral-400">
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

  return (
    <main className="p-6 space-y-6">
      <h1 className="text-xl font-semibold font-mono">{investigation.transaction_id}</h1>
      <div className="flex items-center gap-3">
        <DecisionBadge decision={investigation.decision} />
        <span className="text-sm text-neutral-400">via {investigation.decision_source}</span>
      </div>
      <ScoreGauge
        score={investigation.model_score}
        tReview={metadata.t_review}
        tBlock={metadata.t_block}
      />
      {investigation.triggered_rules.length > 0 && (
        <div>
          <div className="text-xs text-neutral-400 uppercase mb-1">Triggered rules</div>
          <ul className="text-sm font-mono list-disc list-inside">
            {investigation.triggered_rules.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
      )}
      <div>
        <div className="text-xs text-neutral-400 uppercase mb-2">Top SHAP contributions</div>
        <SHAPBarChart data={shapData} />
      </div>
    </main>
  );
}
