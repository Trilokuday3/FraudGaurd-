import { fetchDecisions, fetchDecisionsStats, fetchModelMetadata } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { KpiCard } from "@/components/KpiCard";
import { DashboardBoard } from "@/components/DashboardBoard";

export default async function DashboardPage() {
  const [stats, stats24h, metadata, decisions] = await Promise.all([
    fetchDecisionsStats(),
    fetchDecisionsStats(1440),
    fetchModelMetadata(),
    fetchDecisions({ limit: 100 }),
  ]);

  return (
    <main className="p-6 space-y-6">
      <PageHeader
        title="Fraud detection dashboard"
        subtitle="Real-time transaction monitoring and risk analysis"
      />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard
          label="Flagged transactions"
          value={stats.block_count}
          secondary={`${stats24h.block_count} in the last 24h`}
          tone="danger"
        />
        <KpiCard
          label="Under review"
          value={stats.review_count}
          secondary={`${stats24h.review_count} in the last 24h`}
          tone="warning"
        />
        <KpiCard
          label="Approved"
          value={stats.approve_count}
          secondary={`${stats24h.approve_count} in the last 24h`}
          tone="success"
        />
        <KpiCard
          label="Avg risk score"
          value={`${Math.round(stats.avg_score * 100)}%`}
          secondary={`${metadata.deployed_model_name} · val PR-AUC ${metadata.val_pr_auc.toFixed(3)}`}
        />
      </div>

      <DashboardBoard
        rows={decisions.items}
        tReview={metadata.t_review}
        tBlock={metadata.t_block}
      />
    </main>
  );
}
