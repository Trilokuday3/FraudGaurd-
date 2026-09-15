import { fetchDecisionsStats, fetchModelMetadata } from "@/lib/api";
import { StatTile } from "@/components/StatTile";
import { DecisionBadge } from "@/components/DecisionBadge";

export default async function DashboardPage() {
  const [stats, metadata] = await Promise.all([fetchDecisionsStats(), fetchModelMetadata()]);

  return (
    <main className="p-6 space-y-6">
      <h1 className="text-xl font-semibold">Dashboard</h1>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatTile label="Total scored" value={stats.total} />
        <StatTile label="Avg score" value={stats.avg_score.toFixed(3)} />
        <StatTile label="Deployed model" value={metadata.deployed_model_name} />
        <StatTile label="Val PR-AUC" value={metadata.val_pr_auc.toFixed(3)} />
      </div>
      <div className="flex gap-6 items-center flex-wrap">
        <div className="flex items-center gap-2">
          <DecisionBadge decision="approve" />
          <span className="font-mono">{stats.approve_count}</span>
        </div>
        <div className="flex items-center gap-2">
          <DecisionBadge decision="review" />
          <span className="font-mono">{stats.review_count}</span>
        </div>
        <div className="flex items-center gap-2">
          <DecisionBadge decision="block" />
          <span className="font-mono">{stats.block_count}</span>
        </div>
      </div>
    </main>
  );
}
