// frontend/app/model/page.tsx
import { fetchModelComparison } from "@/lib/api";
import { StatTile } from "@/components/StatTile";
import { SHAPBarChart } from "@/components/SHAPBarChart";
import { CalibrationCurveChart } from "@/components/CalibrationCurveChart";
import { PageHeader } from "@/components/PageHeader";

export default async function ModelCenterPage() {
  const comparison = await fetchModelComparison();
  const shapData = comparison.shap_importances.map((s) => ({
    feature: s.feature,
    value: s.mean_abs_shap,
  }));

  return (
    <main className="p-6 space-y-6">
      <PageHeader
        title="Model center"
        subtitle="How the deployed model compares against every other trained candidate."
      />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {comparison.candidates.map((c) => (
          <StatTile
            key={c.name}
            label={c.is_deployed ? `${c.name} (deployed)` : c.name}
            value={c.val_pr_auc.toFixed(4)}
          />
        ))}
      </div>
      <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4">
        <div className="text-xs text-neutral-500 dark:text-neutral-400 mb-2">
          Calibration curve
        </div>
        <CalibrationCurveChart data={comparison.calibration_curve} />
      </div>
      <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4">
        <div className="text-xs text-neutral-500 dark:text-neutral-400 mb-2">
          Global SHAP importance
        </div>
        <SHAPBarChart data={shapData} />
      </div>
    </main>
  );
}
