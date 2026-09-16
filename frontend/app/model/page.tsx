// frontend/app/model/page.tsx
import { fetchModelComparison } from "@/lib/api";
import { StatTile } from "@/components/StatTile";
import { SHAPBarChart } from "@/components/SHAPBarChart";
import { CalibrationCurveChart } from "@/components/CalibrationCurveChart";

export default async function ModelCenterPage() {
  const comparison = await fetchModelComparison();
  const shapData = comparison.shap_importances.map((s) => ({
    feature: s.feature,
    value: s.mean_abs_shap,
  }));

  return (
    <main className="p-6 space-y-6">
      <h1 className="text-xl font-semibold">Model Center</h1>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {comparison.candidates.map((c) => (
          <StatTile
            key={c.name}
            label={c.is_deployed ? `${c.name} (deployed)` : c.name}
            value={c.val_pr_auc.toFixed(4)}
          />
        ))}
      </div>
      <div>
        <div className="text-xs text-neutral-400 uppercase mb-2">Calibration curve</div>
        <CalibrationCurveChart data={comparison.calibration_curve} />
      </div>
      <div>
        <div className="text-xs text-neutral-400 uppercase mb-2">Global SHAP importance</div>
        <SHAPBarChart data={shapData} />
      </div>
    </main>
  );
}
