"use client";

import { useMemo, useState } from "react";
import { CostCurvePoint } from "@/lib/types";

export function costAt(curve: CostCurvePoint[], tReview: number): number {
  if (curve.length === 0) return 0;
  const sorted = [...curve].sort((a, b) => a.t_review - b.t_review);
  if (tReview <= sorted[0].t_review) return sorted[0].cost;
  if (tReview >= sorted[sorted.length - 1].t_review) return sorted[sorted.length - 1].cost;

  for (let i = 0; i < sorted.length - 1; i++) {
    const a = sorted[i];
    const b = sorted[i + 1];
    if (tReview >= a.t_review && tReview <= b.t_review) {
      const span = b.t_review - a.t_review;
      if (span === 0) return a.cost;
      const ratio = (tReview - a.t_review) / span;
      return a.cost + ratio * (b.cost - a.cost);
    }
  }
  return sorted[sorted.length - 1].cost;
}

interface ThresholdSliderProps {
  curve: CostCurvePoint[];
  initialTReview: number;
}

export function ThresholdSlider({ curve, initialTReview }: ThresholdSliderProps) {
  const [tReview, setTReview] = useState(initialTReview);
  const cost = useMemo(() => costAt(curve, tReview), [curve, tReview]);

  return (
    <div className="space-y-2">
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={tReview}
        onChange={(e) => setTReview(parseFloat(e.target.value))}
        className="w-full"
      />
      <div className="flex justify-between text-sm font-mono">
        <span>t_review = {tReview.toFixed(2)}</span>
        <span>estimated cost = ${cost.toFixed(2)}</span>
      </div>
    </div>
  );
}
