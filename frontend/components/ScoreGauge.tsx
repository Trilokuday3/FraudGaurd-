import React from "react";

interface ScoreGaugeProps {
  score: number;
  tReview?: number;
  tBlock?: number;
}

export function ScoreGauge({ score, tReview = 0.5, tBlock = 0.9 }: ScoreGaugeProps) {
  const pct = Math.round(score * 100);
  const color = score >= tBlock ? "bg-red-500" : score >= tReview ? "bg-amber-500" : "bg-emerald-500";
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-neutral-400 mb-1">
        <span>Fraud score</span>
        <span className="font-mono">{pct}%</span>
      </div>
      <div className="w-full h-2 bg-neutral-800 rounded overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
