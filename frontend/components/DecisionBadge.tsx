import React from "react";

const COLORS: Record<string, string> = {
  approve: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  review: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  block: "bg-red-500/15 text-red-400 border-red-500/30",
};

export function DecisionBadge({ decision }: { decision: string }) {
  const cls = COLORS[decision] ?? "bg-neutral-500/15 text-neutral-400 border-neutral-500/30";
  return (
    <span className={`inline-block px-2 py-0.5 rounded border text-xs font-mono uppercase ${cls}`}>
      {decision}
    </span>
  );
}
