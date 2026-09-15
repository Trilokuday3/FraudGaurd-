import React from "react";

export function StatTile({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="border border-neutral-800 rounded-lg p-4">
      <div className="text-xs text-neutral-400 uppercase tracking-wide">{label}</div>
      <div className="text-2xl font-mono mt-1">{value}</div>
    </div>
  );
}
