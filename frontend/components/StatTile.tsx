import React from "react";

export function StatTile({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 rounded-xl p-4">
      <div className="text-xs text-neutral-500 dark:text-neutral-400 tracking-wide">{label}</div>
      <div className="text-2xl font-mono mt-1 text-neutral-900 dark:text-neutral-50">{value}</div>
    </div>
  );
}
