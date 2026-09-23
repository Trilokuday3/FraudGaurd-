import React from "react";

interface KpiCardProps {
  label: string;
  value: string | number;
  secondary?: string;
  tone?: "neutral" | "danger" | "warning" | "success";
}

const TONE_CLASSES: Record<string, string> = {
  neutral: "bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300",
  danger: "bg-red-50 dark:bg-red-500/10 text-red-600 dark:text-red-400",
  warning: "bg-amber-50 dark:bg-amber-500/10 text-amber-600 dark:text-amber-400",
  success: "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
};

export function KpiCard({ label, value, secondary, tone = "neutral" }: KpiCardProps) {
  return (
    <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4">
      <div className="flex items-start justify-between gap-2">
        <div className="text-xs font-medium text-neutral-500 dark:text-neutral-400">{label}</div>
        <span className={`w-7 h-7 rounded-lg flex items-center justify-center ${TONE_CLASSES[tone]}`}>
          <span className="w-2 h-2 rounded-full bg-current" />
        </span>
      </div>
      <div className="text-2xl font-mono font-semibold mt-2 text-neutral-900 dark:text-neutral-50">
        {value}
      </div>
      {secondary && (
        <div className="text-xs text-neutral-400 dark:text-neutral-500 mt-1">{secondary}</div>
      )}
    </div>
  );
}
