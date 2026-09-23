"use client";

import { useMemo, useState } from "react";
import { DataTable } from "./DataTable";
import { TransactionDetailPanel } from "./TransactionDetailPanel";
import { DecisionRow } from "@/lib/types";

const TABS = [
  { key: "all", label: "All" },
  { key: "block", label: "Flagged" },
  { key: "review", label: "Under review" },
  { key: "approve", label: "Approved" },
] as const;

type TabKey = (typeof TABS)[number]["key"];

interface DashboardBoardProps {
  rows: DecisionRow[];
  tReview: number;
  tBlock: number;
}

export function DashboardBoard({ rows, tReview, tBlock }: DashboardBoardProps) {
  const [tab, setTab] = useState<TabKey>("all");
  const [selected, setSelected] = useState<DecisionRow | null>(null);

  const filtered = useMemo(
    () => (tab === "all" ? rows : rows.filter((r) => r.decision === tab)),
    [rows, tab]
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div className="lg:col-span-2 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <div className="text-sm font-medium text-neutral-900 dark:text-neutral-50">
            Flagged transactions
          </div>
          <div className="flex gap-1 bg-neutral-100 dark:bg-neutral-800 rounded-lg p-1">
            {TABS.map((t) => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                  tab === t.key
                    ? "bg-white dark:bg-neutral-700 text-neutral-900 dark:text-neutral-50 shadow-sm"
                    : "text-neutral-500 dark:text-neutral-400"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
        <DataTable
          rows={filtered}
          onRowClick={(row) => setSelected(row)}
          selectedId={selected?.id}
        />
      </div>

      <TransactionDetailPanel
        transactionId={selected?.transaction_id ?? null}
        tReview={tReview}
        tBlock={tBlock}
      />
    </div>
  );
}
