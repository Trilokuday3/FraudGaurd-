"use client";

import React, { useState } from "react";
import { DecisionRow } from "@/lib/types";
import { DecisionBadge } from "./DecisionBadge";

interface DataTableProps {
  rows: DecisionRow[];
  onRowClick?: (row: DecisionRow) => void;
  selectedId?: number;
  pageSize?: number;
}

function formatAmount(row: DecisionRow): string {
  const amount = row.feature_row?.amount;
  return typeof amount === "number" ? `$${amount.toFixed(2)}` : "—";
}

export function DataTable({ rows, onRowClick, selectedId, pageSize = 10 }: DataTableProps) {
  const [page, setPage] = useState(0);
  const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
  const start = page * pageSize;
  const visible = rows.slice(start, start + pageSize);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs sm:text-sm">
        <thead>
          <tr className="text-left border-b border-neutral-200 dark:border-neutral-800 text-neutral-500 dark:text-neutral-400">
            <th className="py-2 pr-2 sm:pr-4 font-medium">Transaction</th>
            <th className="py-2 pr-2 sm:pr-4 font-medium">Amount</th>
            <th className="py-2 pr-2 sm:pr-4 font-medium">Score</th>
            <th className="py-2 pr-2 sm:pr-4 font-medium">Decision</th>
            <th className="py-2 pr-2 sm:pr-4 font-medium hidden sm:table-cell">Source</th>
            <th className="py-2 pr-2 sm:pr-4 font-medium hidden sm:table-cell">Time</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((row) => (
            <tr
              key={row.id}
              className={`border-b border-neutral-100 dark:border-neutral-800/60 cursor-pointer hover:bg-neutral-50 dark:hover:bg-neutral-800/50 ${
                selectedId === row.id ? "bg-brand/5" : ""
              }`}
              onClick={() => onRowClick?.(row)}
            >
              <td className="py-2 pr-2 sm:pr-4 font-mono text-neutral-900 dark:text-neutral-100">
                {row.transaction_id}
              </td>
              <td className="py-2 pr-2 sm:pr-4 font-mono text-neutral-600 dark:text-neutral-300">
                {formatAmount(row)}
              </td>
              <td className="py-2 pr-2 sm:pr-4 font-mono text-neutral-600 dark:text-neutral-300">
                {row.model_score.toFixed(3)}
              </td>
              <td className="py-2 pr-2 sm:pr-4">
                <DecisionBadge decision={row.decision} />
              </td>
              <td className="py-2 pr-2 sm:pr-4 text-neutral-500 dark:text-neutral-400 hidden sm:table-cell">
                {row.decision_source}
              </td>
              <td className="py-2 pr-2 sm:pr-4 text-neutral-500 dark:text-neutral-400 hidden sm:table-cell">
                {new Date(row.created_at).toLocaleTimeString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex items-center gap-2 pt-3">
        <button
          className="px-2 py-1 text-xs border border-neutral-200 dark:border-neutral-700 rounded-md disabled:opacity-40 text-neutral-600 dark:text-neutral-300"
          onClick={() => setPage((p) => Math.max(0, p - 1))}
          disabled={page === 0}
        >
          Prev
        </button>
        <span className="text-xs text-neutral-400 dark:text-neutral-500">
          Page {page + 1} of {totalPages}
        </span>
        <button
          className="px-2 py-1 text-xs border border-neutral-200 dark:border-neutral-700 rounded-md disabled:opacity-40 text-neutral-600 dark:text-neutral-300"
          onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
          disabled={page >= totalPages - 1}
        >
          Next
        </button>
      </div>
    </div>
  );
}
