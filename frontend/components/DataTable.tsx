"use client";

import React, { useState } from "react";
import { DecisionRow } from "@/lib/types";

interface DataTableProps {
  rows: DecisionRow[];
  onRowClick?: (row: DecisionRow) => void;
  pageSize?: number;
}

export function DataTable({ rows, onRowClick, pageSize = 10 }: DataTableProps) {
  const [page, setPage] = useState(0);
  const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
  const start = page * pageSize;
  const visible = rows.slice(start, start + pageSize);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left border-b border-neutral-700">
            <th className="py-2 pr-4">Transaction</th>
            <th className="py-2 pr-4">Score</th>
            <th className="py-2 pr-4">Decision</th>
            <th className="py-2 pr-4">Source</th>
            <th className="py-2 pr-4">Time</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((row) => (
            <tr
              key={row.id}
              className="border-b border-neutral-800 cursor-pointer hover:bg-neutral-800/50"
              onClick={() => onRowClick?.(row)}
            >
              <td className="py-2 pr-4 font-mono">{row.transaction_id}</td>
              <td className="py-2 pr-4 font-mono">{row.model_score.toFixed(3)}</td>
              <td className="py-2 pr-4">{row.decision}</td>
              <td className="py-2 pr-4">{row.decision_source}</td>
              <td className="py-2 pr-4">{new Date(row.created_at).toLocaleTimeString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex items-center gap-2 pt-2">
        <button
          className="px-2 py-1 text-xs border border-neutral-700 rounded disabled:opacity-40"
          onClick={() => setPage((p) => Math.max(0, p - 1))}
          disabled={page === 0}
        >
          Prev
        </button>
        <span className="text-xs text-neutral-400">
          Page {page + 1} of {totalPages}
        </span>
        <button
          className="px-2 py-1 text-xs border border-neutral-700 rounded disabled:opacity-40"
          onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
          disabled={page >= totalPages - 1}
        >
          Next
        </button>
      </div>
    </div>
  );
}
