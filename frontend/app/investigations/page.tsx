"use client";

import { useEffect, useState, FormEvent } from "react";
import { useRouter } from "next/navigation";
import { fetchDecisions } from "@/lib/api";
import { DecisionRow } from "@/lib/types";
import { DataTable } from "@/components/DataTable";
import { PageHeader } from "@/components/PageHeader";

export default function InvestigationsPage() {
  const [rows, setRows] = useState<DecisionRow[]>([]);
  const [query, setQuery] = useState("");
  const router = useRouter();

  useEffect(() => {
    fetchDecisions({ limit: 100 }).then((data) => setRows(data.items));
  }, []);

  function handleSearch(e: FormEvent) {
    e.preventDefault();
    if (query.trim()) {
      router.push(`/investigations/${encodeURIComponent(query.trim())}`);
    }
  }

  return (
    <main className="p-6 space-y-4">
      <PageHeader
        title="Investigations"
        subtitle="Look up any scored transaction by ID, or browse recent decisions below."
        actions={
          <form onSubmit={handleSearch} className="flex gap-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Transaction ID"
              className="border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 rounded-md px-2.5 py-1.5 text-sm font-mono text-neutral-900 dark:text-neutral-100"
            />
            <button
              type="submit"
              className="px-3 py-1.5 text-sm rounded-md bg-brand text-white hover:bg-brand-light transition-colors"
            >
              Look up
            </button>
          </form>
        }
      />
      <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4">
        <DataTable
          rows={rows}
          onRowClick={(row) => router.push(`/investigations/${encodeURIComponent(row.transaction_id)}`)}
        />
      </div>
    </main>
  );
}
