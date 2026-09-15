"use client";

import { useEffect, useState, FormEvent } from "react";
import { useRouter } from "next/navigation";
import { fetchDecisions } from "@/lib/api";
import { DecisionRow } from "@/lib/types";
import { DataTable } from "@/components/DataTable";

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
      <h1 className="text-xl font-semibold">Investigations</h1>
      <form onSubmit={handleSearch} className="flex gap-2 flex-wrap">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Look up a transaction ID"
          className="border border-neutral-700 bg-transparent rounded px-2 py-1 text-sm font-mono"
        />
        <button type="submit" className="px-3 py-1 text-sm border border-neutral-700 rounded">
          Look up
        </button>
      </form>
      <DataTable
        rows={rows}
        onRowClick={(row) => router.push(`/investigations/${encodeURIComponent(row.transaction_id)}`)}
      />
    </main>
  );
}
