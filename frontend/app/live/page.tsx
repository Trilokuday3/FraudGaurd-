"use client";

import { useEffect, useState } from "react";
import { fetchDecisions } from "@/lib/api";
import { DataTable } from "@/components/DataTable";
import { DecisionRow } from "@/lib/types";

export default function LiveTransactionsPage() {
  const [rows, setRows] = useState<DecisionRow[]>([]);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const data = await fetchDecisions({ limit: 50 });
        if (!cancelled) setRows(data.items);
      } catch {
        // transient failures are expected while the API/replay script is
        // (re)starting locally -- the next poll tick recovers on its own
      }
    }

    poll();
    const id = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <main className="p-6 space-y-4">
      <h1 className="text-xl font-semibold">Live Transactions</h1>
      <p className="text-sm text-neutral-400">
        Updates every few seconds. Run <code>make replay</code> alongside{" "}
        <code>make api</code> to see a continuously growing feed.
      </p>
      <DataTable rows={rows} />
    </main>
  );
}
