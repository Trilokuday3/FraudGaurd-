"use client";

import { useEffect, useState } from "react";
import { fetchDecisions } from "@/lib/api";
import { DataTable } from "@/components/DataTable";
import { PageHeader } from "@/components/PageHeader";
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
      <PageHeader
        title="Live transactions"
        subtitle="Updates every few seconds while the replay script is running."
      />
      <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4">
        <DataTable rows={rows} />
      </div>
    </main>
  );
}
