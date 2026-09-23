"use client";

import { useEffect, useState } from "react";
import { fetchDecisionsStats } from "@/lib/api";
import { DecisionsStatsResponse } from "@/lib/types";
import { PageHeader } from "@/components/PageHeader";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface TrendRow {
  minute: string;
  approve: number;
  review: number;
  block: number;
}

export default function MonitoringPage() {
  const [stats, setStats] = useState<DecisionsStatsResponse | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const data = await fetchDecisionsStats(60);
        if (!cancelled) setStats(data);
      } catch {
        // transient failure; next poll tick recovers
      }
    }

    poll();
    const id = setInterval(poll, 10000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (!stats) {
    return (
      <main className="p-6 text-sm text-neutral-500 dark:text-neutral-400">Loading&hellip;</main>
    );
  }

  const byMinute = new Map<string, TrendRow>();
  for (const b of stats.buckets) {
    const entry = byMinute.get(b.minute) ?? { minute: b.minute, approve: 0, review: 0, block: 0 };
    entry[b.decision] = b.count;
    byMinute.set(b.minute, entry);
  }
  const trend = Array.from(byMinute.values()).sort((a, b) => a.minute.localeCompare(b.minute));

  return (
    <main className="p-6 space-y-6">
      <PageHeader title="Monitoring" subtitle="Last 60 minutes, refreshed every 10 seconds." />
      <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={trend}>
            <CartesianGrid strokeDasharray="3 3" stroke="#9099A8" opacity={0.25} />
            <XAxis dataKey="minute" tick={{ fontSize: 10, fill: "#9099A8" }} />
            <YAxis tick={{ fill: "#9099A8" }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="approve" stackId="a" fill="#10b981" />
            <Bar dataKey="review" stackId="a" fill="#f59e0b" />
            <Bar dataKey="block" stackId="a" fill="#ef4444" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </main>
  );
}
