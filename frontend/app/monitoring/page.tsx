"use client";

import { useEffect, useState } from "react";
import { fetchDecisionsStats } from "@/lib/api";
import { DecisionsStatsResponse } from "@/lib/types";
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
    return <main className="p-6 text-sm text-neutral-400">Loading...</main>;
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
      <h1 className="text-xl font-semibold">Monitoring</h1>
      <p className="text-sm text-neutral-400">Last 60 minutes, refreshed every 10s.</p>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={trend}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="minute" tick={{ fontSize: 10 }} />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="approve" stackId="a" fill="#10b981" />
          <Bar dataKey="review" stackId="a" fill="#f59e0b" />
          <Bar dataKey="block" stackId="a" fill="#ef4444" />
        </BarChart>
      </ResponsiveContainer>
    </main>
  );
}
