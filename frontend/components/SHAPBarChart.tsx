"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

interface SHAPBarChartProps {
  data: { feature: string; value: number }[];
}

export function SHAPBarChart({ data }: SHAPBarChartProps) {
  const sorted = [...data].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 10);
  return (
    <ResponsiveContainer width="100%" height={Math.max(200, sorted.length * 32)}>
      <BarChart data={sorted} layout="vertical" margin={{ left: 24 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#9099A8" opacity={0.25} />
        <XAxis type="number" tick={{ fill: "#9099A8" }} />
        <YAxis
          type="category"
          dataKey="feature"
          width={160}
          tick={{ fontSize: 11, fill: "#9099A8" }}
        />
        <Tooltip />
        <Bar dataKey="value" fill="#f59e0b" />
      </BarChart>
    </ResponsiveContainer>
  );
}
