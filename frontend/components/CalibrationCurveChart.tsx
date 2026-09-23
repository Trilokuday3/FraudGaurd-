"use client";

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface CalibrationCurveChartProps {
  data: { mean_predicted: number; fraction_positive: number }[];
}

export function CalibrationCurveChart({ data }: CalibrationCurveChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#9099A8" opacity={0.25} />
        <XAxis dataKey="mean_predicted" type="number" domain={[0, 1]} tick={{ fill: "#9099A8" }} />
        <YAxis dataKey="fraction_positive" domain={[0, 1]} tick={{ fill: "#9099A8" }} />
        <Tooltip />
        <Line type="monotone" dataKey="fraction_positive" stroke="#f59e0b" dot />
      </LineChart>
    </ResponsiveContainer>
  );
}
