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
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="mean_predicted" type="number" domain={[0, 1]} />
        <YAxis dataKey="fraction_positive" domain={[0, 1]} />
        <Tooltip />
        <Line type="monotone" dataKey="fraction_positive" stroke="#f59e0b" dot />
      </LineChart>
    </ResponsiveContainer>
  );
}
