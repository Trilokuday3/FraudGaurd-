import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DataTable } from "@/components/DataTable";
import { DecisionRow } from "@/lib/types";

function makeRows(n: number): DecisionRow[] {
  return Array.from({ length: n }, (_, i) => ({
    id: i,
    transaction_id: `TXN${i}`,
    model_score: 0.1,
    decision: "approve",
    triggered_rules: [],
    decision_source: "model",
    model_run_id: "run1",
    shap_top_features: {},
    created_at: new Date(2026, 0, 1, 10, i).toISOString(),
  }));
}

describe("DataTable pagination", () => {
  it("shows only pageSize rows on the first page", () => {
    render(<DataTable rows={makeRows(25)} pageSize={10} />);
    expect(screen.getByText("TXN0")).toBeInTheDocument();
    expect(screen.queryByText("TXN10")).not.toBeInTheDocument();
    expect(screen.getByText("Page 1 of 3")).toBeInTheDocument();
  });

  it("advances to the next page on click", () => {
    render(<DataTable rows={makeRows(25)} pageSize={10} />);
    fireEvent.click(screen.getByText("Next"));
    expect(screen.getByText("TXN10")).toBeInTheDocument();
    expect(screen.queryByText("TXN0")).not.toBeInTheDocument();
  });

  it("disables Prev on the first page and Next on the last page", () => {
    render(<DataTable rows={makeRows(5)} pageSize={10} />);
    expect(screen.getByText("Prev")).toBeDisabled();
    expect(screen.getByText("Next")).toBeDisabled();
  });

  it("calls onRowClick with the clicked row", () => {
    let clicked: DecisionRow | null = null;
    render(<DataTable rows={makeRows(3)} onRowClick={(row) => (clicked = row)} />);
    fireEvent.click(screen.getByText("TXN1"));
    expect(clicked?.transaction_id).toBe("TXN1");
  });
});
