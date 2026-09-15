import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { DecisionBadge } from "@/components/DecisionBadge";

describe("DecisionBadge", () => {
  it.each([
    ["approve", "emerald"],
    ["review", "amber"],
    ["block", "red"],
  ])("maps %s to a %s-toned class", (decision, colorHint) => {
    render(<DecisionBadge decision={decision} />);
    expect(screen.getByText(decision).className).toContain(colorHint);
  });

  it("falls back to a neutral class for an unrecognized decision", () => {
    render(<DecisionBadge decision="unknown" />);
    expect(screen.getByText("unknown").className).toContain("neutral");
  });
});
