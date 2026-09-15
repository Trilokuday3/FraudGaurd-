import { describe, it, expect } from "vitest";
import { costAt } from "@/components/ThresholdSlider";

const curve = [
  { t_review: 0.0, cost: 100 },
  { t_review: 0.1, cost: 50 },
  { t_review: 0.2, cost: 80 },
];

describe("costAt", () => {
  it("returns the exact cost at a curve point", () => {
    expect(costAt(curve, 0.1)).toBe(50);
  });

  it("interpolates linearly between two points", () => {
    expect(costAt(curve, 0.05)).toBeCloseTo(75);
  });

  it("clamps below the curve's minimum t_review", () => {
    expect(costAt(curve, -0.5)).toBe(100);
  });

  it("clamps above the curve's maximum t_review", () => {
    expect(costAt(curve, 1.5)).toBe(80);
  });

  it("returns 0 for an empty curve", () => {
    expect(costAt([], 0.1)).toBe(0);
  });
});
