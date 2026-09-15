import { fetchModelMetadata } from "@/lib/api";
import { ThresholdSlider } from "@/components/ThresholdSlider";

export default async function ThresholdSimulatorPage() {
  const metadata = await fetchModelMetadata();

  return (
    <main className="p-6 space-y-6">
      <h1 className="text-xl font-semibold">Threshold Simulator</h1>
      <p className="text-sm text-neutral-400">
        Drag to see the estimated realized cost at each <code>t_review</code>{" "}
        threshold, from the precomputed cost curve. Deployed thresholds:
        t_review={metadata.t_review}, t_block={metadata.t_block}.
      </p>
      <ThresholdSlider curve={metadata.cost_curve} initialTReview={metadata.t_review} />
    </main>
  );
}
