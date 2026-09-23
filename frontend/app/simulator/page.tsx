import { fetchModelMetadata } from "@/lib/api";
import { ThresholdSlider } from "@/components/ThresholdSlider";
import { PageHeader } from "@/components/PageHeader";

export default async function ThresholdSimulatorPage() {
  const metadata = await fetchModelMetadata();

  return (
    <main className="p-6 space-y-6">
      <PageHeader
        title="Threshold simulator"
        subtitle={`Drag to see the estimated realized cost at each review threshold. Deployed: t_review=${metadata.t_review}, t_block=${metadata.t_block}.`}
      />
      <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-5">
        <ThresholdSlider curve={metadata.cost_curve} initialTReview={metadata.t_review} />
      </div>
    </main>
  );
}
