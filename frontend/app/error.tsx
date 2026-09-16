"use client";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <main className="p-6 space-y-4">
      <h1 className="text-xl font-semibold text-red-400">Something went wrong</h1>
      <p className="text-sm text-neutral-400">
        Couldn&apos;t load this page — the FastAPI backend may not be running.
        Check that <code>make api</code> is up at the configured
        <code> NEXT_PUBLIC_API_BASE_URL</code>.
      </p>
      <button
        className="px-3 py-1 text-sm border border-neutral-700 rounded"
        onClick={() => reset()}
      >
        Try again
      </button>
    </main>
  );
}
