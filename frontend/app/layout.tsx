import "./globals.css";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import { Sidebar } from "@/components/Sidebar";
import { MobileNav } from "@/components/MobileNav";
import { fetchDecisionsStats } from "@/lib/api";

const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-plex-sans",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-plex-mono",
});

export const metadata = {
  title: "FraudGuard",
};

// Applies the persisted/system theme before first paint, so there is no
// flash of the wrong theme -- reads localStorage first, falling back to
// the OS preference when nothing has been chosen yet.
const THEME_INIT_SCRIPT = `
(function () {
  try {
    var stored = localStorage.getItem("theme");
    var dark = stored ? stored === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.classList.toggle("dark", dark);
  } catch (e) {}
})();
`;

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const counts = await fetchDecisionsStats().catch(() => null);

  return (
    <html lang="en" className={`${plexSans.variable} ${plexMono.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body className="font-sans bg-neutral-50 dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 min-h-screen">
        <Sidebar
          counts={{
            block: counts?.block_count ?? 0,
            review: counts?.review_count ?? 0,
            approve: counts?.approve_count ?? 0,
          }}
        />
        <MobileNav />
        <div className="md:pl-60">{children}</div>
      </body>
    </html>
  );
}
