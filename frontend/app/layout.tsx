import "./globals.css";
import Link from "next/link";

export const metadata = {
  title: "FraudGuard",
};

const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/live", label: "Live Transactions" },
  { href: "/investigations", label: "Investigations" },
  { href: "/model", label: "Model Center" },
  { href: "/simulator", label: "Threshold Simulator" },
  { href: "/monitoring", label: "Monitoring" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-neutral-950 text-neutral-100 min-h-screen">
        <nav className="border-b border-neutral-800 px-4 py-3 flex gap-4 overflow-x-auto">
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="text-sm whitespace-nowrap hover:text-emerald-400"
            >
              {item.label}
            </Link>
          ))}
        </nav>
        {children}
      </body>
    </html>
  );
}
