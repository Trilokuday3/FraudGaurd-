"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV } from "./Sidebar";
import { ThemeToggle } from "./ThemeToggle";

export function MobileNav() {
  const pathname = usePathname();

  return (
    <div className="md:hidden border-b border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950">
      <nav className="flex flex-wrap gap-1 px-3 py-2">
        {NAV.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`whitespace-nowrap rounded-md px-2.5 py-1.5 text-sm ${
                active
                  ? "bg-brand/10 text-brand font-medium"
                  : "text-neutral-600 dark:text-neutral-300"
              }`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="px-3 pb-2">
        <ThemeToggle />
      </div>
    </div>
  );
}
