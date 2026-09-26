"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./ThemeToggle";

export const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/live", label: "Live Transactions" },
  { href: "/investigations", label: "Investigations" },
  { href: "/model", label: "Model Center" },
  { href: "/simulator", label: "Threshold Simulator" },
  { href: "/monitoring", label: "Monitoring" },
];

export interface SidebarCounts {
  block: number;
  review: number;
  approve: number;
}

interface SidebarContentProps {
  counts: SidebarCounts;
  // Runs when a nav link is followed -- the mobile drawer uses it to close.
  onNavigate?: () => void;
  // Rendered at the end of the brand row (the drawer's close button).
  headerAction?: React.ReactNode;
}

// The sidebar's contents, shared by the fixed desktop sidebar and the
// mobile drawer (MobileNav) so the two can't drift apart.
export function SidebarContent({ counts, onNavigate, headerAction }: SidebarContentProps) {
  const pathname = usePathname();

  return (
    <>
      <div className="flex items-center gap-2 px-1 mb-8">
        <span className="w-8 h-8 rounded-lg bg-brand flex items-center justify-center text-white text-sm font-semibold">
          F
        </span>
        <div>
          <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-50">
            FraudGuard
          </div>
          <div className="text-[11px] text-neutral-500 dark:text-neutral-400">
            Risk management platform
          </div>
        </div>
        {headerAction && <div className="ml-auto">{headerAction}</div>}
      </div>

      <div className="mb-6">
        <div className="text-[11px] font-medium text-neutral-400 dark:text-neutral-500 px-2 mb-2">
          Analysis tools
        </div>
        <nav className="space-y-0.5">
          {NAV.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={onNavigate}
                aria-current={active ? "page" : undefined}
                className={`block rounded-md px-2 py-2 md:py-1.5 text-sm transition-colors ${
                  active
                    ? "bg-brand/10 text-brand font-medium"
                    : "text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>

      <div className="mb-6">
        <div className="text-[11px] font-medium text-neutral-400 dark:text-neutral-500 px-2 mb-2">
          Alert status
        </div>
        <div className="space-y-1 px-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2 text-neutral-600 dark:text-neutral-300">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500" /> Blocked
            </span>
            <span className="font-mono text-neutral-500 dark:text-neutral-400">
              {counts.block}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2 text-neutral-600 dark:text-neutral-300">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500" /> Under review
            </span>
            <span className="font-mono text-neutral-500 dark:text-neutral-400">
              {counts.review}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2 text-neutral-600 dark:text-neutral-300">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" /> Approved
            </span>
            <span className="font-mono text-neutral-500 dark:text-neutral-400">
              {counts.approve}
            </span>
          </div>
        </div>
      </div>

      <div className="mt-auto space-y-2">
        <ThemeToggle />
        <div className="flex items-center gap-2 px-2 pt-2 border-t border-neutral-200 dark:border-neutral-800">
          <span className="w-7 h-7 rounded-full bg-neutral-200 dark:bg-neutral-800 flex items-center justify-center text-[11px] font-medium text-neutral-600 dark:text-neutral-300">
            RA
          </span>
          <div>
            <div className="text-xs font-medium text-neutral-700 dark:text-neutral-200">
              Risk analyst
            </div>
            <div className="text-[11px] text-neutral-400 dark:text-neutral-500">
              Read-only demo
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export function Sidebar({ counts }: { counts: SidebarCounts }) {
  return (
    <aside className="hidden md:flex md:w-60 md:flex-col md:fixed md:inset-y-0 border-r border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 px-4 py-5">
      <SidebarContent counts={counts} />
    </aside>
  );
}
