"use client";

import React, { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import { SidebarContent, SidebarCounts } from "./Sidebar";

// Phone/tablet navigation (below the md breakpoint, where the fixed
// desktop sidebar is hidden): a slim top bar with a menu button that
// opens the same sidebar contents as a slide-in drawer.
export function MobileNav({ counts }: { counts: SidebarCounts }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const menuButton = useRef<HTMLButtonElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const wasOpen = useRef(false);

  // Close after navigating (also covers the browser back button).
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  useEffect(() => {
    if (!open) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    window.addEventListener("keydown", onKeyDown);
    // Stop the page scrolling behind the open drawer.
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [open]);

  // Keyboard users: move focus into the drawer on open, back to the menu
  // button on close.
  useEffect(() => {
    if (open) {
      closeButton.current?.focus();
    } else if (wasOpen.current) {
      menuButton.current?.focus();
    }
    wasOpen.current = open;
  }, [open]);

  return (
    <div className="md:hidden">
      <header className="sticky top-0 z-30 flex items-center gap-3 h-14 px-3 border-b border-neutral-200 dark:border-neutral-800 bg-white/95 dark:bg-neutral-950/95 backdrop-blur">
        <button
          ref={menuButton}
          type="button"
          onClick={() => setOpen(true)}
          aria-label="Open navigation menu"
          aria-expanded={open}
          aria-controls="mobile-drawer"
          className="w-10 h-10 -ml-1 flex items-center justify-center rounded-md text-neutral-700 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-800"
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 20 20"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.75"
            strokeLinecap="round"
            aria-hidden="true"
          >
            <path d="M3 5.5h14M3 10h14M3 14.5h14" />
          </svg>
        </button>
        <span className="w-7 h-7 rounded-lg bg-brand flex items-center justify-center text-white text-xs font-semibold">
          F
        </span>
        <span className="text-sm font-semibold text-neutral-900 dark:text-neutral-50">
          FraudGuard
        </span>
      </header>

      <div
        onClick={() => setOpen(false)}
        aria-hidden="true"
        className={`fixed inset-0 z-40 bg-black/50 transition-opacity duration-200 motion-reduce:transition-none ${
          open ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      />

      <aside
        id="mobile-drawer"
        role="dialog"
        aria-modal="true"
        aria-label="Navigation menu"
        className={`fixed inset-y-0 left-0 z-50 flex w-72 max-w-[85vw] flex-col overflow-y-auto border-r border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 px-4 py-5 shadow-xl transition-[transform,visibility] duration-200 motion-reduce:transition-none ${
          open ? "translate-x-0 visible" : "-translate-x-full invisible"
        }`}
      >
        <SidebarContent
          counts={counts}
          onNavigate={() => setOpen(false)}
          headerAction={
            <button
              ref={closeButton}
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close navigation menu"
              className="w-9 h-9 flex items-center justify-center rounded-md text-neutral-500 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-800"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 18 18"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                aria-hidden="true"
              >
                <path d="M4 4l10 10M14 4L4 14" />
              </svg>
            </button>
          }
        />
      </aside>
    </div>
  );
}
