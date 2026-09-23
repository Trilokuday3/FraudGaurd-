"use client";

import { useEffect, useState } from "react";

const THEME_EVENT = "fraudguard-theme-change";

function applyTheme(dark: boolean) {
  document.documentElement.classList.toggle("dark", dark);
  try {
    localStorage.setItem("theme", dark ? "dark" : "light");
  } catch {
    // localStorage can throw in a private window -- theme just won't
    // persist across reloads, which is a harmless degradation
  }
  // Every ThemeToggle instance on the page (desktop sidebar + mobile nav
  // both render one) needs to reflect the same theme -- each holds its
  // own local state, so a plain click on one wouldn't otherwise update
  // the other's label. A custom event keeps them in sync without lifting
  // theme into shared context for what is otherwise a single boolean.
  window.dispatchEvent(new CustomEvent(THEME_EVENT, { detail: { dark } }));
}

export function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    setIsDark(document.documentElement.classList.contains("dark"));
    function onThemeChange(e: Event) {
      setIsDark((e as CustomEvent<{ dark: boolean }>).detail.dark);
    }
    window.addEventListener(THEME_EVENT, onThemeChange);
    return () => window.removeEventListener(THEME_EVENT, onThemeChange);
  }, []);

  function toggle() {
    applyTheme(!isDark);
  }

  return (
    <button
      onClick={toggle}
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
      className="flex items-center gap-2 w-full rounded-md px-3 py-2 text-sm text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
    >
      <span className="w-4 h-4 rounded-full border border-current flex items-center justify-center text-[10px]">
        {isDark ? "◐" : "○"}
      </span>
      {isDark ? "Dark theme" : "Light theme"}
    </button>
  );
}
