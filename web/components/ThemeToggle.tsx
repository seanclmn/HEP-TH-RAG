"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/cn";
import { type ThemePref, loadThemePref, saveThemePref } from "@/lib/theme";

const options: { value: ThemePref; label: string; icon: React.ReactNode }[] = [
  {
    value: "system",
    label: "System",
    icon: <path d="M3 4.5h14v9H3zM7 16.5h6M10 13.5v3" />,
  },
  {
    value: "light",
    label: "Light",
    icon: (
      <>
        <circle cx="10" cy="10" r="3.2" />
        <path d="M10 2.5v2M10 15.5v2M2.5 10h2M15.5 10h2M4.7 4.7l1.4 1.4M13.9 13.9l1.4 1.4M4.7 15.3l1.4-1.4M13.9 6.1l1.4-1.4" />
      </>
    ),
  },
  {
    value: "dark",
    label: "Dark",
    icon: <path d="M16 12.2A6.5 6.5 0 0 1 7.8 4a6.5 6.5 0 1 0 8.2 8.2z" />,
  },
];

export default function ThemeToggle({ className }: { className?: string }) {
  // "system" on the server and the first client render; the real value is read after mount.
  const [pref, setPref] = useState<ThemePref>("system");

  useEffect(() => setPref(loadThemePref()), []);

  function choose(value: ThemePref) {
    setPref(value);
    saveThemePref(value);
  }

  return (
    <div role="radiogroup" aria-label="Theme" className={cn("tabs tabs-box tabs-xs flex-nowrap", className)}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={pref === o.value}
          title={o.label}
          onClick={() => choose(o.value)}
          className={cn("tab flex-1 gap-1.5", pref === o.value && "tab-active")}
        >
          <svg
            viewBox="0 0 20 20"
            className="size-3.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            {o.icon}
          </svg>
          {o.label}
        </button>
      ))}
    </div>
  );
}
