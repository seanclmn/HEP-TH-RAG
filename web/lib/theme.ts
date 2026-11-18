// Kept out of the "use client" toggle so the server layout can inline the string.

/**
 * Runs in <head> before first paint (see app/layout.tsx), so a dark page never flashes white.
 * Resolves the saved preference ("system" = follow the OS) to data-theme on <html>, and
 * re-resolves when the OS setting changes. window.__applyTheme lets the toggle re-run it.
 */
export const themeScript = `(() => {
  const media = matchMedia("(prefers-color-scheme: dark)");
  const apply = () => {
    let pref = null;
    try { pref = localStorage.getItem("theme"); } catch {}
    const dark = pref === "dark" || (pref !== "light" && media.matches);
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  };
  apply();
  media.addEventListener("change", apply);
  window.__applyTheme = apply;
})();`;

export type ThemePref = "system" | "light" | "dark";

declare global {
  interface Window {
    __applyTheme?: () => void;
  }
}

/** The saved choice; "system" if none (or storage is blocked). */
export function loadThemePref(): ThemePref {
  let saved: string | null = null;
  try {
    saved = localStorage.getItem("theme");
  } catch {}
  return saved === "light" || saved === "dark" ? saved : "system";
}

/** Save the choice and apply it now. */
export function saveThemePref(pref: ThemePref) {
  try {
    if (pref === "system") localStorage.removeItem("theme");
    else localStorage.setItem("theme", pref);
  } catch {}
  window.__applyTheme?.();
}
