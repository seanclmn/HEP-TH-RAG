import type { Metadata } from "next";
import { themeScript } from "@/lib/theme";
import "./globals.css";

export const metadata: Metadata = { title: "HEP-TH RAG" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // The theme script sets data-theme before React hydrates, hence suppressHydrationWarning.
    <html lang="en" data-theme="light" suppressHydrationWarning>
      <body>
        {/* First in <body>, not <head>: extensions inject scripts into <head>, which breaks
            hydration of anything React renders there. Still runs before the page paints. */}
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        {children}
      </body>
    </html>
  );
}
