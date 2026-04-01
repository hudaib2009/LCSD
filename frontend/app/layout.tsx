import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CSD Clinical Dashboard",
  description: "Clinical decision support dashboard for imaging inference."
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-[var(--font-plex)]">
        <div className="relative overflow-hidden">
          <div className="pointer-events-none absolute inset-0 bg-radial-glow opacity-70" />
          <div className="pointer-events-none absolute right-0 top-12 h-64 w-64 rounded-full bg-gradient-to-br from-sea/20 to-transparent blur-3xl" />
          <main className="relative z-10">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
