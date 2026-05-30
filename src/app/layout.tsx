import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Agent Chat",
  description: "Chat interface powered by pi-agent-core and Vercel AI SDK",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: "system-ui, sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
