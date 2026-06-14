"use client";

import { Crepe } from "@milkdown/crepe";
import { useEffect, useRef, useState } from "react";

const initialMarkdown = `# Crepe Markdown Test

This page mounts the Milkdown Crepe editor by itself.

## Formatting

- Bold, italic, inline code, and links should work from the toolbar.
- Lists should render as editable markdown blocks.
- The slash/block menu and toolbar icons should render normally.

> Use this isolated page to check whether the editor itself or the workspace shell is causing a rendering issue.

\`\`\`ts
const editor = new Crepe({
  root,
  defaultValue: markdown,
});
\`\`\`
`;

export default function MarkdownWysiwygTestPage() {
  const editorRootRef = useRef<HTMLDivElement>(null);
  const [markdown, setMarkdown] = useState(initialMarkdown);

  useEffect(() => {
    const root = editorRootRef.current;
    if (!root) return;

    const crepe = new Crepe({
      root,
      defaultValue: initialMarkdown,
    });

    crepe.on((api) => {
      api.markdownUpdated((_ctx, nextMarkdown) => {
        setMarkdown(nextMarkdown);
      });
    });

    const createPromise = crepe.create();

    return () => {
      void createPromise.then(() => crepe.destroy()).catch(() => {});
    };
  }, []);

  return (
    <main className="min-h-screen bg-zinc-50 p-6 text-zinc-950">
      <div className="mx-auto flex max-w-6xl flex-col gap-4">
        <header className="flex items-center justify-between border-b border-zinc-200 pb-3">
          <div>
            <h1 className="text-lg font-semibold">Markdown WYSIWYG Test</h1>
            <p className="text-sm text-zinc-500">
              Isolated Milkdown Crepe render surface
            </p>
          </div>
          <a
            href="/workspace"
            className="rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-sm font-medium hover:bg-zinc-100"
          >
            Workspace
          </a>
        </header>

        <section className="grid min-h-[720px] grid-cols-[minmax(0,1fr)_360px] gap-4">
          <div className="overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-sm">
            <div ref={editorRootRef} className="min-h-[720px]" />
          </div>

          <aside className="overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-sm">
            <div className="border-b border-zinc-200 px-4 py-3 text-sm font-medium">
              Markdown Output
            </div>
            <pre className="h-[677px] overflow-auto p-4 text-xs leading-6 text-zinc-700">
              {markdown}
            </pre>
          </aside>
        </section>
      </div>
    </main>
  );
}
