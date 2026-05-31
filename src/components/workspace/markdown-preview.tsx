"use client";

import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

type MarkdownPreviewProps = {
  content: string;
};

export function MarkdownPreview({ content }: MarkdownPreviewProps) {
  return (
    <div className="h-full overflow-auto bg-background p-6 text-sm leading-6">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: (props) => (
            <h1 className="mb-4 text-2xl font-semibold" {...props} />
          ),
          h2: (props) => (
            <h2 className="mt-6 mb-3 text-lg font-semibold" {...props} />
          ),
          p: (props) => <p className="my-3" {...props} />,
          ul: (props) => (
            <ul className="my-3 list-disc space-y-1 pl-5" {...props} />
          ),
          ol: (props) => (
            <ol className="my-3 list-decimal space-y-1 pl-5" {...props} />
          ),
          code: (props) => (
            <code
              className="rounded bg-muted px-1 py-0.5 font-mono text-xs"
              {...props}
            />
          ),
          pre: (props) => (
            <pre
              className="my-4 overflow-auto rounded-lg bg-slate-950 p-4 text-slate-100"
              {...props}
            />
          ),
        }}
      >
        {content}
      </Markdown>
    </div>
  );
}
