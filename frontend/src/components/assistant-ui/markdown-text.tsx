"use client";

import { StreamdownTextPrimitive } from "@assistant-ui/react-streamdown";
import { code } from "@streamdown/code";
import { mermaid } from "@streamdown/mermaid";
import { memo } from "react";

const MarkdownTextImpl = () => {
  return (
    <StreamdownTextPrimitive
      className="aui-md"
      plugins={{ code, mermaid }}
      shikiTheme={["github-light", "github-dark"]}
      controls={{
        code: true,
        mermaid: {
          copy: true,
          download: true,
          fullscreen: true,
          panZoom: true,
        },
      }}
    />
  );
};

export const MarkdownText = memo(MarkdownTextImpl);
