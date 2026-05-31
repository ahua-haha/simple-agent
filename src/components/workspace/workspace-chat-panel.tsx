"use client";

import { Thread } from "@/components/assistant-ui/thread";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import {
  AssistantChatTransport,
  useChatRuntime,
} from "@assistant-ui/react-ai-sdk";
import { MessageSquareIcon, XIcon } from "lucide-react";
import { useMemo, useState } from "react";

export function WorkspaceChatPanel() {
  const [open, setOpen] = useState(false);
  const transport = useMemo(
    () => new AssistantChatTransport({ api: "/api/chat" }),
    [],
  );
  const runtime = useChatRuntime({ transport });

  return (
    <>
      {!open && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={() => setOpen(true)}
              className="absolute left-3 top-1/2 z-10 size-9 -translate-y-1/2 rounded-full bg-background shadow-md"
              aria-label="Open AI chat"
            >
              <MessageSquareIcon className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="right">Open AI chat</TooltipContent>
        </Tooltip>
      )}

      {open && (
        <div
          className="animate-in fade-in-0 slide-in-from-left-8 absolute left-1/2 top-1/2 z-20 flex h-[85%] w-[85%] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-lg border bg-background shadow-2xl duration-200"
          role="dialog"
          aria-label="AI chat panel"
        >
          <header className="flex h-12 shrink-0 items-center justify-between border-b px-3">
            <div className="flex items-center gap-2">
              <MessageSquareIcon className="size-4 text-muted-foreground" />
              <h2 className="text-sm font-medium">AI Chat</h2>
            </div>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setOpen(false)}
              aria-label="Close AI chat"
            >
              <XIcon className="size-4" />
            </Button>
          </header>
          <AssistantRuntimeProvider runtime={runtime}>
            <div className="min-h-0 flex-1">
              <Thread />
            </div>
          </AssistantRuntimeProvider>
        </div>
      )}
    </>
  );
}
