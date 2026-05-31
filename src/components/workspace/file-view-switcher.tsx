"use client";

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { WorkspaceViewMode } from "@/lib/workspace-sample-data";

type FileViewSwitcherProps = {
  value: WorkspaceViewMode;
  modes: WorkspaceViewMode[];
  onValueChange: (value: WorkspaceViewMode) => void;
};

export function FileViewSwitcher({
  value,
  modes,
  onValueChange,
}: FileViewSwitcherProps) {
  return (
    <Tabs
      value={value}
      onValueChange={(nextValue) =>
        onValueChange(nextValue as WorkspaceViewMode)
      }
    >
      <TabsList>
        {modes.map((mode) => (
          <TabsTrigger key={mode} value={mode} className="capitalize">
            {getViewModeLabel(mode)}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}

function getViewModeLabel(mode: WorkspaceViewMode) {
  if (mode === "raw") return "Raw";
  if (mode === "preview") return "Preview";
  return "WYSIWYG";
}
