"use client";

import { Crepe } from "@milkdown/crepe";
import { Milkdown, MilkdownProvider, useEditor } from "@milkdown/react";

function CrepeEditor() {
  useEditor((root) => {
    return new Crepe({
      root,
      defaultValue: "# Hello Milkdown\n\nStart editing with **Crepe**.",
    });
  });

  return <Milkdown />;
}

export default function CrepeDemoPage() {
  return (
    <MilkdownProvider>
      <CrepeEditor />
    </MilkdownProvider>
  );
}
