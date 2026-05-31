"use client";

import { Crepe, CrepeFeature } from "@milkdown/crepe";
import { useEffect, useRef } from "react";

type MarkdownWysiwygEditorProps = {
  value: string;
  onChange: (value: string) => void;
};

export function MarkdownWysiwygEditor({
  value,
  onChange,
}: MarkdownWysiwygEditorProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const crepeRef = useRef<Crepe | null>(null);
  const onChangeRef = useRef(onChange);
  const ignoreFirstUpdateRef = useRef(true);

  useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;

    const crepe = new Crepe({
      root,
      defaultValue: value,
      features: {
        [CrepeFeature.CodeMirror]: true,
        [CrepeFeature.ListItem]: false,
        [CrepeFeature.LinkTooltip]: false,
        [CrepeFeature.Cursor]: true,
        [CrepeFeature.ImageBlock]: false,
        [CrepeFeature.BlockEdit]: false,
        [CrepeFeature.Toolbar]: false,
        [CrepeFeature.Placeholder]: true,
        [CrepeFeature.Table]: true,
        [CrepeFeature.Latex]: false,
        [CrepeFeature.TopBar]: false,
        [CrepeFeature.AI]: false,
      },
      featureConfigs: {
        [CrepeFeature.Placeholder]: {
          text: "Start writing markdown...",
        },
      },
    });

    crepe.on((api) => {
      api.markdownUpdated((_ctx, markdown) => {
        if (ignoreFirstUpdateRef.current) {
          ignoreFirstUpdateRef.current = false;
          return;
        }
        onChangeRef.current(markdown);
      });
    });

    crepeRef.current = crepe;
    void crepe.create();

    return () => {
      crepeRef.current = null;
      void crepe.destroy();
    };
  }, []);

  return (
    <div className="workspace-wysiwyg h-full overflow-auto bg-background">
      <div ref={rootRef} className="h-full" />
    </div>
  );
}
