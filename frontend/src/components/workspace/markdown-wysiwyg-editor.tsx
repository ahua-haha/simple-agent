"use client";

import { Crepe } from "@milkdown/crepe";
import { Milkdown, MilkdownProvider, useEditor } from "@milkdown/react";
import { useEffect, useRef } from "react";

type MarkdownWysiwygEditorProps = {
  value: string;
  onChange: (value: string) => void;
};

export function MarkdownWysiwygEditor({
  value,
  onChange,
}: MarkdownWysiwygEditorProps) {
  return (
    <div className="workspace-wysiwyg h-full overflow-auto bg-background">
      <MilkdownProvider>
        <CrepeMarkdownEditor value={value} onChange={onChange} />
      </MilkdownProvider>
    </div>
  );
}

function CrepeMarkdownEditor({ value, onChange }: MarkdownWysiwygEditorProps) {
  const onChangeRef = useRef(onChange);
  const initialValueRef = useRef(value);
  const ignoreFirstUpdateRef = useRef(true);

  useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  useEditor((root) => {
    const crepe = new Crepe({
      root,
      defaultValue: initialValueRef.current,
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

    return crepe;
  }, []);

  return <Milkdown />;
}
