"use client";

type RawFileEditorProps = {
  value: string;
  onChange: (value: string) => void;
};

export function RawFileEditor({ value, onChange }: RawFileEditorProps) {
  return (
    <textarea
      value={value}
      onChange={(event) => onChange(event.target.value)}
      spellCheck={false}
      className="h-full min-h-0 w-full resize-none border-0 bg-white p-4 font-mono text-sm leading-6 text-slate-950 outline-none"
      aria-label="Raw file content"
    />
  );
}
