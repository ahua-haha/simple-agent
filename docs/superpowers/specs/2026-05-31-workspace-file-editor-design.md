# Workspace File Editor Design

## Summary

Build a mock workspace file viewer/editor page using shadcn-ui. The page uses a classic split layout: a file tree sidebar on the left and a file editor/viewer area on the right. It uses sample in-memory files only; no filesystem API, persistence API, or save-to-disk behavior is included in this change.

## Goals

- Add a workspace editor page at `/workspace`.
- Show a left sidebar file tree with sample folders and files.
- Show a right editor/viewer area for the selected sample file.
- Support raw content viewing and editing for every sample file.
- Support rendered markdown preview for markdown files.
- Use shadcn-ui components and lucide icons.
- Keep the implementation componentized so a later real filesystem-backed version can replace the mock data layer cleanly.

## Non-Goals

- No real workspace file reads or writes.
- No save API.
- No multi-file tabs.
- No search, rename, delete, create-file, drag/drop, or upload flows.
- No full WYSIWYG markdown editor in this first version.
- No binary file preview.

## Route And Layout

The current assistant chat page remains available. The workspace file editor is added as a separate route at `/workspace`.

The page uses a classic split layout:

- Left sidebar: fixed-width file tree.
- Right pane: flexible editor/viewer area.

The first screen is the workspace tool itself, not a landing page.

## Component Boundaries

The implementation should use small components with clear ownership:

- `WorkspacePage`: owns sample file data, selected-file state, active-view state, and edited content state.
- `FileTree`: renders folders/files and handles file selection.
- `WorkspaceEditorShell`: renders the right-side header, file metadata, view controls, and content area.
- `FileViewSwitcher`: displays available views for the selected file and changes active view.
- `RawFileEditor`: renders an editable raw content surface.
- `MarkdownPreview`: renders markdown content as read-only preview.

These boundaries keep the mock version simple while leaving an obvious future path for real file loading and saving.

## Data Model

Use typed sample data with folders and files. A representative shape:

```ts
type WorkspaceFile = {
  id: string;
  name: string;
  path: string;
  language: "typescript" | "markdown" | "json" | "text";
  content: string;
};

type WorkspaceFolder = {
  id: string;
  name: string;
  children: Array<WorkspaceFolder | WorkspaceFile>;
};
```

The page uses a small helper to flatten the sample tree into files for selection lookup. Edited content is held in browser state keyed by file id.

Dirty state is computed by comparing the current in-memory content to the original sample content for the selected file.

## View Modes

Every file supports `Raw`.

Markdown files also support `Preview`.

When switching between files, if the current active view is not supported by the newly selected file, the page falls back to `Raw`.

`Raw` is editable. `Preview` is read-only.

## UI Details

The UI should feel like a compact workspace tool.

Left sidebar:

- workspace title/header
- folder and file icons
- nested folder/file indentation
- active file highlight
- compact spacing

Right editor area:

- top bar with file name, path, file type, and dirty state
- view switcher for `Raw` and `Preview`
- main content area
- raw mode uses a monospaced editable surface
- markdown preview renders formatted markdown

Use these shadcn-ui components:

- `Button`
- `Tooltip`
- `ScrollArea`
- `Separator`
- `Tabs` or `ToggleGroup`
- `Badge`

Use lucide icons for folders, files, markdown, code, and modified state indicators.

## Error And Empty States

- If no file is selected, show an empty editor state.
- If the selected file id is missing from mock data, fall back to the first available file.
- If a file is empty, show an empty editable raw area.
- If preview is requested for a non-markdown file, switch to `Raw`.
- Markdown preview should tolerate ordinary markdown input without crashing the page.

## Testing And Verification

Verification should include:

- `npm run build`
- Browser smoke test at `/workspace`
- Confirm the file tree renders.
- Confirm clicking files changes the selected file and editor content.
- Confirm raw editing updates in-memory content.
- Confirm dirty state appears after editing.
- Confirm markdown files expose `Raw` and `Preview`.
- Confirm non-markdown files expose only `Raw`.
- Confirm the existing assistant chat page remains available.

## Future Extensions

The design intentionally leaves room for later work:

- real filesystem-backed file tree
- read/write API routes
- save/revert actions
- multi-file tabs
- full WYSIWYG markdown editing
- JSON/table views
- image or binary preview modes
