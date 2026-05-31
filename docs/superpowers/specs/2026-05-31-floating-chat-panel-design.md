# Floating Chat Panel Design

## Summary

Add a workspace-local floating AI chat panel to `/workspace`. The panel opens from a left-edge handle, overlays only the editor area, and uses the real assistant-ui chat runtime connected to `/api/chat`.

## Goals

- Add a floating AI chat panel to the workspace editor page.
- Keep the file tree and editor layout sizes unchanged when the panel opens.
- Keep the file tree visible while the panel is open.
- Size the panel to 85% width and 85% height of the editor area.
- Center the open panel horizontally and vertically inside the editor area.
- Animate the panel in from the left.
- Close the panel with an `X` button in the panel header.
- Reuse the existing assistant-ui `Thread` connected to `/api/chat`.

## Non-Goals

- No selected-file context is sent to chat in this first version.
- No outside-click-to-close behavior.
- No full-screen modal backdrop.
- No global app-level chat overlay.
- No changes to `/api/chat`, `StreamConverter`, or backend session handling.

## Architecture

The panel is local to `/workspace`.

The existing workspace layout remains:

- left sidebar file tree
- right editor area

The right editor area becomes a relative positioning container:

```tsx
<section className="relative flex min-w-0 flex-1">
  <WorkspaceEditorShell />
  <WorkspaceChatPanel />
</section>
```

`WorkspaceEditorShell` continues to render the selected file editor/viewer. `WorkspaceChatPanel` overlays the editor area without resizing or pushing it.

## Components

Add `WorkspaceChatPanel`.

Responsibilities:

- Own open/closed state.
- Render the left-edge handle.
- Render the floating panel when open.
- Create `AssistantChatTransport({ api: "/api/chat" })`.
- Create `useChatRuntime({ transport })`.
- Wrap panel content in `AssistantRuntimeProvider`.
- Render the existing assistant-ui `Thread`.

The existing full-page chat at `/` remains unchanged.

## Interaction

Opening:

- A narrow handle is positioned on the left edge of the editor area.
- Clicking the handle opens the panel.
- The panel animates in from the left.
- The editor layout does not move.

Closing:

- The panel header contains an `X` icon button.
- Clicking `X` closes the panel.

No outside-click-to-close is included, so editing and chat interactions do not accidentally dismiss the panel.

## Placement And Sizing

When open:

- Panel width is 85% of the editor area.
- Panel height is 85% of the editor area.
- Panel is horizontally centered inside the editor area.
- Panel is vertically centered inside the editor area.
- Panel never covers the file tree sidebar.
- Editor content remains visible around panel edges.

On small editor widths, the panel still uses 85% of the editor area and remains centered inside that editor area.

## UI Details

Use existing shadcn-ui and lucide pieces:

- `Button` for the handle and close action.
- `Tooltip` for the handle.
- `XIcon` for closing.
- `MessageSquareIcon` for the handle.

Panel styling:

- background: `bg-background`
- border: `border`
- shadow: elevated but restrained
- rounded corners consistent with the app
- header with title and close button
- body containing assistant-ui `Thread`

The panel is not modal and does not dim the editor.

## Data Flow

- `WorkspaceChatPanel` owns only visibility state.
- assistant-ui owns chat message/runtime state.
- `/api/chat` remains the chat transport endpoint.
- Workspace file editor state remains separate from chat state.

## Testing And Verification

Verification should include:

- `npm run build`
- Browser check at `/workspace`
- Confirm handle appears on the editor area's left edge.
- Confirm clicking the handle opens the panel.
- Confirm the open panel is centered inside the editor area.
- Confirm the open panel is approximately 85% of editor width and height.
- Confirm the editor layout does not resize when the panel opens.
- Confirm the file tree remains visible while the panel is open.
- Confirm `X` closes the panel.
- Confirm chat input renders inside the panel.
- Browser check at `/` to confirm the full-page assistant chat still renders.
