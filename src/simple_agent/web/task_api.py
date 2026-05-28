"""Task tree rendering API — HTML endpoints for browsing the task tree."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

from simple_agent.state.state import Task

TEMPLATES_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))


def _render(template_name: str, **kwargs) -> str:
    template = _jinja_env.get_template(template_name)
    return template.render(**kwargs)


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def task_tree(request: Request):
    get_db = request.app.state.get_db
    rows = get_db().load_all_tasks()
    tasks = Task.from_db_rows(rows) if rows else {}
    root = None
    for task in tasks.values():
        if task.parent_id is None:
            root = task
            break
    html = _render("task_tree.html", root=root, tasks=tasks)
    return HTMLResponse(content=html)


@router.get("/task/{task_id}", response_class=HTMLResponse)
async def task_detail(request: Request, task_id: int):
    get_db = request.app.state.get_db
    rows = get_db().load_all_tasks()
    all_tasks = Task.from_db_rows(rows) if rows else {}
    task = all_tasks.get(task_id)
    if not task:
        return HTMLResponse(content="<h1>Task not found</h1>", status_code=404)
    html = _render("task_detail.html", task=task, tasks=all_tasks)
    return HTMLResponse(content=html)
