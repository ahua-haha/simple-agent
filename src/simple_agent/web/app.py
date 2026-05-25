"""FastAPI app for browsing the task tree."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

from simple_agent.db.db import Database
from simple_agent.state.state import Task

TEMPLATES_DIR = Path(__file__).parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))


def render(template_name: str, **kwargs) -> str:
    template = jinja_env.get_template(template_name)
    return template.render(**kwargs)


_db: Database | None = None


def get_db() -> Database:
    if _db is None:
        raise RuntimeError("Database not initialized.")
    return _db


def create_app(db_path: str) -> FastAPI:
    global _db
    _db = Database(db_path)

    app = FastAPI(title="Simple Agent Web")

    @app.get("/", response_class=HTMLResponse)
    async def task_tree(request: Request):
        rows = get_db().load_all_tasks()
        tasks = Task.from_db_rows(rows) if rows else {}
        root = None
        for task in tasks.values():
            if task.parent_id is None:
                root = task
                break
        html = render("task_tree.html", root=root, tasks=tasks)
        return HTMLResponse(content=html)

    @app.get("/task/{task_id}", response_class=HTMLResponse)
    async def task_detail(request: Request, task_id: int):
        rows = get_db().load_all_tasks()
        all_tasks = Task.from_db_rows(rows) if rows else {}
        task = all_tasks.get(task_id)
        if not task:
            return HTMLResponse(content="<h1>Task not found</h1>", status_code=404)
        html = render("task_detail.html", task=task, tasks=all_tasks)
        return HTMLResponse(content=html)

    return app
