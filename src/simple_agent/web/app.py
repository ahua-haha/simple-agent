"""FastAPI app for browsing task messages."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

from simple_agent.db.db import Database

TEMPLATES_DIR = Path(__file__).parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))


def render(template_name: str, **kwargs) -> str:
    template = jinja_env.get_template(template_name)
    return template.render(**kwargs)


db: Database | None = None


def get_db() -> Database:
    if db is None:
        raise RuntimeError("Database not initialized. Call init(db_path) first.")
    return db


def create_app(db_path: str) -> FastAPI:
    global db
    db = Database(db_path)

    app = FastAPI(title="Simple Agent Web")

    @app.get("/", response_class=HTMLResponse)
    async def task_list(request: Request, limit: int = 50):
        database = get_db()
        tasks = database.list_tasks(limit=limit)
        html = render("task_list.html", tasks=tasks)
        return HTMLResponse(content=html)

    @app.get("/task/{task_id}", response_class=HTMLResponse)
    async def task_detail(request: Request, task_id: int):
        database = get_db()
        task = database.get_task(task_id)
        if not task:
            return HTMLResponse(content="<h1>Task not found</h1>", status_code=404)
        html = render("task_detail.html", task=task)
        return HTMLResponse(content=html)

    @app.get("/api/tasks")
    async def api_task_list(limit: int = 50):
        database = get_db()
        return database.list_tasks(limit=limit)

    @app.get("/api/task/{task_id}")
    async def api_task_detail(task_id: int):
        database = get_db()
        task = database.get_task(task_id)
        if not task:
            return {"error": "not found"}
        return task

    return app
