from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.api.routes import router

app = FastAPI(title="Energy Forecasting API")

# Include API routes
app.include_router(router)

# Serve dashboard
BASE_DIR = Path(__file__).resolve().parent
templates_path = BASE_DIR / "templates"

app.mount("/static", StaticFiles(directory=templates_path), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    with open(templates_path / "dashboard.html", "r") as f:
        return f.read()
