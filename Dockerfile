FROM python:3.13-slim

# uv for dependency management (project uses pyproject.toml + uv.lock)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

# Install runtime dependencies only (no dev group) from the lockfile
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY frontend ./frontend

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
