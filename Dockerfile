FROM python:3.12-slim

WORKDIR /app

# Install uv first (cached layer)
RUN pip install uv

# Copy only dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies (cached unless pyproject.toml/uv.lock changes)
RUN uv sync --no-dev --frozen

# Copy application code (changes more frequently)
COPY app ./app
COPY solution.py ./

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
