# Use Python 3.11 slim image
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies with uv
RUN uv sync --frozen

# Copy application code
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV UV_SYSTEM_PYTHON=1

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["uv", "run", "streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]