# Imagem da UI, separada da API: so Streamlit + requests, nada de modelos nem
# de didatico/. E o lado "cliente" do desenho -- conversa com a API por HTTP.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HOME=/tmp

COPY pyproject.toml uv.lock ./
# so o grupo "ui" do pyproject: a imagem nao leva fastembed, qdrant, etc.
RUN uv sync --frozen --only-group ui --no-install-project --no-cache

COPY .streamlit ./.streamlit
COPY ui/app.py ./ui/app.py

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.port", "8501", \
     "--server.address", "0.0.0.0", "--server.headless", "true"]
