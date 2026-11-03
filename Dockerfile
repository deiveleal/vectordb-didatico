# Uma imagem para tudo: pipeline, busca pela linha de comando e API.
# O que roda e decidido pelo docker-compose.yml (ou pelo comando do docker run).
#
# Dependencias pelo uv.lock (--frozen): exatamente as versoes testadas.
# Instaladas antes de copiar o codigo, para a camada ser reaproveitada
# quando so o codigo muda.
#
# Os modelos (embedding, reranker, tradutor; ~230 MB) NAO vao na imagem:
# sao baixados na primeira execucao para /app/modelos, que o compose monta
# como volume -- a imagem fica pequena e o download acontece uma vez so.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    DIDATICO_DADOS=/app/data \
    DIDATICO_MODELOS=/app/modelos \
    # cache do tokenizador (Hugging Face) junto dos outros modelos
    HF_HOME=/app/modelos/huggingface \
    HOME=/tmp

COPY pyproject.toml uv.lock ./
# --no-default-groups: so as dependencias do projeto (sem dev nem ui).
# --no-cache: sem ele o cache de download do uv (~480 MB) ficaria na imagem
RUN uv sync --frozen --no-default-groups --no-install-project --no-cache

COPY didatico ./didatico

# pastas que viram volume: criadas aqui com permissao para qualquer usuario,
# porque o compose roda com o UID de quem chamou (para os arquivos de data/
# no host serem seus, e nao do root)
RUN mkdir -p /app/data /app/modelos && chmod 777 /app/data /app/modelos

EXPOSE 8000
CMD ["uvicorn", "didatico.api:app", "--host", "0.0.0.0", "--port", "8000"]
