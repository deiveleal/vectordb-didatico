"""API (FastAPI): busca semantica e insercao de artigos.

E a unica porta de entrada da UI (ui/app.py), como no desenho de arquitetura:
a UI nao importa nada de src/, so faz HTTP.

Busca (so recuperacao: devolve artigos e trechos, nao gera resposta com LLM)
    GET  /search?pergunta=...&tema=rag&k=5&trechos_por_artigo=2&rerank=true&traduzir=true
    GET  /artigos/{arxiv_id}?secao=result&max_trechos=8   trechos na ordem do texto
    GET  /artigos/{arxiv_id}/documento                     o HTML/PDF guardado
    GET  /health

Insercao (assincrona: responde na hora com um id; ver ingestao.py)
    POST /ingestoes/link   {"link": "https://arxiv.org/abs/2609.28614", "tema": null}
    POST /ingestoes/pdf    multipart: arquivo, titulo (opcional), tema (opcional)
    GET  /ingestoes/{id}   status: na_fila -> rodando -> concluida | falhou
    GET  /ingestoes        as mais recentes

Rodando:
    uv run uvicorn src.api:app --port 8000
    abra http://localhost:8000/docs para testar no navegador

Os modelos e o Qdrant sao abertos uma vez, quando a API sobe (lifespan), e
reaproveitados em todas as requisicoes. No modo local (sem QDRANT_URL), so
este processo pode abrir o indice: pare a API antes de rodar o pipeline pela
linha de comando, ou insira artigos pela propria API.
"""

from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src import config
from src.busca import Buscador
from src.etapa3_gold import ARQUIVO as ARQUIVO_GOLD
from src.etapa4_documentos import localizar_documento
from src.ingestao import Ingestoes
from src.util import ler_jsonl

Tema = Literal["rag", "agents", "llm"]
MAX_BYTES_PDF = 50 * 1024 * 1024

estado: dict = {}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    buscador = Buscador()
    estado["buscador"] = buscador
    estado["ingestoes"] = Ingestoes(buscador)
    yield
    buscador.qdrant.close()


app = FastAPI(title="arXiv radar didatico", lifespan=ciclo_de_vida)


def _buscador() -> Buscador:
    return estado["buscador"]


def _ingestoes() -> Ingestoes:
    return estado["ingestoes"]


# --- Busca ---------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    buscador = _buscador()
    with buscador.trava:
        pontos = {t: buscador.qdrant.count(t).count for t in buscador.collections(None)}
    return {
        "status": "ok" if any(pontos.values()) else "indice vazio",
        "qdrant": config.QDRANT_URL or "local",
        "trechos_por_tema": pontos,
        "ingestoes_na_fila": _ingestoes().fila.qsize(),
    }


@app.get("/search")
def search(pergunta: str = Query(min_length=1),
           tema: Tema | None = None,
           k: int = Query(5, ge=1, le=10),
           trechos_por_artigo: int = Query(2, ge=1, le=3),
           rerank: bool = False,
           traduzir: bool = True) -> dict:
    buscador = _buscador()
    with buscador.trava:
        resposta = buscador.buscar(pergunta, tema, k, trechos_por_artigo, rerank, traduzir)
    return asdict(resposta)


@app.get("/artigos/{arxiv_id}")
def artigo(arxiv_id: str, secao: str | None = None,
           max_trechos: int = Query(8, ge=1, le=20)) -> dict:
    buscador = _buscador()
    with buscador.trava:
        achado = buscador.artigo(arxiv_id, secao, max_trechos)
    if achado is None:
        raise HTTPException(404, f"artigo {arxiv_id} nao indexado")
    return achado


@app.get("/artigos/{arxiv_id}/documento")
def documento(arxiv_id: str) -> FileResponse:
    """O documento guardado na etapa 4 -- e o "link" de um PDF enviado que nao
    e do arXiv (nao ha pagina dele no arXiv para apontar)."""
    artigo = next((a for a in ler_jsonl(ARQUIVO_GOLD) if a["arxiv_id"] == arxiv_id), None)
    caminho = localizar_documento(artigo) if artigo else None
    if caminho is None:
        raise HTTPException(404, f"sem documento guardado para {arxiv_id}")
    tipo = "application/pdf" if caminho.suffix == ".pdf" else "text/html"
    return FileResponse(caminho, media_type=tipo, filename=caminho.name,
                        content_disposition_type="inline")


# --- Insercao ------------------------------------------------------------------

class PedidoLink(BaseModel):
    link: str
    tema: Tema | None = None


@app.post("/ingestoes/link", status_code=202)
def inserir_link(pedido: PedidoLink) -> dict:
    try:
        return _ingestoes().link(pedido.link, pedido.tema).publica()
    except ValueError as erro:
        raise HTTPException(422, str(erro))


@app.post("/ingestoes/pdf", status_code=202)
async def inserir_pdf(arquivo: UploadFile = File(...),
                      titulo: str | None = Form(None),
                      tema: Tema | None = Form(None)) -> dict:
    conteudo = await arquivo.read(MAX_BYTES_PDF + 1)
    if len(conteudo) > MAX_BYTES_PDF:
        raise HTTPException(413, f"PDF maior que {MAX_BYTES_PDF // 1024 // 1024} MB")
    try:
        tarefa = _ingestoes().pdf(conteudo, arquivo.filename or "artigo.pdf", titulo, tema)
    except ValueError as erro:
        raise HTTPException(422, str(erro))
    return tarefa.publica()


@app.get("/ingestoes/{tarefa_id}")
def ingestao(tarefa_id: str) -> dict:
    if (tarefa := _ingestoes().tarefas.get(tarefa_id)) is None:
        raise HTTPException(404, f"insercao {tarefa_id} nao encontrada (a API reiniciou?)")
    return tarefa.publica()


@app.get("/ingestoes")
def ingestoes(limite: int = Query(20, ge=1, le=100)) -> list[dict]:
    tarefas = list(_ingestoes().tarefas.values())[-limite:]
    return [t.publica() for t in reversed(tarefas)]
