"""API de busca (FastAPI): a busca.py exposta por HTTP.

So recuperacao: devolve artigos e trechos, nao gera resposta com LLM.

    uv run uvicorn didatico.api:app --port 8000
    curl 'http://localhost:8000/search?pergunta=memória+de+longo+prazo+para+agentes&k=3'
    abra http://localhost:8000/docs para testar no navegador

Os modelos e o Qdrant sao abertos uma vez, quando a API sobe (lifespan), e
reaproveitados em todas as requisicoes. O Qdrant local nao aceita dois
processos na mesma pasta: pare a API antes de rodar o pipeline.
"""

from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from didatico.busca import Buscador

estado: dict[str, Buscador] = {}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    estado["buscador"] = Buscador()
    yield
    estado.pop("buscador").qdrant.close()


app = FastAPI(title="arXiv radar didatico", lifespan=ciclo_de_vida)


@app.get("/health")
def health() -> dict:
    buscador = estado["buscador"]
    pontos = {t: buscador.qdrant.count(t).count for t in buscador.collections(None)}
    return {"status": "ok" if any(pontos.values()) else "indice vazio", "trechos_por_tema": pontos}


@app.get("/search")
def search(pergunta: str = Query(min_length=1),
           tema: Literal["rag", "agents", "llm"] | None = None,
           k: int = Query(5, ge=1, le=10),
           trechos_por_artigo: int = Query(2, ge=1, le=3),
           rerank: bool = False) -> dict:
    return asdict(estado["buscador"].buscar(pergunta, tema, k, trechos_por_artigo, rerank))


@app.get("/artigos/{arxiv_id}")
def artigo(arxiv_id: str, secao: str | None = None,
           max_trechos: int = Query(8, ge=1, le=20)) -> dict:
    if (achado := estado["buscador"].artigo(arxiv_id, secao, max_trechos)) is None:
        raise HTTPException(404, f"artigo {arxiv_id} nao indexado")
    return achado
