"""Etapa 7 -- indice: chunks -> vetores -> Qdrant.

Dois papeis que no original sao servicos Docker, aqui sao bibliotecas:

- embeddings: fastembed roda o bge-small-en-v1.5 em ONNX, na CPU, dentro do
  proprio processo. O modelo (~70 MB) e baixado uma vez para modelos/;
- banco vetorial: qdrant-client em MODO LOCAL -- QdrantClient(path=...) grava
  o indice numa pasta, sem servidor. Mesma API do Qdrant de verdade; trocar
  para servidor e trocar path= por url= (e o que QDRANT_URL faz, ver
  config.py). Limite do modo local: um processo por vez abre a pasta (nao
  rode a API e o pipeline ao mesmo tempo).

Uma collection por tema (rag, agents, llm), cosseno, 384 dimensoes. O id de
cada ponto e UUID5(chunk_id): o mesmo chunk sempre cai no mesmo id, entao
indexar de novo substitui em vez de duplicar.

Para saber o que ja foi indexado, guarda o hash de cada arquivo de chunks em
data/indice_estado.json: arquivo igual e pulado; arquivo mudou (chunker
novo, texto extraido de novo), os pontos antigos do artigo saem e os novos
entram -- um artigo que passou de 20 para 18 chunks nao deixa 2 orfaos.

Uso: python -m didatico.etapa7_indice
"""

import hashlib
import json
import logging
import uuid
from collections import Counter

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from didatico import config
from didatico.util import configurar_log, em_lotes, ler_jsonl

logger = logging.getLogger(__name__)

# Um estado por destino: o que esta no indice local nao esta no servidor.
ESTADO = config.DADOS / ("indice_estado_servidor.json" if config.QDRANT_URL
                         else "indice_estado.json")
NAMESPACE_PONTOS = uuid.uuid5(uuid.NAMESPACE_URL, "arxiv-chunks")
LOTE = 64


def id_do_ponto(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE_PONTOS, chunk_id))


def abrir_qdrant() -> QdrantClient:
    """Servidor, se QDRANT_URL estiver definido; senao, a pasta data/qdrant.
    O resto do codigo nao sabe a diferenca: a API do cliente e a mesma."""
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    config.QDRANT.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.QDRANT))


def carregar_modelo_embedding() -> TextEmbedding:
    return TextEmbedding(config.MODELO_EMBEDDING, cache_dir=str(config.MODELOS / "fastembed"))


def garantir_collections(qdrant: QdrantClient) -> None:
    for tema in config.TEMAS:
        if not qdrant.collection_exists(tema):
            qdrant.create_collection(tema, vectors_config=models.VectorParams(
                size=config.DIMENSAO, distance=models.Distance.COSINE))


def _hash(caminho) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def _indice_vazio(qdrant: QdrantClient) -> bool:
    return all(not qdrant.collection_exists(t) or qdrant.count(t).count == 0
               for t in config.TEMAS)


def run(qdrant: QdrantClient | None = None) -> dict[str, int]:
    fechar = qdrant is None
    qdrant = qdrant or abrir_qdrant()
    try:
        estado: dict[str, str] = json.loads(ESTADO.read_text()) if ESTADO.exists() else {}
        if estado and _indice_vazio(qdrant):
            # o estado diz "ja indexado", mas o indice foi apagado (ex.: docker
            # compose down -v): confiar no estado deixaria a busca vazia
            logger.warning("Indice vazio apesar do estado salvo -- reindexando tudo")
            estado = {}
        arquivos = sorted(config.CHUNKS.glob("*.jsonl"))
        pendentes = [(a, h) for a in arquivos if estado.get(a.stem) != (h := _hash(a))]
        contagem: Counter[str] = Counter(ja_indexados=len(arquivos) - len(pendentes))
        if not pendentes:
            logger.info("Indice: nada novo para indexar (%d artigos ja no indice)", len(estado))
            return dict(contagem)

        garantir_collections(qdrant)
        modelo = carregar_modelo_embedding()
        for arquivo, h in pendentes:
            chunks = ler_jsonl(arquivo)
            arxiv_id = arquivo.stem
            # sai o que havia do artigo -- em qualquer tema, pois o tema pode ter mudado
            filtro = models.Filter(must=[models.FieldCondition(
                key="arxiv_id", match=models.MatchValue(value=arxiv_id))])
            for tema in config.TEMAS:
                qdrant.delete(tema, points_selector=models.FilterSelector(filter=filtro))

            for lote in em_lotes(chunks, LOTE):
                vetores = list(modelo.embed([c["texto_embedding"] for c in lote]))
                qdrant.upsert(lote[0]["tema"], points=[
                    models.PointStruct(
                        id=id_do_ponto(c["chunk_id"]),
                        vector=v.tolist(),
                        # o payload leva o que a busca mostra: nada de ir buscar
                        # o texto em outro lugar depois
                        payload={k: c[k] for k in ("chunk_id", "arxiv_id", "tema", "titulo",
                                                   "url", "data_v1", "secao", "ordem", "texto")},
                    ) for c, v in zip(lote, vetores)])
            estado[arxiv_id] = h
            ESTADO.write_text(json.dumps(estado, indent=1))   # a cada artigo: queda nao perde tudo
            contagem["artigos"] += 1
            contagem["chunks"] += len(chunks)
    finally:
        if fechar:
            qdrant.close()

    logger.info("Indice: %s", dict(contagem))
    return dict(contagem)


if __name__ == "__main__":
    configurar_log()
    run()
