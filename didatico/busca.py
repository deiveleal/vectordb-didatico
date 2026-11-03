"""Busca semantica: pergunta -> artigos com os trechos que mais casam.

O caminho de uma pergunta (a metade de baixo do desenho):

1. traducao: pergunta em portugues vira ingles (traducao.py);
2. vetor: a pergunta ganha o prefixo de consulta do bge e passa pelo MESMO
   modelo que vetorizou os trechos -- vetores de modelos diferentes nao se
   comparam;
3. Qdrant: busca por cosseno na collection do tema (ou nas tres), AGRUPANDO
   por arxiv_id -- sem isso, o top-5 podia ser 5 trechos do mesmo artigo;
4. opcional, rerank: o Qdrant traz os melhores trechos candidatos, um
   cross-encoder le pergunta + trecho JUNTOS e da a relevancia de cada par, e
   os trechos sao agrupados por artigo nessa nova ordem. Mais preciso, mais
   lento (o modelo roda uma vez por candidato).

O resultado e por ARTIGO: titulo, link, tema, score do melhor trecho e os
melhores trechos dele, cada um com a secao de onde veio.

Uso:
    python -m didatico.busca "memória de longo prazo para agentes"
    python -m didatico.busca "reward hacking in research agents" --tema agents -k 3 --rerank
"""

import argparse
import contextlib
import logging
import math
import threading
from dataclasses import asdict, dataclass, field

from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from didatico import config
from didatico.etapa7_indice import abrir_qdrant, carregar_modelo_embedding
from didatico.traducao import Tradutor
from didatico.util import configurar_log

logger = logging.getLogger(__name__)


@dataclass
class Trecho:
    secao: str
    texto: str
    score: float
    ordem: int


@dataclass
class Resultado:
    arxiv_id: str
    titulo: str
    tema: str
    url: str
    data_v1: str
    score: float                  # do melhor trecho (do reranker, se houve)
    trechos: list[Trecho] = field(default_factory=list)


@dataclass
class RespostaBusca:
    pergunta: str                 # como veio
    pergunta_busca: str           # o que virou vetor
    idioma: str
    traduzida: bool
    reranqueada: bool
    resultados: list[Resultado]


def _dados(ponto) -> dict:
    """O payload do ponto (o qdrant-client o tipa como opcional)."""
    return ponto.payload or {}


def _trecho(ponto, score: float) -> Trecho:
    dados = _dados(ponto)
    return Trecho(dados["secao"], dados["texto"], score, dados["ordem"])


def _resultado(payload: dict, score: float, trechos: list[Trecho]) -> Resultado:
    return Resultado(arxiv_id=payload["arxiv_id"], titulo=payload["titulo"],
                     tema=payload["tema"], url=payload["url"], data_v1=payload["data_v1"] or "",
                     score=score, trechos=trechos)


class Buscador:
    """Carrega os modelos uma vez e responde quantas perguntas vierem."""

    def __init__(self, qdrant: QdrantClient | None = None):
        self.qdrant = qdrant or abrir_qdrant()
        self.embedding = carregar_modelo_embedding()
        self.tradutor = Tradutor()
        self._reranker: TextCrossEncoder | None = None
        # No modo local o cliente do Qdrant e uma pasta aberta por este
        # processo: busca (leitura) e insercao (escrita) se revezam por esta
        # trava. O servidor Qdrant ja lida com acesso concorrente sozinho.
        self.trava: contextlib.AbstractContextManager = (
            contextlib.nullcontext() if config.QDRANT_URL else threading.Lock())

    @property
    def reranker(self) -> TextCrossEncoder:
        if self._reranker is None:      # so carrega quem pedir rerank
            self._reranker = TextCrossEncoder(
                config.MODELO_RERANKER, cache_dir=str(config.MODELOS / "fastembed"))
        return self._reranker

    def collections(self, tema: str | None) -> list[str]:
        if tema is not None and tema not in config.TEMAS:
            raise ValueError(f"tema {tema!r} invalido; use um de {config.TEMAS}")
        temas = [tema] if tema else list(config.TEMAS)
        return [t for t in temas if self.qdrant.collection_exists(t)]

    def buscar(self, pergunta: str, tema: str | None = None, k: int = 5,
               trechos_por_artigo: int = 2, rerank: bool = False,
               traduzir: bool = True) -> RespostaBusca:
        pergunta = pergunta.strip()
        if not pergunta:
            raise ValueError("pergunta vazia")
        k = max(1, min(k, 10))
        trechos_por_artigo = max(1, min(trechos_por_artigo, 3))
        collections = self.collections(tema)

        if traduzir:
            pergunta_busca, idioma, traduzida = self.tradutor.preparar_pergunta(pergunta)
        else:       # para comparar: a pergunta vira vetor do jeito que veio
            pergunta_busca, idioma, traduzida = pergunta, "?", False
        vetor = list(self.embedding.embed([config.PREFIXO_CONSULTA + pergunta_busca]))[0].tolist()

        if rerank:
            resultados = self._com_rerank(pergunta_busca, vetor, collections, k, trechos_por_artigo)
        else:
            resultados = self._por_vetor(vetor, collections, k, trechos_por_artigo)
        return RespostaBusca(pergunta, pergunta_busca, idioma, traduzida, rerank, resultados)

    def _por_vetor(self, vetor, collections, k, trechos_por_artigo) -> list[Resultado]:
        resultados = []
        for collection in collections:
            grupos = self.qdrant.query_points_groups(
                collection, query=vetor, group_by="arxiv_id", limit=k,
                group_size=trechos_por_artigo, with_payload=True).groups
            for grupo in grupos:
                hits = grupo.hits
                resultados.append(_resultado(_dados(hits[0]), hits[0].score,
                                             [_trecho(h, h.score) for h in hits]))
        # com mais de um tema, junta pelo score: comparavel entre collections
        # porque modelo e distancia sao os mesmos
        return sorted(resultados, key=lambda r: r.score, reverse=True)[:k]

    def _com_rerank(self, pergunta, vetor, collections, k, trechos_por_artigo) -> list[Resultado]:
        n_candidatos = max(config.MIN_CANDIDATOS_RERANK, 4 * k)
        candidatos = []
        for collection in collections:
            candidatos += self.qdrant.query_points(
                collection, query=vetor, limit=n_candidatos, with_payload=True).points
        candidatos = sorted(candidatos, key=lambda p: p.score, reverse=True)[:n_candidatos]
        if not candidatos:
            return []

        # titulo junto do trecho: o trecho sozinho perde o contexto
        textos = [f"{_dados(p)['titulo']}\n{_dados(p)['texto']}" for p in candidatos]
        # o cross-encoder devolve um logit (-11 a +11, mais ou menos); a sigmoide
        # leva para 0..1, como o TEI do projeto original
        scores = [1 / (1 + math.exp(-s)) for s in self.reranker.rerank(pergunta, textos)]

        por_artigo: dict[str, list] = {}
        for ponto, score in zip(candidatos, scores):
            por_artigo.setdefault(_dados(ponto)["arxiv_id"], []).append((float(score), ponto))
        resultados = []
        for pares in por_artigo.values():
            pares.sort(key=lambda par: par[0], reverse=True)
            melhores = pares[:trechos_por_artigo]
            resultados.append(_resultado(_dados(melhores[0][1]), melhores[0][0],
                                         [_trecho(p, s) for s, p in melhores]))
        return sorted(resultados, key=lambda r: r.score, reverse=True)[:k]

    def artigo(self, arxiv_id: str, secao: str | None = None, max_trechos: int = 8) -> dict | None:
        """Os trechos indexados de um artigo, na ordem do texto (leitura, sem busca)."""
        filtro = models.Filter(must=[models.FieldCondition(
            key="arxiv_id", match=models.MatchValue(value=arxiv_id))])
        for tema in self.collections(None):
            pontos, _ = self.qdrant.scroll(tema, scroll_filter=filtro, limit=1000,
                                           with_payload=True, with_vectors=False)
            if pontos:
                break
        else:
            return None
        payloads = sorted((_dados(p) for p in pontos), key=lambda pl: pl["ordem"])
        if secao:
            payloads = [pl for pl in payloads if secao.lower() in pl["secao"].lower()]
        primeiro = _dados(pontos[0])
        return {
            "arxiv_id": arxiv_id, "titulo": primeiro["titulo"], "tema": primeiro["tema"],
            "url": primeiro["url"], "total_trechos": len(pontos),
            "trechos": [{"secao": pl["secao"], "ordem": pl["ordem"], "texto": pl["texto"]}
                        for pl in payloads[:max_trechos]],
        }


def imprimir(resposta: RespostaBusca) -> None:
    if resposta.traduzida:
        print(f"(traduzida: {resposta.pergunta_busca!r})")
    if not resposta.resultados:
        print("Nada encontrado -- o indice esta vazio? Rode o pipeline primeiro.")
    for n, r in enumerate(resposta.resultados, start=1):
        print(f"\n[{n}] {r.titulo}\n    {r.url}  tema={r.tema}  score={r.score:.3f}")
        for t in r.trechos:
            print(f"    - ({t.secao}) {t.texto[:220]}...")


if __name__ == "__main__":
    configurar_log()
    parser = argparse.ArgumentParser(description="Busca semantica nos artigos indexados")
    parser.add_argument("pergunta")
    parser.add_argument("--tema", choices=config.TEMAS)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--trechos", type=int, default=2)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--sem-traducao", action="store_true",
                        help="busca a pergunta como veio, para comparar")
    parser.add_argument("--json", action="store_true", help="saida em JSON")
    args = parser.parse_args()
    resposta = Buscador().buscar(args.pergunta, args.tema, args.k, args.trechos, args.rerank,
                                 traduzir=not args.sem_traducao)
    if args.json:
        import json
        print(json.dumps(asdict(resposta), ensure_ascii=False, indent=1))
    else:
        imprimir(resposta)
