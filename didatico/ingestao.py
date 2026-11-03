"""Insercao de UM artigo pela API: link do arXiv ou PDF enviado.

E o mesmo pipeline da linha de comando, restrito a um artigo:

    bronze (pagina /abs ou PDF) -> silver -> gold -> documento -> texto -> chunks -> indice

Por que uma fila? Indexar leva de alguns segundos a um minuto (3 s entre
requisicoes ao arXiv, embeddings na CPU). Uma requisicao HTTP parada esse
tempo todo e ruim para quem chama, entao a API responde na hora com um id de
tarefa e uma thread de fundo faz o trabalho. Quem chamou pergunta
GET /ingestoes/<id> ate ver "concluida" ou "falhou" -- e o que a UI faz.

Uma thread so, uma insercao por vez: as etapas reescrevem os mesmos arquivos
(silver, gold) e nao podem rodar em paralelo. Pelo mesmo motivo, cada
insercao segura a trava_de_escrita, que tambem segura o pipeline da linha de
comando se ele estiver rodando em outro processo.

As tarefas ficam so na memoria: reiniciar a API esquece o historico (os
artigos ja indexados continuam no indice, claro).
"""

import datetime as dt
import logging
import queue
import threading
import uuid
from dataclasses import asdict, dataclass, field

from didatico import (config, etapa1_bronze, etapa2_silver, etapa3_gold, etapa4_documentos,
                      etapa5_texto, etapa6_chunks, etapa7_indice)
from didatico.busca import Buscador
from didatico.util import ClienteArxiv, ler_jsonl, normalizar_id, trava_de_escrita

logger = logging.getLogger(__name__)

MAX_TAREFAS_GUARDADAS = 100


class FalhaNaInsercao(Exception):
    """Erro esperado (artigo fora do tema, link sem pagina...), com mensagem para o usuario."""


@dataclass
class Tarefa:
    id: str
    tipo: str                         # "link" ou "pdf"
    entrada: str                      # o link ou o nome do arquivo
    tema: str | None                  # escolhido na insercao; None = automatico
    titulo: str | None = None         # so para PDF: titulo digitado
    status: str = "na_fila"           # na_fila -> rodando -> concluida | falhou
    etapa: str = ""
    criada_em: str = field(default_factory=lambda: _agora())
    terminada_em: str | None = None
    resultado: dict | None = None
    erro: str | None = None
    _pdf: bytes | None = field(default=None, repr=False)

    def publica(self) -> dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


def _agora() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class Ingestoes:
    """A fila de insercoes e a thread que a consome."""

    def __init__(self, buscador: Buscador):
        self.buscador = buscador
        self.cliente = ClienteArxiv()
        self.tarefas: dict[str, Tarefa] = {}
        self.fila: queue.Queue[Tarefa] = queue.Queue()
        threading.Thread(target=self._trabalhar, daemon=True, name="ingestao").start()

    # --- entrada (chamado pelos endpoints) ----------------------------------

    def link(self, link: str, tema: str | None) -> Tarefa:
        normalizar_id(link)               # link sem id do arXiv falha ja, com 422
        return self._enfileirar(Tarefa(_novo_id(), "link", link.strip(), _validar(tema)))

    def pdf(self, conteudo: bytes, nome_arquivo: str, titulo: str | None,
            tema: str | None) -> Tarefa:
        if not conteudo.startswith(b"%PDF"):
            raise ValueError("o arquivo enviado nao e um PDF")
        return self._enfileirar(Tarefa(_novo_id(), "pdf", nome_arquivo, _validar(tema),
                                       titulo=titulo, _pdf=conteudo))

    def _enfileirar(self, tarefa: Tarefa) -> Tarefa:
        self.tarefas[tarefa.id] = tarefa
        # esquece as mais antigas: o historico e so para acompanhar
        for antiga in list(self.tarefas)[:-MAX_TAREFAS_GUARDADAS]:
            if self.tarefas[antiga].status in ("concluida", "falhou"):
                del self.tarefas[antiga]
        self.fila.put(tarefa)
        return tarefa

    # --- trabalho (thread de fundo) -----------------------------------------

    def _trabalhar(self) -> None:
        while True:
            tarefa = self.fila.get()
            tarefa.status = "rodando"
            try:
                tarefa.resultado = self._processar(tarefa)
                tarefa.status = "concluida"
            except FalhaNaInsercao as erro:
                tarefa.status, tarefa.erro = "falhou", str(erro)
            except Exception as erro:                       # bug ou rede: registra e segue
                logger.exception("Insercao %s falhou", tarefa.id)
                tarefa.status, tarefa.erro = "falhou", f"{type(erro).__name__}: {erro}"
            finally:
                tarefa.etapa = ""
                tarefa.terminada_em = _agora()
                tarefa._pdf = None                          # libera a memoria do arquivo

    def _processar(self, tarefa: Tarefa) -> dict:
        def etapa(texto: str) -> None:
            tarefa.etapa = texto
            logger.info("Insercao %s: %s", tarefa.id, texto)

        etapa("esperando a vez (outro processo gravando em data/)")
        with trava_de_escrita():
            # 1. bronze: a entrada crua
            if tarefa.tipo == "link":
                arxiv_id = normalizar_id(tarefa.entrada)
                etapa("1/7 bronze: baixando a pagina do artigo no arXiv")
                etapa1_bronze.run([], [arxiv_id], self.cliente)
                if not etapa1_bronze.caminho_pagina_artigo(arxiv_id).exists():
                    raise FalhaNaInsercao(f"o arXiv nao devolveu a pagina de {arxiv_id}")
            else:
                etapa("1/7 bronze: guardando o PDF enviado")
                assert tarefa._pdf is not None
                arxiv_id = etapa1_bronze.guardar_pdf_enviado(
                    tarefa._pdf, tarefa.entrada, tarefa.titulo)
            if tarefa.tema:
                etapa1_bronze.registrar_tema_manual(arxiv_id, tarefa.tema)

            # 2 e 3. silver e gold: sao refeitas inteiras, mas sao rapidas
            etapa("2/7 silver: lendo os metadados")
            etapa2_silver.run()
            etapa("3/7 gold: aplicando as regras (tema)")
            etapa3_gold.run()
            artigo = next((a for a in ler_jsonl(etapa3_gold.ARQUIVO)
                           if a["arxiv_id"] == arxiv_id), None)
            if artigo is None:
                raise FalhaNaInsercao(
                    f"{arxiv_id} nao fala de RAG, agentes nem LLM pelas palavras do titulo "
                    "e do abstract -- escolha o tema na insercao para incluir mesmo assim")

            # 4 a 6: so este artigo (refazer: um PDF novo substitui o anterior)
            somente = {arxiv_id}
            etapa("4/7 documento: obtendo o texto completo")
            etapa4_documentos.run(cliente=self.cliente, somente=somente)
            if etapa4_documentos.localizar_documento(artigo) is None:
                raise FalhaNaInsercao(f"nao consegui obter o documento de {arxiv_id}")
            etapa("5/7 texto: extraindo secoes e paragrafos")
            etapa5_texto.run(refazer=True, somente=somente)
            etapa("6/7 chunks: quebrando em trechos")
            etapa6_chunks.run(refazer=True, somente=somente)

            # 7. indice: com o Qdrant e o modelo que a API ja tem abertos
            etapa("7/7 indice: calculando embeddings e gravando no Qdrant")
            with self.buscador.trava:
                etapa7_indice.run(qdrant=self.buscador.qdrant, modelo=self.buscador.embedding,
                                  somente=somente)

        documento = etapa4_documentos.localizar_documento(artigo)
        return {
            "arxiv_id": arxiv_id,
            "titulo": artigo["titulo"],
            "tema": artigo["tema"],
            "evidencia": artigo["evidencia"],
            "url": artigo["url"],
            "fonte": documento.suffix.lstrip(".") if documento else None,
            "chunks": len(ler_jsonl(etapa6_chunks.caminho_chunks(arxiv_id))),
        }


def _novo_id() -> str:
    return uuid.uuid4().hex[:10]


def _validar(tema: str | None) -> str | None:
    if tema in (None, "", "auto"):
        return None
    if tema not in config.TEMAS:
        raise ValueError(f"tema {tema!r} invalido; use um de {config.TEMAS} ou deixe em branco")
    return tema
