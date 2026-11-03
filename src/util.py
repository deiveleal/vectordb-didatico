"""Pecas pequenas que todas as etapas usam: HTTP educado, ids do arXiv, JSONL."""

import fcntl
import json
import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import requests

from src import config

logger = logging.getLogger(__name__)


# --- HTTP ----------------------------------------------------------------------

class ClienteArxiv:
    """GET no site do arXiv respeitando o intervalo minimo entre requisicoes.

    Uma instancia por execucao: ela lembra quando foi a ultima requisicao e
    espera o que faltar antes da proxima. Erro de rede ou 5xx tenta de novo
    algumas vezes, esperando mais a cada tentativa.
    """

    def __init__(self, intervalo: float = config.INTERVALO_ENTRE_REQUISICOES):
        self.intervalo = intervalo
        self._ultima = 0.0
        self.sessao = requests.Session()
        self.sessao.headers["User-Agent"] = config.USER_AGENT

    def get(self, url: str, tentativas: int = 3) -> requests.Response:
        for tentativa in range(1, tentativas + 1):
            espera = self._ultima + self.intervalo - time.monotonic()
            if espera > 0:
                time.sleep(espera)
            self._ultima = time.monotonic()
            try:
                resposta = self.sessao.get(url, timeout=60)
            except requests.RequestException as erro:
                logger.warning("GET %s falhou (%s), tentativa %d", url, erro, tentativa)
            else:
                if resposta.status_code < 500:
                    return resposta
                logger.warning("GET %s -> %d, tentativa %d", url, resposta.status_code, tentativa)
            time.sleep(self.intervalo * tentativa)
        raise RuntimeError(f"nao consegui baixar {url}")


# --- Ids do arXiv --------------------------------------------------------------

# Id novo: AAMM.NNNNN, com versao opcional (2609.28614v2).
ID_ARXIV = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def normalizar_id(link_ou_id: str) -> str:
    """'https://arxiv.org/abs/2609.28614v2', '/pdf/2609.28614' ou '2609.28614'
    -> '2609.28614' (sem versao: a pagina /abs/<id> sempre mostra a ultima)."""
    achado = ID_ARXIV.search(link_ou_id)
    if not achado:
        raise ValueError(f"nao achei um id do arXiv em {link_ou_id!r}")
    return achado.group(1)


# --- JSONL ---------------------------------------------------------------------
# Um objeto JSON por linha: da para abrir no editor, contar linhas com wc -l e
# ler com pandas.read_json(..., lines=True). E o "formato de tabela" daqui.

def ler_jsonl(caminho: Path) -> list[dict]:
    if not caminho.exists():
        return []
    with caminho.open(encoding="utf-8") as arquivo:
        return [json.loads(linha) for linha in arquivo if linha.strip()]


def gravar_jsonl(caminho: Path, linhas: Iterable[dict]) -> int:
    """Grava num arquivo temporario e troca no fim: quem le nunca ve o arquivo
    pela metade (a mesma ideia do commit atomico do Iceberg, em miniatura)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    n = 0
    with temporario.open("w", encoding="utf-8") as arquivo:
        for linha in linhas:
            arquivo.write(json.dumps(linha, ensure_ascii=False) + "\n")
            n += 1
    temporario.replace(caminho)
    return n


@contextmanager
def trava_de_escrita() -> Iterator[None]:
    """Um escritor por vez em data/, entre PROCESSOS.

    O pipeline pela linha de comando e a ingestao pela API (aba "Inserir" da
    UI) reescrevem os mesmos arquivos (silver, gold, chunks). Dois ao mesmo
    tempo perderiam linhas um do outro. flock e uma trava do sistema
    operacional num arquivo: o segundo espera o primeiro terminar, e ela se
    solta sozinha se o processo morrer.
    """
    config.DADOS.mkdir(parents=True, exist_ok=True)
    with (config.DADOS / ".trava_escrita").open("w") as arquivo:
        if not _tentar_travar(arquivo):
            logger.info("Outro processo esta gravando em data/ -- esperando terminar")
            fcntl.flock(arquivo, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(arquivo, fcntl.LOCK_UN)


def _tentar_travar(arquivo) -> bool:
    try:
        fcntl.flock(arquivo, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def em_lotes(itens: list, tamanho: int) -> Iterator[list]:
    for inicio in range(0, len(itens), tamanho):
        yield itens[inicio:inicio + tamanho]


def configurar_log() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    # o httpx (usado pelo qdrant-client) loga cada requisicao em INFO
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # avisa a cada execucao que nao ha HF_TOKEN -- desnecessario: os modelos sao publicos
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
