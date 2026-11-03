"""Etapa 1 -- bronze: baixa o HTML do arXiv e guarda do jeito que veio.

Tres entradas -- duas sao paginas comuns do site (nenhuma API):

- listagem do dia: https://arxiv.org/list/<categoria>/new, uma por categoria.
  Traz os artigos anunciados no dia, com titulo, autores, categorias e abstract.
- link de artigo: https://arxiv.org/abs/<id>, para incluir um artigo
  especifico (aceita tambem /pdf/<id>, /html/<id> ou so o id).
- PDF enviado (aba "Inserir" da UI): o arquivo como veio, mais um JSON com o
  que foi digitado no formulario.

Regra da camada bronze: NADA de interpretar o conteudo aqui. Grava o HTML
cru e anota no manifesto de onde veio, quando e com que hash. Se o parse da
etapa 2 tiver um bug, corrige-se o parse e roda de novo a partir daqui, sem
baixar nada outra vez.

    data/bronze/listas/AAAA-MM-DD/cs.CL.html   listagem baixada naquele dia
    data/bronze/artigos/2609.28614.html        pagina /abs de um link
    data/bronze/uploads/<id>.pdf + <id>.json   PDF enviado e dados do formulario
    data/bronze/temas_manuais.json             tema escolhido na insercao
    data/bronze/manifesto.jsonl                uma linha por arquivo recebido

Arquivo que ja existe nao e baixado de novo: rodar duas vezes no mesmo dia
nao repete requisicoes ao arXiv.

Uso: python -m src.etapa1_bronze --categorias cs.CL --links 2609.28614
"""

import argparse
import datetime as dt
import hashlib
import json
import logging
import re
from pathlib import Path

import pymupdf

from src import config
from src.util import ClienteArxiv, configurar_log, normalizar_id

logger = logging.getLogger(__name__)

MANIFESTO = config.BRONZE / "manifesto.jsonl"
UPLOADS = config.BRONZE / "uploads"
TEMAS_MANUAIS = config.BRONZE / "temas_manuais.json"
# O arXiv carimba a margem de todo PDF: "arXiv:2610.40927v1 [cs.AI] 29 Oct 2026"
CARIMBO_ARXIV = re.compile(r"arXiv:(\d{4}\.\d{4,5})(v\d+)?")


def caminho_listagem(categoria: str, dia: dt.date) -> Path:
    return config.BRONZE / "listas" / dia.isoformat() / f"{categoria}.html"


def caminho_pagina_artigo(arxiv_id: str) -> Path:
    return config.BRONZE / "artigos" / f"{arxiv_id}.html"


def _anotar(origem: str, url: str | None, destino: Path, conteudo: bytes) -> None:
    with MANIFESTO.open("a", encoding="utf-8") as manifesto:
        manifesto.write(json.dumps({
            "origem": origem,
            "url": url,
            "arquivo": str(destino.relative_to(config.DADOS)),
            "sha256": hashlib.sha256(conteudo).hexdigest(),
            "bytes": len(conteudo),
            "baixado_em": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }) + "\n")


def _gravar(cliente: ClienteArxiv, url: str, destino: Path, origem: str) -> bool:
    """Baixa url em destino e anota no manifesto. False se o arXiv respondeu erro."""
    resposta = cliente.get(url)
    if resposta.status_code != 200:
        logger.warning("%s -> HTTP %d, nada gravado", url, resposta.status_code)
        return False
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(resposta.content)
    _anotar(origem, url, destino, resposta.content)
    logger.info("bronze <- %s (%d KB)", url, len(resposta.content) // 1024)
    return True


def id_do_pdf(conteudo: bytes) -> str:
    """O id do arXiv, se o PDF tiver o carimbo; senao 'local-' + inicio do hash.

    Com o id do arXiv, o mesmo artigo que tambem veio pela listagem vira UMA
    linha na silver (a duplicata e removida la). O hash faz o mesmo arquivo
    enviado duas vezes cair no mesmo id.
    """
    with pymupdf.open(stream=conteudo, filetype="pdf") as pdf:
        primeira = pdf[0].get_text() if len(pdf) else ""
    if achado := CARIMBO_ARXIV.search(primeira):
        return achado.group(1)
    return "local-" + hashlib.sha256(conteudo).hexdigest()[:12]


def guardar_pdf_enviado(conteudo: bytes, nome_arquivo: str, titulo: str | None = None) -> str:
    """Grava o PDF enviado e o que veio no formulario. Devolve o id do artigo."""
    if not conteudo.startswith(b"%PDF"):
        raise ValueError("o arquivo nao e um PDF")
    arxiv_id = id_do_pdf(conteudo)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    destino = UPLOADS / f"{arxiv_id}.pdf"
    destino.write_bytes(conteudo)
    (UPLOADS / f"{arxiv_id}.json").write_text(json.dumps({
        "arxiv_id": arxiv_id,
        "nome_arquivo": nome_arquivo,
        "titulo_informado": (titulo or "").strip() or None,
        "enviado_em": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    _anotar("upload", None, destino, conteudo)
    logger.info("bronze <- PDF enviado %r como %s (%d KB)",
                nome_arquivo, arxiv_id, len(conteudo) // 1024)
    return arxiv_id


def registrar_tema_manual(arxiv_id: str, tema: str) -> None:
    """Quem insere um artigo pode escolher o tema: a gold usa este no lugar da
    classificacao automatica (que deixaria de fora artigo sem as palavras-chave)."""
    if tema not in config.TEMAS:
        raise ValueError(f"tema {tema!r} invalido; use um de {config.TEMAS}")
    temas = json.loads(TEMAS_MANUAIS.read_text()) if TEMAS_MANUAIS.exists() else {}
    temas[arxiv_id] = tema
    TEMAS_MANUAIS.parent.mkdir(parents=True, exist_ok=True)
    TEMAS_MANUAIS.write_text(json.dumps(temas, indent=1))


def run(categorias: list[str], links: list[str], cliente: ClienteArxiv | None = None,
        dia: dt.date | None = None) -> dict[str, int]:
    cliente = cliente or ClienteArxiv()
    dia = dia or dt.date.today()
    contagem = {"baixados": 0, "ja_existiam": 0, "falhas": 0}

    alvos = [(f"{config.SITE}/list/{c}/new", caminho_listagem(c, dia), "listagem")
             for c in categorias]
    for link in links:
        arxiv_id = normalizar_id(link)
        alvos.append((f"{config.SITE}/abs/{arxiv_id}", caminho_pagina_artigo(arxiv_id), "link"))

    for url, destino, origem in alvos:
        if destino.exists():
            contagem["ja_existiam"] += 1
            continue
        contagem["baixados" if _gravar(cliente, url, destino, origem) else "falhas"] += 1

    logger.info("Bronze: %s", contagem)
    return contagem


if __name__ == "__main__":
    configurar_log()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--categorias", nargs="*", default=list(config.CATEGORIAS))
    parser.add_argument("--links", nargs="*", default=[])
    args = parser.parse_args()
    run(args.categorias, args.links)
