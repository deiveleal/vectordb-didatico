"""Etapa 1 -- bronze: baixa o HTML do arXiv e guarda do jeito que veio.

Duas entradas, as duas paginas comuns do site (nenhuma API):

- listagem do dia: https://arxiv.org/list/<categoria>/new, uma por categoria.
  Traz os artigos anunciados no dia, com titulo, autores, categorias e abstract.
- link de artigo: https://arxiv.org/abs/<id>, para incluir um artigo
  especifico (aceita tambem /pdf/<id>, /html/<id> ou so o id).

Regra da camada bronze: NADA de interpretar o conteudo aqui. Grava o HTML
cru e anota no manifesto de onde veio, quando e com que hash. Se o parse da
etapa 2 tiver um bug, corrige-se o parse e roda de novo a partir daqui, sem
baixar nada outra vez.

    data/bronze/listas/AAAA-MM-DD/cs.CL.html   listagem baixada naquele dia
    data/bronze/artigos/2609.28614.html        pagina /abs de um link
    data/bronze/manifesto.jsonl                uma linha por arquivo baixado

Arquivo que ja existe nao e baixado de novo: rodar duas vezes no mesmo dia
nao repete requisicoes ao arXiv.

Uso: python -m didatico.etapa1_bronze --categorias cs.CL --links 2609.28614
"""

import argparse
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path

from didatico import config
from didatico.util import ClienteArxiv, configurar_log, normalizar_id

logger = logging.getLogger(__name__)

MANIFESTO = config.BRONZE / "manifesto.jsonl"


def caminho_listagem(categoria: str, dia: dt.date) -> Path:
    return config.BRONZE / "listas" / dia.isoformat() / f"{categoria}.html"


def caminho_pagina_artigo(arxiv_id: str) -> Path:
    return config.BRONZE / "artigos" / f"{arxiv_id}.html"


def _gravar(cliente: ClienteArxiv, url: str, destino: Path, origem: str) -> bool:
    """Baixa url em destino e anota no manifesto. False se o arXiv respondeu erro."""
    resposta = cliente.get(url)
    if resposta.status_code != 200:
        logger.warning("%s -> HTTP %d, nada gravado", url, resposta.status_code)
        return False
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(resposta.content)
    with MANIFESTO.open("a", encoding="utf-8") as manifesto:
        manifesto.write(json.dumps({
            "origem": origem,
            "url": url,
            "arquivo": str(destino.relative_to(config.DADOS)),
            "sha256": hashlib.sha256(resposta.content).hexdigest(),
            "bytes": len(resposta.content),
            "baixado_em": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }) + "\n")
    logger.info("bronze <- %s (%d KB)", url, len(resposta.content) // 1024)
    return True


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
