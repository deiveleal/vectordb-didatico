"""O pipeline inteiro, na ordem do desenho -- o papel do Prefect no original.

    bronze -> silver -> gold -> documentos -> texto -> chunks -> indice -> backup

Cada etapa le o que a anterior gravou em data/ e so faz o que falta, entao
rodar de novo e seguro: o que ja foi baixado, extraido ou indexado e pulado.
Se uma etapa falhar, corrija e rode de novo -- recomeca de onde parou.

Exemplos:
    # listagens do dia das categorias padrao (cs.CL, cs.AI, cs.LG)
    uv run python -m didatico.pipeline

    # primeira vez: so 5 documentos, para ver tudo funcionando em ~1 minuto
    uv run python -m didatico.pipeline --categorias cs.CL --limite 5

    # so artigos especificos, por link (sem listagem)
    uv run python -m didatico.pipeline --links https://arxiv.org/abs/2609.28614 2609.28653

Para rodar todo dia, agende no cron (ex.: 21h, depois do anuncio do arXiv):
    0 21 * * 1-5  cd /caminho/vectordb_didatico && uv run python -m didatico.pipeline
"""

import argparse
import logging
import time

from didatico import (backup, config, etapa1_bronze, etapa2_silver, etapa3_gold,
                      etapa4_documentos, etapa5_texto, etapa6_chunks, etapa7_indice)
from didatico.util import ClienteArxiv, configurar_log

logger = logging.getLogger("didatico.pipeline")


def run(categorias: list[str], links: list[str], limite: int | None = None,
        fazer_backup: bool = True) -> None:
    cliente = ClienteArxiv()        # um so, para o intervalo valer entre etapas
    etapas = [
        ("1 bronze", lambda: etapa1_bronze.run(categorias, links, cliente)),
        ("2 silver", etapa2_silver.run),
        ("3 gold", etapa3_gold.run),
        ("4 documentos", lambda: etapa4_documentos.run(limite, cliente)),
        ("5 texto", etapa5_texto.run),
        ("6 chunks", etapa6_chunks.run),
        ("7 indice", etapa7_indice.run),
    ]
    if fazer_backup:
        etapas.append(("8 backup", backup.run))

    inicio = time.monotonic()
    for nome, etapa in etapas:
        t = time.monotonic()
        logger.info("=== etapa %s ===", nome)
        etapa()
        logger.info("=== etapa %s: %.1f s ===", nome, time.monotonic() - t)
    logger.info("Pipeline concluido em %.0f s", time.monotonic() - inicio)


if __name__ == "__main__":
    configurar_log()
    parser = argparse.ArgumentParser(description="Pipeline arXiv -> Qdrant, versao didatica")
    parser.add_argument("--categorias", nargs="*",
                        help=f"padrao: {' '.join(config.CATEGORIAS)} (ou nenhuma, se houver --links)")
    parser.add_argument("--links", nargs="*", default=[],
                        help="links /abs, /pdf ou ids de artigos especificos")
    parser.add_argument("--limite", type=int,
                        help="no maximo N documentos baixados nesta execucao")
    parser.add_argument("--sem-backup", action="store_true")
    args = parser.parse_args()

    categorias = args.categorias
    if categorias is None:
        categorias = [] if args.links else list(config.CATEGORIAS)
    run(categorias, args.links, args.limite, not args.sem_backup)
