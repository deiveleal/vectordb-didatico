"""Etapa 4 -- documents: baixa o texto completo de cada artigo da gold.

O arXiv publica a maioria dos artigos tambem em HTML (arxiv.org/html/<id>),
gerado do LaTeX pelo LaTeXML. Esse HTML ja vem com a estrutura marcada --
<section>, titulos de secao, abstract, legendas, bibliografia e apendices
com classes proprias --, que e exatamente o que o projeto original pede ao
GROBID para reconstruir a partir do PDF. Entao:

- tem HTML  -> baixa o HTML (melhor estrutura, arquivo menor);
- nao tem   -> baixa o PDF, e a etapa 5 extrai o texto com PyMuPDF.

Artigo que ja tem documento e pulado: rodar de novo so baixa o que falta.

    data/documents/<tema>/<arxiv_id>.html   ou   .pdf

Uso: python -m didatico.etapa4_documentos --limite 5
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

from didatico import config
from didatico.etapa3_gold import ARQUIVO as ARQUIVO_GOLD
from didatico.util import ClienteArxiv, configurar_log, ler_jsonl

logger = logging.getLogger(__name__)


def localizar_documento(artigo: dict) -> Path | None:
    """O documento ja baixado do artigo (HTML tem preferencia), ou None."""
    pasta = config.DOCUMENTOS / artigo["tema"]
    for extensao in (".html", ".pdf"):
        if (caminho := pasta / f"{artigo['arxiv_id']}{extensao}").exists():
            return caminho
    return None


def baixar(cliente: ClienteArxiv, artigo: dict) -> str:
    """Baixa o documento de um artigo. Devolve "html", "pdf" ou "falha"."""
    pasta = config.DOCUMENTOS / artigo["tema"]
    pasta.mkdir(parents=True, exist_ok=True)
    arxiv_id = artigo["arxiv_id"]

    if artigo.get("tem_html"):
        resposta = cliente.get(f"{config.SITE}/html/{arxiv_id}")
        # sem HTML o arXiv responde 404 ou uma pagina de aviso sem o documento
        if resposta.status_code == 200 and "ltx_document" in resposta.text:
            (pasta / f"{arxiv_id}.html").write_bytes(resposta.content)
            return "html"
        logger.info("%s: sem HTML utilizavel (HTTP %d), indo para o PDF",
                    arxiv_id, resposta.status_code)

    resposta = cliente.get(f"{config.SITE}/pdf/{arxiv_id}")
    if resposta.status_code == 200 and resposta.content.startswith(b"%PDF"):
        (pasta / f"{arxiv_id}.pdf").write_bytes(resposta.content)
        return "pdf"
    logger.warning("%s: PDF tambem falhou (HTTP %d)", arxiv_id, resposta.status_code)
    return "falha"


def run(limite: int | None = None, cliente: ClienteArxiv | None = None) -> dict[str, int]:
    """Baixa o que falta. `limite` = no maximo quantos downloads nesta execucao
    (cada artigo leva >= 3 s pelo intervalo do arXiv)."""
    gold = ler_jsonl(ARQUIVO_GOLD)
    faltando = [a for a in gold if localizar_documento(a) is None]
    contagem: Counter[str] = Counter(ja_tinha=len(gold) - len(faltando))
    if limite is not None and len(faltando) > limite:
        contagem["ficou_para_depois"] = len(faltando) - limite
        faltando = faltando[:limite]
    if faltando:
        logger.info("Baixando %d documentos (~%d s pelo intervalo do arXiv)",
                    len(faltando), len(faltando) * config.INTERVALO_ENTRE_REQUISICOES)

    cliente = cliente or ClienteArxiv()
    for i, artigo in enumerate(faltando, start=1):
        contagem[baixar(cliente, artigo)] += 1
        if i % 10 == 0:
            logger.info("... %d/%d", i, len(faltando))
    logger.info("Documentos: %s", dict(contagem))
    return dict(contagem)


if __name__ == "__main__":
    configurar_log()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limite", type=int)
    run(parser.parse_args().limite)
