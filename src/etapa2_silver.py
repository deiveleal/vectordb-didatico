"""Etapa 2 -- silver: HTML cru do bronze -> uma linha limpa por artigo.

So parse e limpeza tecnica, nenhuma regra de negocio:

- le todas as listagens, paginas /abs e PDFs enviados do bronze;
- tira de cada uma titulo, autores, abstract, categorias, datas e versoes;
- normaliza espacos (o HTML quebra linha no meio do titulo);
- remove a duplicata entre categorias: um artigo de cs.CL que tambem e de
  cs.LG aparece nas duas listagens e e o mesmo artigo -- fica uma linha, com
  `listado_em` dizendo onde foi visto.

Como saber se o artigo e novo, sem API? A propria listagem diz: ela tem tres
blocos, "New submissions", "Cross submissions" (novo, mas de outra categoria)
e "Replacement submissions" (versao nova de artigo antigo). A pagina /abs
mostra o historico [v1], [v2], ... Quem decide o que entra e a etapa 3.

PDF enviado: titulo do formulario (ou a maior fonte da 1a pagina) e o
abstract que estiver entre "Abstract" e a introducao. Sem autores nem
categorias -- o PDF nao os traz de um jeito confiavel.

A silver e refeita inteira a partir do bronze a cada execucao: simples e
idempotente (rodar duas vezes da o mesmo arquivo).

    data/silver/artigos.jsonl

Uso: python -m src.etapa2_silver
"""

import datetime as dt
import json
import logging
import re
from email.utils import parsedate_to_datetime
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup, Tag

from src import config
from src.util import configurar_log, gravar_jsonl

logger = logging.getLogger(__name__)

ARQUIVO = config.SILVER / "artigos.jsonl"

MESES = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
# "Computation and Language (cs.CL); Machine Learning (cs.LG)" -> cs.CL, cs.LG
CODIGO_CATEGORIA = re.compile(r"\(([a-z-]+(?:\.[A-Za-z-]+)?)\)")
DATA_DA_LISTAGEM = re.compile(r"listings for \w+, (\d{1,2}) (\w+) (\d{4})")
VERSAO = re.compile(r"\[v(\d+)\]\s*([^(\[]+?)\s*\(")


def limpar(texto: str | None) -> str:
    return " ".join(texto.split()) if texto else ""


def _sem_rotulo(elemento: Tag | None) -> str:
    """Texto do elemento sem o <span class="descriptor"> ("Title:", "Subjects:")."""
    if elemento is None:
        return ""
    for rotulo in elemento.select("span.descriptor"):
        rotulo.extract()
    return limpar(elemento.get_text(" "))


def _categorias(texto: str) -> list[str]:
    return CODIGO_CATEGORIA.findall(texto)


# --- Listagem /list/<categoria>/new --------------------------------------------

def tipo_do_bloco(titulo_h3: str) -> str:
    titulo = titulo_h3.lower()
    if titulo.startswith("new"):
        return "new"
    if titulo.startswith("cross"):
        return "cross"
    return "replacement"


def data_da_listagem(sopa: BeautifulSoup) -> str | None:
    """'Showing new listings for Monday, 2 November 2026' -> '2026-11-02'.

    Sem strptime("%B"): ele depende do idioma do sistema, e aqui o texto e
    sempre ingles.
    """
    achado = DATA_DA_LISTAGEM.search(sopa.get_text(" "))
    if not achado:
        return None
    dia, mes, ano = achado.groups()
    return dt.date(int(ano), MESES[mes.lower()], int(dia)).isoformat()


def ler_listagem(html: str, categoria: str) -> list[dict]:
    sopa = BeautifulSoup(html, "html.parser")
    data_anuncio = data_da_listagem(sopa)
    artigos = []
    for bloco in sopa.select("dl#articles"):
        h3 = bloco.find("h3")
        tipo = tipo_do_bloco(h3.get_text() if h3 else "")
        # cada artigo e um par <dt> (id e links) + <dd> (metadados)
        for dt_tag in bloco.find_all("dt"):
            dd = dt_tag.find_next_sibling("dd")
            link_abs = dt_tag.select_one('a[title="Abstract"]')
            if dd is None or link_abs is None:
                continue
            arxiv_id = link_abs.get("id") or limpar(link_abs.get_text()).removeprefix("arXiv:")
            subjects = _sem_rotulo(dd.select_one("div.list-subjects"))
            categorias = _categorias(subjects)
            abstract = dd.select_one("p.mathjax")
            artigos.append({
                "arxiv_id": arxiv_id,
                "titulo": _sem_rotulo(dd.select_one("div.list-title")),
                "autores": [limpar(a.get_text()) for a in dd.select("div.list-authors a")],
                "abstract": limpar(abstract.get_text(" ")) if abstract else "",
                "categorias": categorias,
                "categoria_principal": categorias[0] if categorias else None,
                "tipo_anuncio": tipo,
                # new e cross sao a v1; replacement e v2 ou mais (a listagem nao diz qual)
                "n_versoes": 1 if tipo in ("new", "cross") else None,
                "data_v1": data_anuncio if tipo in ("new", "cross") else None,
                "data_anuncio": data_anuncio,
                "tem_html": dt_tag.select_one('a[title="View HTML"]') is not None,
                "listado_em": [categoria],
            })
    return artigos


# --- Pagina /abs/<id> ----------------------------------------------------------

def ler_pagina_artigo(html: str) -> dict | None:
    sopa = BeautifulSoup(html, "html.parser")

    def meta(nome: str) -> str:
        tag = sopa.find("meta", attrs={"name": nome})
        return limpar(str(tag.get("content", ""))) if tag else ""

    arxiv_id = meta("citation_arxiv_id")
    if not arxiv_id:
        return None
    historico = sopa.select_one("div.submission-history")
    # "[v1] Wed, 23 Sep 2026 17:54:01 UTC (10,605 KB) [v2] ..."
    versoes = VERSAO.findall(historico.get_text(" ")) if historico else []
    data_v1 = None
    if versoes:
        # "Wed, 23 Sep 2026 17:54:01 UTC" -- formato de data de e-mail
        data_v1 = parsedate_to_datetime(versoes[0][1]).date().isoformat()
    categorias = _categorias(_sem_rotulo(sopa.select_one("td.subjects")))
    return {
        "arxiv_id": arxiv_id,
        "titulo": meta("citation_title"),
        "autores": [limpar(a.get_text()) for a in sopa.select("div.authors a")],
        "abstract": meta("citation_abstract"),
        "categorias": categorias,
        "categoria_principal": categorias[0] if categorias else None,
        "tipo_anuncio": "link",
        "n_versoes": len(versoes) or None,
        "data_v1": data_v1,
        "data_anuncio": None,
        "tem_html": sopa.select_one("a#latexml-download-link") is not None,
        "listado_em": [],
    }


# --- PDF enviado -----------------------------------------------------------------

# O abstract vai do titulo "Abstract" ate o primeiro titulo do corpo
FIM_DO_ABSTRACT = re.compile(
    r"\n\s*(?:1\.?\s*\n?\s*Introduction|I\.\s+INTRODUCTION|Introduction\s*\n|"
    r"Keywords|Index Terms|CCS Concepts)", re.IGNORECASE)
INICIO_DO_ABSTRACT = re.compile(r"\bAbstract\b[\s.:—-]*", re.IGNORECASE)


def titulo_do_pdf(pdf: pymupdf.Document) -> str:
    """O texto com a MAIOR fonte da primeira pagina -- quase sempre o titulo."""
    trechos: list[tuple[float, str]] = []
    for bloco in pdf[0].get_text("dict")["blocks"]:
        for linha in bloco.get("lines", []):
            for span in linha["spans"]:
                if span["text"].strip():
                    trechos.append((round(span["size"], 1), span["text"]))
    if not trechos:
        return ""
    maior = max(tamanho for tamanho, _ in trechos)
    return limpar(" ".join(texto for tamanho, texto in trechos if tamanho == maior))


def abstract_do_pdf(pdf: pymupdf.Document) -> str:
    texto = "\n".join(pdf[i].get_text() for i in range(min(2, len(pdf))))
    if not (inicio := INICIO_DO_ABSTRACT.search(texto)):
        return ""
    resto = texto[inicio.end():]
    fim = FIM_DO_ABSTRACT.search(resto)
    abstract = resto[:fim.start()] if fim else resto[:2500]
    return limpar(re.sub(r"-\n(?=[a-z])", "", abstract))


def ler_pdf_enviado(pdf_caminho: Path) -> dict:
    formulario = json.loads(pdf_caminho.with_suffix(".json").read_text(encoding="utf-8"))
    with pymupdf.open(pdf_caminho) as pdf:
        titulo = formulario["titulo_informado"] or titulo_do_pdf(pdf) or formulario["nome_arquivo"]
        abstract = abstract_do_pdf(pdf)
    return {
        "arxiv_id": formulario["arxiv_id"],
        "titulo": titulo,
        "autores": [],
        "abstract": abstract,
        "categorias": [],
        "categoria_principal": None,
        "tipo_anuncio": "upload",
        "n_versoes": None,
        "data_v1": formulario["enviado_em"][:10],
        "data_anuncio": None,
        "tem_html": False,
        "listado_em": [],
        "pdf_enviado": str(pdf_caminho.relative_to(config.DADOS)),
    }


# --- Juntando tudo -------------------------------------------------------------

# De onde vem o melhor metadado: a pagina /abs tem o historico de versoes; a
# listagem tem autores e categorias; o PDF enviado so tem o que deu para tirar
# do texto.
PRIORIDADE = {"link": 3, "new": 2, "cross": 2, "replacement": 2, "upload": 1}
INSERIDO_MANUALMENTE = {"link", "upload"}


def juntar(artigos: list[dict]) -> list[dict]:
    """Uma linha por arxiv_id: os metadados da melhor origem, mais o que as
    outras acrescentam (onde foi listado, data do anuncio, PDF enviado)."""
    grupos: dict[str, list[dict]] = {}
    for artigo in artigos:
        grupos.setdefault(artigo["arxiv_id"], []).append(artigo)

    linhas = []
    for vistos in grupos.values():
        # max() fica com o primeiro em caso de empate: a listagem lida primeiro
        melhor = max(vistos, key=lambda a: PRIORIDADE[a["tipo_anuncio"]])
        linha = {**melhor, "listado_em": sorted({c for a in vistos for c in a["listado_em"]})}
        # pedido explicito (link ou PDF) entra na gold mesmo se for antigo
        linha["inserido_manualmente"] = any(a["tipo_anuncio"] in INSERIDO_MANUALMENTE
                                            for a in vistos)
        linha["data_anuncio"] = next((a["data_anuncio"] for a in vistos if a["data_anuncio"]),
                                     None)
        linha["pdf_enviado"] = next((a["pdf_enviado"] for a in vistos if a.get("pdf_enviado")),
                                    None)
        linhas.append(linha)
    return sorted(linhas, key=lambda a: a["arxiv_id"])


def run() -> int:
    artigos: list[dict] = []
    listagens = sorted((config.BRONZE / "listas").glob("*/*.html"))
    for arquivo in listagens:
        # o nome do arquivo e a categoria: listas/2026-11-02/cs.CL.html
        artigos += ler_listagem(arquivo.read_text(encoding="utf-8"), arquivo.stem)
    paginas: list[Path] = sorted((config.BRONZE / "artigos").glob("*.html"))
    for arquivo in paginas:
        if (artigo := ler_pagina_artigo(arquivo.read_text(encoding="utf-8"))) is not None:
            artigos.append(artigo)
        else:
            logger.warning("%s nao parece uma pagina /abs do arXiv", arquivo.name)
    enviados = sorted((config.BRONZE / "uploads").glob("*.pdf"))
    for arquivo in enviados:
        try:
            artigos.append(ler_pdf_enviado(arquivo))
        except Exception:
            logger.exception("PDF enviado %s ilegivel -- fora da silver", arquivo.name)

    linhas = juntar(artigos)
    gravar_jsonl(ARQUIVO, linhas)
    logger.info("Silver: %d listagens + %d paginas + %d PDFs enviados -> %d entradas "
                "-> %d artigos unicos", len(listagens), len(paginas), len(enviados),
                len(artigos), len(linhas))
    return len(linhas)


if __name__ == "__main__":
    configurar_log()
    run()
