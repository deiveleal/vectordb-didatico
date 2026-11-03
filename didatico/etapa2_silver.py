"""Etapa 2 -- silver: HTML cru do bronze -> uma linha limpa por artigo.

So parse e limpeza tecnica, nenhuma regra de negocio:

- le todas as listagens e paginas /abs do bronze;
- tira de cada uma titulo, autores, abstract, categorias, datas e versoes;
- normaliza espacos (o HTML quebra linha no meio do titulo);
- remove a duplicata entre categorias: um artigo de cs.CL que tambem e de
  cs.LG aparece nas duas listagens e e o mesmo artigo -- fica uma linha, com
  `listado_em` dizendo onde foi visto.

Como saber se o artigo e novo, sem API? A propria listagem diz: ela tem tres
blocos, "New submissions", "Cross submissions" (novo, mas de outra categoria)
e "Replacement submissions" (versao nova de artigo antigo). A pagina /abs
mostra o historico [v1], [v2], ... Quem decide o que entra e a etapa 3.

A silver e refeita inteira a partir do bronze a cada execucao: simples e
idempotente (rodar duas vezes da o mesmo arquivo).

    data/silver/artigos.jsonl

Uso: python -m didatico.etapa2_silver
"""

import datetime as dt
import logging
import re
from email.utils import parsedate_to_datetime
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from didatico import config
from didatico.util import configurar_log, gravar_jsonl

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


# --- Juntando tudo -------------------------------------------------------------

def juntar(artigos: list[dict]) -> list[dict]:
    """Uma linha por arxiv_id. A pagina /abs ganha da listagem (tem o historico
    de versoes); entre listagens, fica a primeira e soma-se `listado_em`."""
    por_id: dict[str, dict] = {}
    for artigo in artigos:
        atual = por_id.get(artigo["arxiv_id"])
        if atual is None:
            por_id[artigo["arxiv_id"]] = artigo
            continue
        listado_em = sorted(set(atual["listado_em"]) | set(artigo["listado_em"]))
        if artigo["tipo_anuncio"] == "link" and atual["tipo_anuncio"] != "link":
            artigo = {**artigo, "data_anuncio": atual["data_anuncio"]}
            por_id[artigo["arxiv_id"]] = artigo
        por_id[artigo["arxiv_id"]]["listado_em"] = listado_em
    return sorted(por_id.values(), key=lambda a: a["arxiv_id"])


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

    linhas = juntar(artigos)
    gravar_jsonl(ARQUIVO, linhas)
    logger.info("Silver: %d listagens + %d paginas -> %d entradas -> %d artigos unicos",
                len(listagens), len(paginas), len(artigos), len(linhas))
    return len(linhas)


if __name__ == "__main__":
    configurar_log()
    run()
