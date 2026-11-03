"""Etapa 5 -- texto: documento do artigo -> paragrafos com a secao de cada um.

Faz o papel do GROBID no projeto original, em Python:

- HTML (o caso comum): o LaTeXML ja marca tudo. <section class="ltx_section">
  e uma secao, <h2>/<h3>/<h4> sao titulos, <p class="ltx_p"> e paragrafo,
  <figcaption> e legenda. Apendice e bibliografia tem classe propria
  (ltx_appendix, ltx_bibliography) -- e so nao ler. Formula vira o LaTeX
  dela (atributo alttext), se for curta; nota de rodape sai.
- PDF (quando nao ha HTML): PyMuPDF le os blocos de texto de cada pagina, na
  ordem. Bloco de uma linha com cara de titulo numerado ("3 Method",
  "4.2 Results") abre secao; o resto e paragrafo. Para em "References" ou
  "Appendix". E uma heuristica: funciona bem em artigo LaTeX de uma ou duas
  colunas, e pior em layout incomum -- por isso o HTML vem primeiro.

O abstract NAO vem daqui: ja esta na gold (veio da listagem) e a etapa 6 o
usa como chunk 0.

Saida, ao lado do documento -- um JSON legivel:

    data/documents/<tema>/<arxiv_id>.secoes.json
    {"arxiv_id": ..., "fonte": "html" | "pdf",
     "blocos": [{"grupo": "2 Related work",
                 "secao": "2 Related work › Reward hacking in LLM agents",
                 "texto": "Reward hacking, or specification gaming, ..."}, ...]}

`grupo` e a secao de primeiro nivel (um chunk nunca atravessa grupos);
`secao` inclui a subsecao, quando ha.

Uso: python -m didatico.etapa5_texto
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup, Tag

from didatico.etapa3_gold import ARQUIVO as ARQUIVO_GOLD
from didatico.etapa4_documentos import localizar_documento
from didatico.util import configurar_log, ler_jsonl

logger = logging.getLogger(__name__)

# Secoes que so poluiriam a busca. Casa com o titulo, sem caixa.
SECOES_DESCARTADAS = re.compile(
    r"acknowledg|funding|author contributions?|competing interests?|"
    r"conflicts? of interest|ethics statement|data availability|"
    r"use of (ai|llms?)|llm usage|disclosure|references|bibliography",
    re.IGNORECASE,
)
FIGURAS = "Figures and tables"
# Formula maior que isto vira "[formula]": LaTeX longo so atrapalha o embedding.
MAX_CHARS_FORMULA = 40
MIN_PALAVRAS_PARAGRAFO = 5


def caminho_secoes(documento: Path) -> Path:
    return documento.with_suffix(".secoes.json")


def limpar(texto: str) -> str:
    """Espacos normalizados e sem espaco antes de pontuacao ("word ." -> "word.")."""
    texto = " ".join(texto.split())
    texto = re.sub(r"\s+([.,;:!?)\]])", r"\1", texto)
    return re.sub(r"([(\[])\s+", r"\1", texto)


# --- HTML (LaTeXML) ------------------------------------------------------------

def _titulo(secao: Tag, nivel: str) -> str:
    cabecalho = secao.find(nivel, recursive=False)
    return limpar(cabecalho.get_text(" ")) if cabecalho else ""


def blocos_do_html(html: str) -> list[dict]:
    sopa = BeautifulSoup(html, "html.parser")

    # formulas: troca o MathML (que vira lixo no get_text) pelo LaTeX curto
    for formula in sopa.find_all("math"):
        latex = str(formula.get("alttext", ""))
        formula.replace_with(f" {latex} " if len(latex) <= MAX_CHARS_FORMULA else " [formula] ")
    for nota in sopa.select("span.ltx_note"):     # notas de rodape
        nota.decompose()

    blocos: list[dict] = []
    documento = sopa.select_one("article.ltx_document") or sopa
    # so as secoes de primeiro nivel: ltx_appendix e ltx_bibliography ficam de fora
    for secao in documento.find_all("section", class_="ltx_section", recursive=False):
        grupo = _titulo(secao, "h2") or "Body"
        if SECOES_DESCARTADAS.search(grupo):
            continue
        rotulo = grupo
        # h3 (subsecao) e h4 (paragrafo com titulo) mudam o rotulo; <p> e texto
        for elemento in secao.find_all(["h3", "h4", "p"]):
            if elemento.find_parent(["figure", "table"]):
                continue                     # legendas e tabelas: tratadas abaixo
            if elemento.name in ("h3", "h4"):
                subtitulo = limpar(elemento.get_text(" "))
                rotulo = f"{grupo} › {subtitulo}" if subtitulo else grupo
                continue
            texto = limpar(elemento.get_text(" "))
            if len(texto.split()) >= MIN_PALAVRAS_PARAGRAFO:
                blocos.append({"grupo": grupo, "secao": rotulo, "texto": texto})

        for legenda in secao.find_all("figcaption"):
            # legenda de figura costuma resumir um resultado: vale um bloco
            if texto := limpar(legenda.get_text(" ")):
                blocos.append({"grupo": FIGURAS, "secao": FIGURAS, "texto": texto})
    return blocos


# --- PDF (PyMuPDF) -------------------------------------------------------------

# "3 Method", "4.2 Results and Discussion", "IV. EXPERIMENTS"
TITULO_NUMERADO = re.compile(r"^((?:\d{1,2}(?:\.\d{1,2})*)|[IVX]{1,4}\.)\s+([A-Z].{1,80})$")
FIM_DO_CORPO = re.compile(r"^(references|bibliography|appendix|appendices)\b", re.IGNORECASE)


def blocos_do_pdf(caminho: Path) -> list[dict]:
    blocos: list[dict] = []
    grupo: str | None = None
    rotulo: str | None = None
    with pymupdf.open(caminho) as pdf:
        for pagina in pdf:
            # cada bloco: (x0, y0, x1, y1, texto, n_bloco, tipo); tipo 0 = texto
            for *_, texto, _, tipo in pagina.get_text("blocks", sort=False):
                if tipo != 0:
                    continue
                linhas = [l.strip() for l in texto.splitlines() if l.strip()]
                if not linhas:
                    continue
                if len(linhas) == 1:
                    linha = linhas[0]
                    if FIM_DO_CORPO.match(linha):
                        return blocos
                    if m := TITULO_NUMERADO.match(linha):
                        numero, titulo = m.groups()
                        if "." in numero.rstrip(".") and grupo:   # 4.2 -> subsecao
                            rotulo = f"{grupo} › {linha}"
                        else:
                            grupo = rotulo = f"{numero} {titulo}"
                        continue
                if grupo is None:
                    continue        # antes da 1a secao: titulo, autores, abstract
                # junta as linhas desfazendo a hifenizacao ("retrie-\nval")
                paragrafo = re.sub(r"-\n(?=[a-z])", "", "\n".join(linhas)).replace("\n", " ")
                paragrafo = limpar(paragrafo)
                if (len(paragrafo.split()) >= MIN_PALAVRAS_PARAGRAFO
                        and not SECOES_DESCARTADAS.search(grupo)):
                    blocos.append({"grupo": grupo, "secao": rotulo, "texto": paragrafo})
    return blocos


# --- Etapa ---------------------------------------------------------------------

def extrair(documento: Path) -> tuple[str, list[dict]]:
    if documento.suffix == ".html":
        return "html", blocos_do_html(documento.read_text(encoding="utf-8"))
    return "pdf", blocos_do_pdf(documento)


def run(refazer: bool = False) -> dict[str, int]:
    contagem: Counter[str] = Counter()
    for artigo in ler_jsonl(ARQUIVO_GOLD):
        documento = localizar_documento(artigo)
        if documento is None:
            contagem["sem_documento"] += 1
            continue
        destino = caminho_secoes(documento)
        if destino.exists() and not refazer:
            contagem["ja_tinha"] += 1
            continue
        try:
            fonte, blocos = extrair(documento)
        except Exception:
            logger.exception("Extracao de %s falhou -- seguindo", documento.name)
            contagem["falha"] += 1
            continue
        if not blocos:
            logger.warning("%s: nenhum paragrafo extraido -- so o abstract vai para a busca",
                           documento.name)
        destino.write_text(json.dumps(
            {"arxiv_id": artigo["arxiv_id"], "fonte": fonte, "blocos": blocos},
            ensure_ascii=False, indent=1), encoding="utf-8")
        contagem[fonte] += 1
    logger.info("Texto: %s", dict(contagem))
    return dict(contagem)


if __name__ == "__main__":
    configurar_log()
    run()
