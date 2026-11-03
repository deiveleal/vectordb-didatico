"""Testes das partes que nao precisam de rede nem de modelo.

O HTML abaixo imita o do arXiv (mesmas classes e ids), reduzido ao minimo.
"""

import datetime as dt

import pytest

from didatico.etapa2_silver import juntar, ler_listagem, ler_pagina_artigo
from didatico.etapa3_gold import classificar_tema, e_artigo_novo
from didatico.etapa5_texto import blocos_do_html, limpar
from didatico.etapa6_chunks import frases
from didatico.traducao import aplicar_glossario, detectar_idioma
from didatico.util import normalizar_id

LISTAGEM = """
<h3>Showing new listings for Monday, 2 November 2026</h3>
<dl id='articles'><h3>New submissions (showing 1 of 1 entries)</h3>
  <dt><a href ="/abs/2610.00001" title="Abstract" id="2610.00001">arXiv:2610.00001</a>
      [<a href="/pdf/2610.00001" title="Download PDF">pdf</a>,
       <a href="https://arxiv.org/html/2610.00001v1" title="View HTML">html</a>]</dt>
  <dd><div class='meta'>
    <div class='list-title mathjax'><span class='descriptor'>Title:</span>
      Retrieval-Augmented
      Agents</div>
    <div class='list-authors'><a href="#">Ana Silva</a>, <a href="#">Bo Li</a></div>
    <div class='list-subjects'><span class='descriptor'>Subjects:</span>
      <span class="primary-subject">Computation and Language (cs.CL)</span>; Machine Learning (cs.LG)</div>
    <p class='mathjax'>We study RAG
      for agents.</p>
  </div></dd>
</dl>
<dl id='articles'><h3>Replacement submissions (showing 1 of 1 entries)</h3>
  <dt><a href ="/abs/2501.00002" title="Abstract" id="2501.00002">arXiv:2501.00002</a></dt>
  <dd><div class='meta'>
    <div class='list-title mathjax'><span class='descriptor'>Title:</span> Old LLM paper</div>
    <div class='list-subjects'><span class='descriptor'>Subjects:</span> Artificial Intelligence (cs.AI)</div>
    <p class='mathjax'>An old paper about LLMs.</p>
  </div></dd>
</dl>
"""

PAGINA_ABS = """
<meta name="citation_title" content="Reward Hacking in Agents" />
<meta name="citation_arxiv_id" content="2609.28614" />
<meta name="citation_abstract" content="Agentic   systems hack rewards." />
<div class="authors"><span class="descriptor">Authors:</span><a href="#">Yue Huang</a></div>
<td class="tablecell subjects"><span class="primary-subject">Computation and Language (cs.CL)</span></td>
<div class="submission-history"><h2>Submission history</h2> From: Yue Huang <br/>
  <strong>[v1]</strong> Wed, 23 Sep 2026 17:54:01 UTC (10,605 KB)<br/>
  <strong>[v2]</strong> Fri, 2 Oct 2026 09:00:00 UTC (10,700 KB)<br/></div>
"""

ARTIGO_HTML = """
<article class="ltx_document">
 <section class="ltx_section" id="S1"><h2 class="ltx_title ltx_title_section">1 Introduction</h2>
  <div class="ltx_para"><p class="ltx_p">Agents use tools . We measure
    <math alttext="x^2"><mi>x</mi></math> and more things here
    <span class="ltx_note">a footnote that must go</span> today.</p></div>
  <section class="ltx_subsection"><h3 class="ltx_title">1.1 Setup</h3>
   <div class="ltx_para"><p class="ltx_p">The setup has five words or more.</p></div></section>
  <figure class="ltx_figure"><p class="ltx_p">text inside the figure is not a paragraph</p>
   <figcaption>Figure 1: The pipeline overview diagram.</figcaption></figure>
 </section>
 <section class="ltx_section" id="S2"><h2 class="ltx_title">Acknowledgments</h2>
  <div class="ltx_para"><p class="ltx_p">We thank everybody who helped us a lot.</p></div></section>
 <section class="ltx_bibliography" id="bib"><h2>References</h2><p class="ltx_p">Smith 2020 a b c d e.</p></section>
 <section class="ltx_appendix" id="A1"><h2>Appendix A Prompts</h2><p class="ltx_p">Long prompt text here, ignored.</p></section>
</article>
"""


def test_listagem_separa_novos_de_replacement():
    novo, antigo = ler_listagem(LISTAGEM, "cs.CL")
    assert novo["arxiv_id"] == "2610.00001"
    assert novo["titulo"] == "Retrieval-Augmented Agents"
    assert novo["autores"] == ["Ana Silva", "Bo Li"]
    assert novo["categorias"] == ["cs.CL", "cs.LG"]
    assert novo["abstract"] == "We study RAG for agents."
    assert (novo["tipo_anuncio"], novo["n_versoes"], novo["data_v1"]) == ("new", 1, "2026-11-02")
    assert novo["tem_html"] is True
    assert (antigo["tipo_anuncio"], antigo["n_versoes"], antigo["data_v1"]) == ("replacement", None, None)


def test_pagina_abs_le_historico_de_versoes():
    artigo = ler_pagina_artigo(PAGINA_ABS)
    assert artigo["arxiv_id"] == "2609.28614"
    assert artigo["n_versoes"] == 2
    assert artigo["data_v1"] == "2026-09-23"
    assert artigo["abstract"] == "Agentic systems hack rewards."


def test_juntar_remove_duplicata_entre_categorias():
    a, _ = ler_listagem(LISTAGEM, "cs.CL")
    b, _ = ler_listagem(LISTAGEM, "cs.LG")
    (unico,) = juntar([a, b])
    assert unico["listado_em"] == ["cs.CL", "cs.LG"]


@pytest.mark.parametrize("texto, tema", [
    ("A retrieval-augmented generation system with LLMs", "rag"),
    ("Agentic workflows for science", "agents"),
    ("Multi-agent LLM debate", "agents"),
    ("Agents in classic reinforcement learning", None),   # agente sem LLM: fora
    ("Scaling laws of large language models", "llm"),
    ("Protein folding with diffusion", None),
])
def test_classificar_tema(texto, tema):
    resultado = classificar_tema(texto)
    assert (resultado[0] if resultado else None) == tema


def test_artigo_novo():
    hoje = dt.date(2026, 11, 2)
    base = {"tipo_anuncio": "new", "n_versoes": 1}
    assert e_artigo_novo({**base, "data_v1": "2026-10-30"}, hoje)
    assert not e_artigo_novo({**base, "data_v1": "2026-10-01"}, hoje)          # velho
    assert not e_artigo_novo({"tipo_anuncio": "replacement", "n_versoes": None,
                              "data_v1": None}, hoje)
    assert e_artigo_novo({"tipo_anuncio": "link", "n_versoes": 3,
                          "data_v1": "2020-01-01"}, hoje)                     # link entra


def test_html_secoes_formulas_e_descartes():
    blocos = blocos_do_html(ARTIGO_HTML)
    secoes = [b["secao"] for b in blocos]
    assert secoes == ["1 Introduction", "1 Introduction › 1.1 Setup", "Figures and tables"]
    intro = blocos[0]["texto"]
    assert "x^2" in intro and "footnote" not in intro and "tools." in intro
    assert blocos[2]["texto"] == "Figure 1: The pipeline overview diagram."
    texto_todo = " ".join(b["texto"] for b in blocos)
    assert "thank" not in texto_todo and "Smith" not in texto_todo and "prompt" not in texto_todo


def test_frases_nao_quebra_em_abreviacao():
    assert frases("As shown by Lu et al. 2024, it works. See Fig. 3 for more. Done.") == [
        "As shown by Lu et al. 2024, it works.", "See Fig. 3 for more.", "Done."]


def test_limpar_pontuacao():
    assert limpar("word , and ( x ) .") == "word, and (x)."


@pytest.mark.parametrize("entrada", [
    "https://arxiv.org/abs/2609.28614", "https://arxiv.org/pdf/2609.28614v2",
    "arxiv.org/html/2609.28614v1", "2609.28614"])
def test_normalizar_id(entrada):
    assert normalizar_id(entrada) == "2609.28614"


def test_normalizar_id_invalido():
    with pytest.raises(ValueError):
        normalizar_id("https://example.com/artigo")


@pytest.mark.parametrize("pergunta, idioma", [
    ("memória de longo prazo para agentes", "pt"),
    ("como funciona RAG", "pt"),
    ("reward hacking in autonomous research agents", "en"),
    ("LoRA rank allocation", "en"),
])
def test_detectar_idioma(pergunta, idioma):
    assert detectar_idioma(pergunta) == idioma


def test_glossario():
    assert aplicar_glossario("checagem de fatos com geração aumentada por recuperação") == \
        "fact checking com retrieval-augmented generation"
