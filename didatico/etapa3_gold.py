"""Etapa 3 -- gold: as regras de negocio sobre a silver.

1. So artigo novo: v1, anunciado ate JANELA_ARTIGO_NOVO_DIAS dias atras.
   Fica de fora o bloco "Replacement submissions" da listagem (artigo antigo
   com versao nova). Excecao: artigo pedido por link entra mesmo antigo ou
   revisado -- quem passou o link quer aquele artigo.
2. So os temas de interesse, por palavras em titulo + abstract, na ordem de
   prioridade RAG > Agents > LLM: artigo de RAG que tambem fala de LLM fica so
   em RAG. Sem tema nenhum, fica de fora (inclusive o pedido por link: ele
   precisa de um tema para saber em que collection entrar).

E a mesma classificacao do projeto original (src/load_gold.py), copiada de
proposito: e a regra de negocio, e o resto do pipeline depende dela.

Upsert por arxiv_id: um artigo de dias anteriores continua na gold mesmo que
ja tenha saido da janela de 7 dias; o mesmo artigo visto de novo atualiza a
linha, nao duplica.

    data/gold/artigos.jsonl

Uso: python -m didatico.etapa3_gold
"""

import datetime as dt
import logging
import re
from collections import Counter

from didatico import config
from didatico.etapa2_silver import ARQUIVO as ARQUIVO_SILVER
from didatico.util import configurar_log, gravar_jsonl, ler_jsonl

logger = logging.getLogger(__name__)

ARQUIVO = config.GOLD / "artigos.jsonl"

# Siglas em maiusculas sao comparadas com caixa (RAG, LLM, GPT); palavras, sem
# caixa -- (?i:...) liga o ignore-case so no trecho.
TERMOS_LLM = re.compile(
    r"(?i:\blarge language models?\b|\blanguage models?\b|\bfoundation models?\b"
    r"|\bvision[- ]language models?\b|\bchatgpt\b)"
    r"|\bL?LMs?\b|\bVLMs?\b|\bGPT[- ]?\d*|\bLLaMA\b"
)
TERMOS_RAG = re.compile(r"(?i:retrieval[- ]augment\w*|\bgraphrag\b)|\bRAG\b")
TERMOS_AGENTIC = re.compile(r"(?i:\bagentic\b)")
TERMOS_AGENTE = re.compile(r"(?i:\bagents?\b|\bmulti[- ]agent\b)")


def classificar_tema(texto: str) -> tuple[str, str] | None:
    """(tema, trecho que decidiu) ou None se o artigo nao interessa.

    "agentic" basta para Agents; "agent(s)" so conta junto com um termo de
    LLM -- sozinho, em cs.LG, quase sempre e aprendizado por reforco classico.
    """
    if m := TERMOS_RAG.search(texto):
        return "rag", m.group(0)
    if m := TERMOS_AGENTIC.search(texto):
        return "agents", m.group(0)
    llm = TERMOS_LLM.search(texto)
    if (m := TERMOS_AGENTE.search(texto)) and llm:
        return "agents", f"{m.group(0)} + {llm.group(0)}"
    if llm:
        return "llm", llm.group(0)
    return None


def e_artigo_novo(artigo: dict, hoje: dt.date) -> bool:
    if artigo["tipo_anuncio"] == "link":
        return True
    if artigo["n_versoes"] != 1 or not artigo["data_v1"]:
        return False
    idade = (hoje - dt.date.fromisoformat(artigo["data_v1"])).days
    return 0 <= idade <= config.JANELA_ARTIGO_NOVO_DIAS


def run(hoje: dt.date | None = None) -> dict[str, int]:
    hoje = hoje or dt.date.today()
    silver = ler_jsonl(ARQUIVO_SILVER)
    novos = [a for a in silver if e_artigo_novo(a, hoje)]

    selecionados = []
    for artigo in novos:
        if (classificacao := classificar_tema(f"{artigo['titulo']} {artigo['abstract']}")) is None:
            if artigo["tipo_anuncio"] == "link":
                logger.warning("%s (pedido por link) nao fala de RAG, agentes nem LLM -- fora",
                               artigo["arxiv_id"])
            continue
        tema, evidencia = classificacao
        selecionados.append({
            **artigo,
            "tema": tema,
            "evidencia": evidencia,   # o trecho que decidiu o tema, para auditar
            "url": f"{config.SITE}/abs/{artigo['arxiv_id']}",
            "pdf_url": f"{config.SITE}/pdf/{artigo['arxiv_id']}",
        })

    # upsert: o que ja estava na gold fica; o que veio agora entra ou substitui
    gold = {a["arxiv_id"]: a for a in ler_jsonl(ARQUIVO)}
    inseridos = sum(1 for a in selecionados if a["arxiv_id"] not in gold)
    gold.update({a["arxiv_id"]: a for a in selecionados})
    gravar_jsonl(ARQUIVO, sorted(gold.values(), key=lambda a: a["arxiv_id"]))

    por_tema = Counter(a["tema"] for a in selecionados)
    logger.info("Gold: silver %d -> novos %d -> de interesse %d %s | %d inseridos, "
                "%d atualizados, %d na gold", len(silver), len(novos), len(selecionados),
                dict(por_tema), inseridos, len(selecionados) - inseridos, len(gold))
    return dict(por_tema)


if __name__ == "__main__":
    configurar_log()
    run()
