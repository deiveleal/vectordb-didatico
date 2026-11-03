"""Traducao da PERGUNTA de busca, portugues -> ingles.

Por que traduzir? O bge-small-en-v1.5 so entende ingles: pergunta em
portugues vira um vetor longe dos trechos certos. Os artigos ja estao em
ingles, entao basta traduzir a pergunta (curta e barata), nunca o acervo.

Tres pecas, todas locais:

1. detectar o idioma: conta palavras tipicas de portugues e de ingles. Para
   perguntas de busca isso basta, e da para ler a regra inteira em 10 linhas.
2. glossario: termos tecnicos em portugues -> o termo que os artigos usam.
   Aplicado ANTES do tradutor, que verteria ao pe da letra ("geracao
   aumentada por recuperacao" -> "increased generation by recovery").
3. tradutor: o modelo pt->en do Argos Translate -- o mesmo que o
   LibreTranslate do projeto original usa por dentro --, rodado direto com
   ctranslate2 (o motor) + sentencepiece (quebra o texto em subpalavras).
   Sem PyTorch. ~70 MB, baixado uma vez para modelos/.

Se o modelo nao estiver disponivel, a busca segue com a pergunta original:
pior, mas funcionando.
"""

import logging
import re
import zipfile
from functools import cached_property
from pathlib import Path

import requests

from didatico import config

logger = logging.getLogger(__name__)

# --- 1. Idioma -----------------------------------------------------------------

PALAVRAS_PT = set("""
de da do das dos para com como que uma um uns umas na no nas nos em ao aos pela
pelo pelas pelos sobre entre sem mais menos qual quais quando onde porque não
nao é sao são ser está estao também tambem ou e o a os as se seu sua isso este
esta
""".split())
PALAVRAS_EN = set("""
the of for with how what which when where why is are be to in on by from and or
a an as at this that these those using via vs into over under between without
""".split())
ACENTOS_PT = re.compile(r"[ãõçáéíóúâêô]", re.IGNORECASE)


def detectar_idioma(texto: str) -> str:
    """'pt' ou 'en'. Empate (ex.: "LoRA rank allocation") fica em ingles:
    traduzir sem necessidade pode estragar termo tecnico."""
    palavras = re.findall(r"\w+", texto.lower())
    pt = sum(p in PALAVRAS_PT for p in palavras) + 2 * len(ACENTOS_PT.findall(texto))
    en = sum(p in PALAVRAS_EN for p in palavras)
    return "pt" if pt > en else "en"


# --- 2. Glossario --------------------------------------------------------------
# O mais especifico primeiro ("modelos de linguagem de grande porte" antes de
# "modelos de linguagem").
GLOSSARIO: list[tuple[str, str]] = [
    (r"gera[cç][aã]o aumentada (por|via|com) (recupera[cç][aã]o|busca)",
     "retrieval-augmented generation"),
    (r"recupera[cç][aã]o de informa[cç][aã]o", "information retrieval"),
    (r"(grandes modelos de linguagem|modelos? de linguagem de grande porte|"
     r"modelos? de linguagem grandes?)", "large language models"),
    (r"modelos? de linguagem (de|para) vis[aã]o", "vision-language models"),
    (r"modelos? de linguagem", "language models"),
    (r"agentes aut[oô]nomos", "autonomous agents"),
    (r"multi-?agentes?", "multi-agent"),
    (r"ag[eê]ntic[oa]s?", "agentic"),
    (r"uso de ferramentas", "tool use"),
    (r"ajuste fino", "fine-tuning"),
    (r"aprendizado por refor[cç]o", "reinforcement learning"),
    (r"cadeia de pensamento", "chain of thought"),
    (r"engenharia de prompts?", "prompt engineering"),
    (r"janela de contexto", "context window"),
    (r"base de conhecimento", "knowledge base"),
    (r"busca sem[aâ]ntica", "semantic search"),
    (r"bancos? de dados vetoriais?|bancos? vetoriais?", "vector database"),
    (r"reordena[cç][aã]o|reranqueamento", "reranking"),
    (r"checagem de fatos|verifica[cç][aã]o de fatos", "fact checking"),
    (r"alucina[cç](ão|ões)", "hallucination"),
]
_GLOSSARIO = [(re.compile(rf"\b{pt}\b", re.IGNORECASE), en) for pt, en in GLOSSARIO]


def aplicar_glossario(texto: str) -> str:
    for padrao, termo in _GLOSSARIO:
        texto = padrao.sub(termo, texto)
    return texto


# --- 3. Tradutor ---------------------------------------------------------------

class Tradutor:
    """Carrega o modelo na primeira traducao (e o baixa, se preciso)."""

    def __init__(self, pasta: Path = config.MODELOS):
        self.pasta = pasta

    @property
    def pasta_modelo(self) -> Path:
        return self.pasta / "translate-pt_en-1_9"

    def baixar_modelo(self) -> None:
        """O .argosmodel e um zip: model/ (ctranslate2), sentencepiece.model e
        uma pasta stanza/ que nao usamos."""
        if self.pasta_modelo.exists():
            return
        self.pasta.mkdir(parents=True, exist_ok=True)
        zip_ = self.pasta / "translate-pt_en.argosmodel"
        logger.info("Baixando o modelo de traducao pt->en (~70 MB, uma vez so)")
        with requests.get(config.URL_MODELO_TRADUCAO, stream=True, timeout=120) as resposta:
            resposta.raise_for_status()
            with zip_.open("wb") as arquivo:
                for pedaco in resposta.iter_content(1 << 20):
                    arquivo.write(pedaco)
        with zipfile.ZipFile(zip_) as z:
            z.extractall(self.pasta)
        zip_.unlink()

    @cached_property
    def _motor(self):
        import ctranslate2
        import sentencepiece

        self.baixar_modelo()
        quebrador = sentencepiece.SentencePieceProcessor(
            model_file=str(self.pasta_modelo / "sentencepiece.model"))
        motor = ctranslate2.Translator(str(self.pasta_modelo / "model"), device="cpu")
        return quebrador, motor

    def para_ingles(self, texto: str) -> str:
        quebrador, motor = self._motor
        # texto -> subpalavras ("▁memória", "▁de", ...) -> modelo -> subpalavras em ingles
        entrada = quebrador.encode(texto, out_type=str)
        saida = motor.translate_batch([entrada])[0].hypotheses[0]
        # "▁" marca inicio de palavra; e assim que o proprio Argos remonta o texto
        return "".join(saida).replace("▁", " ").strip()

    def preparar_pergunta(self, pergunta: str) -> tuple[str, str, bool]:
        """(pergunta em ingles, idioma detectado, se foi traduzida)."""
        idioma = detectar_idioma(pergunta)
        if idioma == "en":
            return pergunta, idioma, False
        try:
            traduzida = self.para_ingles(aplicar_glossario(pergunta))
        except Exception as erro:            # sem rede para baixar, zip corrompido...
            logger.warning("Tradutor indisponivel (%s) -- buscando com a pergunta original", erro)
            return pergunta, idioma, False
        logger.info("Pergunta traduzida: %r -> %r", pergunta, traduzida)
        return traduzida, idioma, True
