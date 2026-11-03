"""Etapa 6 -- chunks: paragrafos de cada artigo -> trechos para o embedding.

Por que quebrar? O modelo de embedding le no maximo 512 tokens e devolve UM
vetor para o texto todo. Um artigo inteiro num vetor so vira uma media
borrada; trechos de ~350 tokens deixam a busca achar o paragrafo que fala da
pergunta.

Estrategia (a mesma do original, simplificada):

- chunk 0 e o abstract: o melhor resumo, e o que mais casa com busca generica;
- o corpo e empacotado por secao: paragrafos juntados ate ALVO_TOKENS, nunca
  atravessando secao de primeiro nivel e nunca cortando frase. Paragrafo maior
  que o alvo e quebrado por frase;
- entre dois chunks da mesma secao repete-se a ultima frase (se for curta),
  para o trecho nao comecar sem contexto;
- legendas de figura e tabela formam a secao "Figures and tables".

Cada chunk guarda dois textos:
- `texto`: o trecho limpo, que a busca mostra;
- `texto_embedding`: o mesmo com "Title: ... / Section: ..." na frente, que e
  o que vira vetor -- o trecho sozinho perde o contexto de onde veio.

Tokens contados com o tokenizador DO MODELO (config.MODELO_EMBEDDING): contar
palavras erraria para mais ou para menos, e o modelo corta sem avisar.

    data/chunks/<arxiv_id>.jsonl   um chunk por linha

Uso: python -m didatico.etapa6_chunks
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from tokenizers import Tokenizer

from didatico import config
from didatico.etapa3_gold import ARQUIVO as ARQUIVO_GOLD
from didatico.etapa4_documentos import localizar_documento
from didatico.etapa5_texto import caminho_secoes
from didatico.util import configurar_log, gravar_jsonl, ler_jsonl

logger = logging.getLogger(__name__)

# Fim de frase: ponto/exclamacao/interrogacao, espaco e maiuscula ou "(".
# Nao quebra em "et al. 2024" (digito depois) nem em "Fig. 3".
FIM_DE_FRASE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
NAO_E_FIM = re.compile(r"\b(?:al|Fig|Eq|Sec|Tab|e\.g|i\.e|vs|cf)\.$")


def frases(texto: str) -> list[str]:
    pedacos = FIM_DE_FRASE.split(texto)
    resultado: list[str] = []
    for pedaco in pedacos:
        if resultado and NAO_E_FIM.search(resultado[-1]):
            resultado[-1] += " " + pedaco      # "Fig." nao termina frase
        else:
            resultado.append(pedaco)
    return [f for f in resultado if f.strip()]


@dataclass
class Trecho:
    """Chunk em construcao."""
    grupo: str
    secoes: list[str] = field(default_factory=list)
    frases: list[str] = field(default_factory=list)
    tokens: int = 0

    def adicionar(self, frase: str, tokens: int, secao: str) -> None:
        self.frases.append(frase)
        self.tokens += tokens
        if secao not in self.secoes:
            self.secoes.append(secao)


class Chunker:
    def __init__(self) -> None:
        # baixa so o tokenizer.json do modelo (centenas de KB), uma vez
        self.tokenizador = Tokenizer.from_pretrained(config.MODELO_EMBEDDING)

    def tokens(self, texto: str) -> int:
        return len(self.tokenizador.encode(texto, add_special_tokens=False).ids)

    def _quebrar_frase_longa(self, frase: str) -> list[str]:
        """Frase maior que o alvo (tabela achatada, lista sem ponto) vira
        pedacos por palavra -- raro, mas sem isso o modelo cortaria o fim."""
        pedacos: list[str] = []
        atual: list[str] = []
        for palavra in frase.split():
            if atual and self.tokens(" ".join(atual + [palavra])) > config.ALVO_TOKENS:
                pedacos.append(" ".join(atual))
                atual = []
            atual.append(palavra)
        return pedacos + [" ".join(atual)] if atual else pedacos

    def empacotar(self, blocos: list[dict]) -> list[Trecho]:
        """Junta os blocos em trechos de ate ALVO_TOKENS, sem atravessar grupo."""
        trechos: list[Trecho] = []
        atual: Trecho | None = None

        for bloco in blocos:
            if atual is None or atual.grupo != bloco["grupo"]:
                if atual and atual.frases:
                    trechos.append(atual)
                atual = Trecho(bloco["grupo"])

            for frase in frases(bloco["texto"]):
                for pedaco in ([frase] if self.tokens(frase) <= config.ALVO_TOKENS
                               else self._quebrar_frase_longa(frase)):
                    n = self.tokens(pedaco)
                    if atual.frases and atual.tokens + n > config.ALVO_TOKENS:
                        # encheu: fecha e abre outro, repetindo a ultima frase
                        ultima = atual.frases[-1]
                        trechos.append(atual)
                        atual = Trecho(bloco["grupo"])
                        t = self.tokens(ultima)
                        if t <= config.MAX_SOBREPOSICAO_TOKENS and t + n <= config.ALVO_TOKENS:
                            atual.adicionar(ultima, t, bloco["secao"])
                    atual.adicionar(pedaco, n, bloco["secao"])
        if atual and atual.frases:
            trechos.append(atual)
        return trechos

    def chunks_do_artigo(self, artigo: dict, blocos: list[dict]) -> list[dict]:
        abstract = [{"grupo": "Abstract", "secao": "Abstract", "texto": artigo["abstract"] or ""}]
        trechos = self.empacotar(abstract) + self.empacotar(blocos)   # abstract vazio: 0 trechos

        chunks = []
        for ordem, trecho in enumerate(trechos):
            # varias subsecoes no mesmo trecho: o rotulo e a secao de cima
            secao = trecho.secoes[0] if len(trecho.secoes) == 1 else trecho.grupo
            texto = " ".join(trecho.frases)
            texto_embedding = f"Title: {artigo['titulo']}\nSection: {secao}\n\n{texto}"
            n_tokens = self.tokens(texto_embedding) + 2      # + [CLS] e [SEP]
            if n_tokens > config.LIMITE_MODELO:
                logger.warning("%s#%d tem %d tokens: o modelo vai cortar o fim",
                               artigo["arxiv_id"], ordem, n_tokens)
            chunks.append({
                "chunk_id": f"{artigo['arxiv_id']}#{ordem:04d}",
                "arxiv_id": artigo["arxiv_id"],
                "tema": artigo["tema"],
                "titulo": artigo["titulo"],
                "url": artigo["url"],
                "data_v1": artigo["data_v1"],
                "secao": secao,
                "ordem": ordem,
                "texto": texto,
                "texto_embedding": texto_embedding,
                "n_tokens": n_tokens,
            })
        return chunks


def caminho_chunks(arxiv_id: str):
    return config.CHUNKS / f"{arxiv_id}.jsonl"


def run(refazer: bool = False) -> dict[str, int]:
    contagem: Counter[str] = Counter()
    chunker: Chunker | None = None          # so carrega o tokenizador se precisar
    for artigo in ler_jsonl(ARQUIVO_GOLD):
        documento = localizar_documento(artigo)
        if documento is None or not caminho_secoes(documento).exists():
            contagem["sem_texto"] += 1      # etapa 4 ou 5 ainda nao rodou para ele
            continue
        destino = caminho_chunks(artigo["arxiv_id"])
        # refaz se o texto foi extraido de novo depois dos chunks
        if (destino.exists() and not refazer
                and destino.stat().st_mtime >= caminho_secoes(documento).stat().st_mtime):
            contagem["ja_tinha"] += 1
            continue
        chunker = chunker or Chunker()
        blocos = json.loads(caminho_secoes(documento).read_text(encoding="utf-8"))["blocos"]
        chunks = chunker.chunks_do_artigo(artigo, blocos)
        gravar_jsonl(destino, chunks)
        contagem["artigos"] += 1
        contagem["chunks"] += len(chunks)
    logger.info("Chunks: %s", dict(contagem))
    return dict(contagem)


if __name__ == "__main__":
    configurar_log()
    run()
