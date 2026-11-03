"""Tudo o que se costuma ajustar fica aqui, num lugar so.

O projeto original espalha isso por variaveis de ambiente e pelo
docker-compose; aqui sao constantes Python, para ler de cima a baixo.
"""

import os
from pathlib import Path

# --- Pastas -----------------------------------------------------------------
# Cada camada do desenho vira uma pasta em data/. Da para abrir qualquer
# arquivo num editor e ver o que a etapa produziu.
RAIZ = Path(__file__).resolve().parents[1]
DADOS = Path(os.environ.get("DIDATICO_DADOS", RAIZ / "data"))

BRONZE = DADOS / "bronze"          # HTML cru, como veio do arXiv
SILVER = DADOS / "silver"          # um JSON por linha, um artigo por linha
GOLD = DADOS / "gold"              # so os artigos de interesse, com tema
DOCUMENTOS = DADOS / "documents"   # HTML/PDF do artigo + secoes extraidas
CHUNKS = DADOS / "chunks"          # trechos prontos para o embedding
QDRANT = DADOS / "qdrant"          # banco vetorial em modo local (arquivos)
BACKUP = DADOS / "backup"          # zips das camadas gold, chunks e qdrant
MODELOS = Path(os.environ.get("DIDATICO_MODELOS", RAIZ / "modelos"))

# --- Fonte: paginas HTML do arXiv ----------------------------------------------
# Sem API: a lista do dia e a pagina /list/<categoria>/new, a mesma que uma
# pessoa abre no navegador. Ela traz titulo, autores, categorias e abstract.
SITE = "https://arxiv.org"
CATEGORIAS = ("cs.CL", "cs.AI", "cs.LG")

# Termos de uso do arXiv: no maximo 1 requisicao a cada 3 segundos, com um
# User-Agent que identifique quem esta baixando.
INTERVALO_ENTRE_REQUISICOES = 3.0
USER_AGENT = os.environ.get(
    "DIDATICO_USER_AGENT", "vectordb-didatico/0.1 (estudo; defina DIDATICO_USER_AGENT)")

# --- Regras da gold ------------------------------------------------------------
JANELA_ARTIGO_NOVO_DIAS = 7
TEMAS = ("rag", "agents", "llm")   # um tema = uma collection no Qdrant

# --- Chunks e embeddings -------------------------------------------------------
# O tokenizador tem que ser o do modelo de embedding: e ele que decide quantos
# tokens um trecho tem, e o modelo corta sem aviso o que passa de 512.
MODELO_EMBEDDING = "BAAI/bge-small-en-v1.5"   # 384 dimensoes, roda bem em CPU
DIMENSAO = 384
LIMITE_MODELO = 512
ALVO_TOKENS = 350
MAX_SOBREPOSICAO_TOKENS = 60
# O bge pede este prefixo na PERGUNTA (nao nos trechos).
PREFIXO_CONSULTA = "Represent this sentence for searching relevant passages: "

# --- Busca ---------------------------------------------------------------------
MODELO_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"   # cross-encoder, 80 MB
MIN_CANDIDATOS_RERANK = 52
# Modelo de traducao pt -> en do Argos Translate (o mesmo que o LibreTranslate
# usa por dentro). E so um arquivo para baixar uma vez; ver traducao.py.
URL_MODELO_TRADUCAO = "https://argos-net.com/v1/translate-pt_en-1_9.argosmodel"

# --- Backup --------------------------------------------------------------------
BACKUPS_MANTIDOS = 7
