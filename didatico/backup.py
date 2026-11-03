"""Backup: um zip com gold, chunks e o indice do Qdrant, mantendo os ultimos 7.

O bronze nao entra: e grande e da para baixar de novo. O que custa caro
refazer e o que veio depois -- documentos baixados com 3 s de intervalo,
embeddings calculados. (Os documentos tambem nao entram, para o zip ficar
pequeno; acrescente-os em PASTAS se quiser.)

O indice entra de um de dois jeitos:
- modo local: a pasta data/qdrant e copiada. Com o indice fechado, a copia e
  um snapshot valido -- por isso o pipeline roda o backup DEPOIS de fechar;
- servidor (QDRANT_URL, no docker compose): o Qdrant gera um snapshot de cada
  collection, o backup baixa para dentro do zip (qdrant_snapshots/<tema>.snapshot)
  e apaga o snapshot do servidor. Para restaurar: POST
  /collections/<tema>/snapshots/upload no Qdrant.

    data/backup/AAAA-MM-DD_HHMMSS.zip

Uso: python -m didatico.backup
"""

import datetime as dt
import logging
import zipfile

import requests

from didatico import config
from didatico.etapa7_indice import abrir_qdrant
from didatico.util import configurar_log

logger = logging.getLogger(__name__)

PASTAS = (config.GOLD, config.CHUNKS) + (() if config.QDRANT_URL else (config.QDRANT,))
ARQUIVOS = (config.DADOS / "indice_estado.json", config.DADOS / "indice_estado_servidor.json")


def _snapshots_do_servidor(z: zipfile.ZipFile, base_url: str) -> None:
    qdrant = abrir_qdrant()
    try:
        for tema in config.TEMAS:
            if not qdrant.collection_exists(tema):
                continue
            snapshot = qdrant.create_snapshot(tema)
            if snapshot is None:
                raise RuntimeError(f"o Qdrant nao gerou snapshot de {tema}")
            url = f"{base_url.rstrip('/')}/collections/{tema}/snapshots/{snapshot.name}"
            with requests.get(url, stream=True, timeout=300) as resposta:
                resposta.raise_for_status()
                with z.open(f"qdrant_snapshots/{tema}.snapshot", "w") as destino:
                    for pedaco in resposta.iter_content(1 << 20):
                        destino.write(pedaco)
            qdrant.delete_snapshot(tema, snapshot.name)
    finally:
        qdrant.close()


def run() -> str:
    config.BACKUP.mkdir(parents=True, exist_ok=True)
    destino = config.BACKUP / f"{dt.datetime.now():%Y-%m-%d_%H%M%S}.zip"
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for pasta in PASTAS:
            for arquivo in sorted(pasta.rglob("*")):
                if arquivo.is_file() and arquivo.name != ".lock":
                    z.write(arquivo, arquivo.relative_to(config.DADOS))
        for arquivo in ARQUIVOS:
            if arquivo.exists():
                z.write(arquivo, arquivo.relative_to(config.DADOS))
        if config.QDRANT_URL:
            _snapshots_do_servidor(z, config.QDRANT_URL)

    antigos = sorted(config.BACKUP.glob("*.zip"))[:-config.BACKUPS_MANTIDOS]
    for antigo in antigos:
        antigo.unlink()
    logger.info("Backup: %s (%d KB), %d antigos apagados",
                destino.name, destino.stat().st_size // 1024, len(antigos))
    return str(destino)


if __name__ == "__main__":
    configurar_log()
    run()
