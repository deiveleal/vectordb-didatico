"""Backup: um zip com gold, chunks e o indice do Qdrant, mantendo os ultimos 7.

O bronze nao entra: e grande e da para baixar de novo. O que custa caro
refazer e o que veio depois -- documentos baixados com 3 s de intervalo,
embeddings calculados. (Os documentos tambem nao entram, para o zip ficar
pequeno; acrescente-os em PASTAS se quiser.)

O Qdrant local e uma pasta de arquivos: copiar com o indice fechado e um
snapshot valido. Por isso o pipeline roda o backup DEPOIS de fechar o indice.

    data/backup/AAAA-MM-DD_HHMMSS.zip

Uso: python -m didatico.backup
"""

import datetime as dt
import logging
import zipfile

from didatico import config
from didatico.util import configurar_log

logger = logging.getLogger(__name__)

PASTAS = (config.GOLD, config.CHUNKS, config.QDRANT)
ARQUIVOS = (config.DADOS / "indice_estado.json",)


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

    antigos = sorted(config.BACKUP.glob("*.zip"))[:-config.BACKUPS_MANTIDOS]
    for antigo in antigos:
        antigo.unlink()
    logger.info("Backup: %s (%d KB), %d antigos apagados",
                destino.name, destino.stat().st_size // 1024, len(antigos))
    return str(destino)


if __name__ == "__main__":
    configurar_log()
    run()
