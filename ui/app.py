"""Interface (Streamlit): buscar artigos e inserir artigos novos.

So conversa com a API por HTTP -- nao importa nada de didatico/, como
qualquer outro cliente faria. E a seta "UI -> API" do desenho de arquitetura.

Duas abas:
- Buscar: a pergunta vai para GET /search, e cada artigo volta com os
  trechos que casaram. Sem LLM: o que aparece e exatamente o que a busca
  recuperou, com o score de cada trecho -- da para ver o efeito do rerank e
  da traducao mudando as opcoes na barra lateral.
- Inserir artigo: link do arXiv ou PDF. A API responde na hora com o id da
  tarefa (POST /ingestoes/...) e a interface pergunta GET /ingestoes/<id> a
  cada segundo, mostrando a etapa do pipeline em que o artigo esta.

Uso (da raiz do projeto, para valer o .streamlit/config.toml):
    uv run streamlit run ui/app.py
    API_URL=http://outro-host:8000 uv run streamlit run ui/app.py

API_URL e o endereco que ESTE processo usa para falar com a API. No docker
compose ele e interno (http://api:8000), e o navegador nao o enxerga: os
links que o navegador abre (o PDF enviado, servido pela API) usam
API_URL_PUBLICA (padrao: o mesmo que API_URL).
"""

import os
import time

import requests
import streamlit as st

API_PADRAO = os.environ.get("API_URL", "http://localhost:8000")
API_PUBLICA = os.environ.get("API_URL_PUBLICA", "").rstrip("/")
TEMAS = {"Todos": None, "RAG": "rag", "Agents": "agents", "LLM": "llm"}
TEMAS_INSERCAO = {"Automático (palavras do título e do abstract)": None,
                  "RAG": "rag", "Agents": "agents", "LLM": "llm"}
CORES_TEMA = {"rag": "blue", "agents": "green", "llm": "violet"}
EXEMPLOS = [
    "memória de longo prazo para agentes",
    "checagem de fatos com geração aumentada por recuperação",
    "reward hacking in autonomous research agents",
    "LoRA rank allocation",
]
ICONE_STATUS = {"na_fila": "⏳", "rodando": "⚙️", "concluida": "✅", "falhou": "❌"}
LIMITE_ESPERA_S = 600

st.set_page_config(page_title="arXiv Radar didático", page_icon="🔎", layout="wide")


# --- Cliente HTTP --------------------------------------------------------------

class ErroDaApi(Exception):
    pass


def chamar(metodo: str, caminho: str, timeout: float = 120, **kwargs) -> dict | list:
    """Uma chamada a API. Erro de rede ou HTTP vira ErroDaApi com mensagem legivel."""
    try:
        resposta = requests.request(metodo, f"{api_url}{caminho}", timeout=timeout, **kwargs)
    except requests.RequestException as erro:
        raise ErroDaApi(f"API fora do ar em {api_url} ({type(erro).__name__})") from erro
    if resposta.status_code >= 400:
        try:
            detalhe = resposta.json().get("detail", resposta.text)
        except ValueError:
            detalhe = resposta.text
        raise ErroDaApi(f"HTTP {resposta.status_code}: {detalhe}")
    return resposta.json()


def criar_tarefa(caminho: str, **kwargs) -> str:
    """POST de uma insercao; devolve o id da tarefa para acompanhar."""
    tarefa = chamar("POST", caminho, **kwargs)
    assert isinstance(tarefa, dict)
    return tarefa["id"]


def link_absoluto(url: str) -> str:
    """PDF enviado sem id do arXiv tem url relativa (/artigos/<id>/documento): servida pela API."""
    return f"{API_PUBLICA or api_url}{url}" if url.startswith("/") else url


# --- Barra lateral ---------------------------------------------------------------

with st.sidebar:
    st.header("API")
    api_url = st.text_input("Endereço da API", value=API_PADRAO).rstrip("/")
    try:
        saude = chamar("GET", "/health", timeout=10)
        assert isinstance(saude, dict)
        icone = "✅" if saude["status"] == "ok" else "⚠️"
        st.markdown(f"**Status:** {icone} {saude['status']}")
        st.caption(f"Qdrant: {saude['qdrant']}")
        if saude["trechos_por_tema"]:
            st.caption("Trechos indexados: " + " · ".join(
                f"{t} {n:,}".replace(",", ".") for t, n in saude["trechos_por_tema"].items()))
        if saude["ingestoes_na_fila"]:
            st.caption(f"Inserções na fila: {saude['ingestoes_na_fila']}")
    except ErroDaApi as erro:
        st.error(str(erro))
    st.button("Atualizar status", use_container_width=True)

    st.header("Busca")
    st.selectbox("Tema", list(TEMAS), key="tema")
    st.slider("Artigos (k)", 1, 10, 5, key="k")
    st.slider("Trechos por artigo", 1, 3, 2, key="trechos")
    st.toggle("Reranking (cross-encoder)", value=False, key="rerank",
              help="Reordena os candidatos lendo pergunta e trecho juntos: mais preciso, mais lento.")
    st.toggle("Traduzir pergunta para inglês", value=True, key="traduzir",
              help="O modelo de embedding só entende inglês. Desligue para ver a diferença.")
    st.toggle("Comparar com e sem reranking", value=False, key="comparar")


# --- Aba Buscar ------------------------------------------------------------------

def buscar(pergunta: str, rerank: bool) -> tuple[dict, float]:
    params = {"pergunta": pergunta, "k": st.session_state.k,
              "trechos_por_artigo": st.session_state.trechos,
              "rerank": str(rerank).lower(), "traduzir": str(st.session_state.traduzir).lower()}
    if TEMAS[st.session_state.tema]:
        params["tema"] = TEMAS[st.session_state.tema]
    inicio = time.monotonic()
    resposta = chamar("GET", "/search", params=params)
    assert isinstance(resposta, dict)
    return resposta, time.monotonic() - inicio


def mostrar_cabecalho(busca: dict, segundos: float) -> None:
    partes = []
    if busca["traduzida"]:
        partes.append(f"traduzida → **{busca['pergunta_busca']}**")
    elif busca["idioma"] == "en":
        partes.append("em inglês, sem tradução")
    else:
        partes.append("buscada como veio, sem tradução")
    partes.append("reranqueada" if busca["reranqueada"] else "ordem por vetor (cosseno)")
    partes.append(f"{len(busca['resultados'])} artigos em {segundos:.1f}s")
    st.caption(" · ".join(partes))


def mostrar_leitura(arxiv_id: str, chave: str) -> None:
    """Os trechos do artigo na ordem do texto (GET /artigos/<id>), sob demanda."""
    leituras = st.session_state.setdefault("leituras", {})
    secao = st.text_input("Filtrar por seção (opcional)", key=f"secao_{chave}",
                          placeholder="ex.: result, method, conclusion")
    if st.button("Carregar trechos", key=f"ler_{chave}"):
        params: dict = {"max_trechos": 20}
        if secao.strip():
            params["secao"] = secao.strip()
        try:
            leituras[chave] = chamar("GET", f"/artigos/{arxiv_id}", params=params)
        except ErroDaApi as erro:
            leituras[chave] = {"erro": str(erro)}
    if (leitura := leituras.get(chave)) is None:
        return
    if "erro" in leitura:
        st.error(leitura["erro"])
        return
    st.caption(f"{len(leitura['trechos'])} de {leitura['total_trechos']} trechos indexados")
    for t in leitura["trechos"]:
        st.markdown(f"**#{t['ordem']} · {t['secao'] or '—'}**")
        st.write(t["texto"])


def mostrar_artigo(pos: int, r: dict, compacto: bool, prefixo: str) -> None:
    cor = CORES_TEMA.get(r["tema"], "gray")
    url = link_absoluto(r["url"])
    local = r["arxiv_id"].startswith("local-")
    with st.container(border=True):
        st.markdown(f"**{pos}. [{r['titulo']}]({url})**")
        links = (f"[PDF enviado]({url})" if local else
                 f"[arXiv]({url}) · [PDF](https://arxiv.org/pdf/{r['arxiv_id']})")
        st.markdown(f":{cor}[{r['tema']}] · `{r['arxiv_id']}` · {r['data_v1'] or '—'} · "
                    f"score **{r['score']:.3f}** · {links}")
        if compacto:
            return
        with st.expander(f"Trechos que casaram ({len(r['trechos'])})", expanded=pos == 1):
            for t in r["trechos"]:
                st.markdown(f"**{t['secao'] or '—'}** · score {t['score']:.3f}")
                st.write(t["texto"])
        with st.expander("Ler o artigo (trechos na ordem do texto)"):
            mostrar_leitura(r["arxiv_id"], f"{prefixo}_{r['arxiv_id']}")


def mostrar_busca(busca: dict, segundos: float, compacto: bool = False, prefixo: str = "b") -> None:
    mostrar_cabecalho(busca, segundos)
    if not busca["resultados"]:
        st.info("Nenhum artigo encontrado. O índice tem artigos? Veja o status na barra lateral.")
    for i, r in enumerate(busca["resultados"], 1):
        mostrar_artigo(i, r, compacto, prefixo)


def aba_buscar() -> None:
    with st.form("busca"):
        pergunta = st.text_input("Pergunta", placeholder=EXEMPLOS[0], label_visibility="collapsed")
        enviar = st.form_submit_button("Buscar", type="primary")
    st.caption("Exemplos: " + " · ".join(f"_{e}_" for e in EXEMPLOS))

    if enviar:
        if not pergunta.strip():
            st.warning("Escreva uma pergunta.")
            return
        st.session_state.leituras = {}
        try:
            with st.spinner("Buscando..."):
                if st.session_state.comparar:
                    st.session_state.resultado_busca = {"comparar": [buscar(pergunta, True),
                                                           buscar(pergunta, False)]}
                else:
                    st.session_state.resultado_busca = {"unica": buscar(pergunta, st.session_state.rerank)}
        except ErroDaApi as erro:
            st.session_state.resultado_busca = None
            st.error(str(erro))

    # o resultado fica na sessao: os botoes "Carregar trechos" recarregam a
    # pagina, e sem isso a busca sumiria
    if not (guardada := st.session_state.get("resultado_busca")):
        return
    if "comparar" in guardada:
        colunas = st.columns(2)
        for coluna, titulo, (busca, segundos), prefixo in zip(
                colunas, ("Com reranking", "Sem reranking (vetor)"), guardada["comparar"],
                ("r", "v")):
            with coluna:
                st.subheader(titulo)
                mostrar_busca(busca, segundos, compacto=True, prefixo=prefixo)
    else:
        busca, segundos = guardada["unica"]
        mostrar_busca(busca, segundos)
        with st.expander("Resposta completa da API (JSON)"):
            st.json(busca)


# --- Aba Inserir -----------------------------------------------------------------

def acompanhar(ids: list[str]) -> None:
    """Pergunta o status de cada tarefa a cada segundo, ate todas terminarem."""
    caixas = {i: st.empty() for i in ids}
    inicio = time.monotonic()
    pendentes = set(ids)
    while pendentes and time.monotonic() - inicio < LIMITE_ESPERA_S:
        for tarefa_id in list(pendentes):
            try:
                tarefa = chamar("GET", f"/ingestoes/{tarefa_id}", timeout=10)
            except ErroDaApi as erro:
                caixas[tarefa_id].error(f"{tarefa_id}: {erro}")
                pendentes.discard(tarefa_id)
                continue
            assert isinstance(tarefa, dict)
            with caixas[tarefa_id].container():
                mostrar_tarefa(tarefa)
            if tarefa["status"] in ("concluida", "falhou"):
                pendentes.discard(tarefa_id)
        if pendentes:
            time.sleep(1)
    if pendentes:
        st.warning("Ainda rodando -- acompanhe em 'Inserções recentes' abaixo.")


def mostrar_tarefa(tarefa: dict) -> None:
    icone = ICONE_STATUS.get(tarefa["status"], "•")
    titulo = f"{icone} **{tarefa['entrada']}**"
    if tarefa["status"] in ("na_fila", "rodando"):
        st.info(f"{titulo} — {tarefa['etapa'] or tarefa['status'].replace('_', ' ')}")
    elif tarefa["status"] == "falhou":
        st.error(f"{titulo} — {tarefa['erro']}")
    else:
        r = tarefa["resultado"]
        st.success(f"{titulo} — indexado como `{r['arxiv_id']}`")
        st.markdown(f"[{r['titulo']}]({link_absoluto(r['url'])}) · "
                    f":{CORES_TEMA.get(r['tema'], 'gray')}[{r['tema']}] "
                    f"({r['evidencia']}) · {r['chunks']} trechos · texto do {(r['fonte'] or '?').upper()}")


def aba_inserir() -> None:
    st.markdown("Insira um artigo para ele entrar na busca. Ele passa pelas mesmas etapas "
                "do pipeline: bronze → silver → gold → documento → texto → chunks → índice.")
    origem = st.radio("Origem", ["Link do arXiv", "Arquivo PDF"], horizontal=True)
    with st.form("insercao", clear_on_submit=True):
        if origem == "Link do arXiv":
            links = st.text_area("Links ou ids, um por linha",
                                 placeholder="https://arxiv.org/abs/2609.28614\n2609.28653")
            arquivos, titulo = [], ""
        else:
            arquivos = st.file_uploader("PDF(s), até 50 MB cada", type=["pdf"],
                                        accept_multiple_files=True) or []
            titulo = st.text_input("Título (opcional, só com um arquivo)",
                                   help="Sem título, a API usa a maior fonte da 1ª página.")
            links = ""
        tema_rotulo = st.selectbox("Tema", list(TEMAS_INSERCAO),
                                   help="Automático usa a mesma regra da gold; artigo sem as "
                                        "palavras-chave fica de fora. Escolher o tema inclui mesmo assim.")
        enviar = st.form_submit_button("Inserir", type="primary")

    if enviar:
        tema = TEMAS_INSERCAO[tema_rotulo]
        ids: list[str] = []
        try:
            for link in (l.strip() for l in links.splitlines()):
                if link:
                    ids.append(criar_tarefa("/ingestoes/link", json={"link": link, "tema": tema}))
            for arquivo in arquivos:
                dados = {"tema": tema} if tema else {}
                if titulo.strip() and len(arquivos) == 1:
                    dados["titulo"] = titulo.strip()
                ids.append(criar_tarefa("/ingestoes/pdf", data=dados, files={
                    "arquivo": (arquivo.name, arquivo.getvalue(), "application/pdf")}))
        except ErroDaApi as erro:
            st.error(str(erro))
        if not ids and not links.strip() and not arquivos:
            st.warning("Informe um link ou escolha um PDF.")
        if ids:
            acompanhar(ids)

    st.divider()
    st.subheader("Inserções recentes")
    st.button("Atualizar lista")
    try:
        tarefas = chamar("GET", "/ingestoes", params={"limite": 20}, timeout=10)
    except ErroDaApi as erro:
        st.error(str(erro))
        return
    if not tarefas:
        st.caption("Nenhuma inserção desde que a API subiu.")
    for tarefa in tarefas:
        assert isinstance(tarefa, dict)
        mostrar_tarefa(tarefa)


# --- Pagina ----------------------------------------------------------------------

st.title("🔎 arXiv Radar · versão didática")
st.caption("Artigos de RAG, Agents e LLM do arXiv. Busca semântica sem LLM: "
           "o que aparece é exatamente o que foi recuperado.")
aba_busca, aba_insercao = st.tabs(["🔎 Buscar", "➕ Inserir artigo"])
with aba_busca:
    aba_buscar()
with aba_insercao:
    aba_inserir()
