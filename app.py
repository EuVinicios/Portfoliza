from __future__ import annotations
import io, os, re, base64, json, uuid, html
from typing import Dict, Tuple, Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Gerador de Portfólios de Investimento",
    page_icon="💹",
    layout="wide"
)

# =========================
# IMPORTS OPCIONAIS
# =========================
HAS_PDFPLUMBER = False
HAS_AGGRID = False
HAS_YF = False
HAS_BCB = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    pass

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    HAS_AGGRID = True
except Exception:
    pass

try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    pass

try:
    from bcb import Expectativas as BCBExpectativas
    HAS_BCB = True
except Exception:
    pass

# =========================
# TEMA DE GRÁFICOS
# =========================
PALETA = px.colors.qualitative.Vivid
TEMPLATE = "plotly_white"

# =========================
# MOCK DEFAULT (FALLBACK)
# =========================
DEFAULT_CARTEIRAS = {
    "Conservador": {
        "rentabilidade_esperada_aa": 0.08,
        "alocacao": {
            "Renda Fixa Pós-Fixada": 0.70,
            "Renda Fixa Inflação": 0.20,
            "Crédito Privado": 0.10,
        }
    },
    "Moderado": {
        "rentabilidade_esperada_aa": 0.10,
        "alocacao": {
            "Renda Fixa Pós-Fixada": 0.40,
            "Renda Fixa Inflação": 0.25,
            "Crédito Privado": 0.15,
            "Fundos Imobiliários": 0.10,
            "Ações e Fundos de Índice": 0.10,
        }
    },
    "Arrojado": {
        "rentabilidade_esperada_aa": 0.12,
        "alocacao": {
            "Renda Fixa Pós-Fixada": 0.20,
            "Renda Fixa Inflação": 0.10,
            "Crédito Privado": 0.20,
            "Fundos Imobiliários": 0.20,
            "Ações e Fundos de Índice": 0.30,
        }
    }
}

# =========================
# HELPERS NUMÉRICOS E DE FORMATAÇÃO
# =========================
def _parse_float(txt: str, default: float=0.0) -> float:
    if txt is None:
        return default
    s = str(txt).strip().replace(".", "").replace(",", ".")
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default

def number_input_allow_blank(label: str, default: float, key: str, help: Optional[str]=None):
    # Formata o número padrão para o formato brasileiro para usar como placeholder
    placeholder = f"{default:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    val_str = st.text_input(label, value=st.session_state.get(key, placeholder), key=key, help=help)
    return _parse_float(val_str, default=0.0)

def _fmt_num_br(v: float, nd: int = 2) -> str:
    try:
        return f"{float(v):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return str(v)

def fmt_brl(v: float) -> str:
    return "R$ " + _fmt_num_br(v, 2)

def fmt_pct100_br(pct: float) -> str:
    return _fmt_num_br(pct, 2) + " %"

# =========================
# YAHOO FINANÇAS HELPERS
# =========================
YF_TICKERS = {
    "Dólar (USD/BRL)": ["USDBRL=X"], "Euro (EUR/BRL)": ["EURBRL=X"], "Ibovespa": ["^BVSP"],
    "IFIX (aprox.)": ["IFIX.SA", "^IFIX"], "S&P 500": ["^GSPC"], "Nasdaq": ["^IXIC"],
    "Bitcoin": ["BTC-USD"], "Ouro (Comex)": ["GC=F"], "Petróleo WTI": ["CL=F"], "US 10Y": ["^TNX"],
}

if HAS_YF:
    @st.cache_data(ttl=900, show_spinner=False)
    def _yf_download_cached(symbol: str, period: str="5d", interval: str="1d") -> Optional[pd.DataFrame]:
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            return df if df is not None and not df.empty else None
        except Exception:
            return None

def _yf_last_close_change(symbols: List[str]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if not HAS_YF: return None, None, None
    for s in symbols:
        df = _yf_download_cached(s)
        if df is not None and not df.empty:
            close_series = df["Close"].dropna()
            if len(close_series) >= 2:
                last, prev = float(close_series.iloc[-1]), float(close_series.iloc[-2])
                chg = (last / prev - 1.0) * 100.0
                return last, chg, s
    return None, None, None

def render_market_strip(cdi_aa: float, ipca_aa: float, selic_aa: Optional[float]=None):
    quotes = []
    for nome, syms in YF_TICKERS.items():
        px, chg, _ = _yf_last_close_change(syms)
        if px is None: continue
        val = f"R$ {_fmt_num_br(px, 4)}" if "BRL" in nome else f"US$ {_fmt_num_br(px, 0)}" if "Bitcoin" in nome else f"{_fmt_num_br(px/10, 2)}%" if "10Y" in nome else _fmt_num_br(px, 2)
        pct = "" if chg is None else (("+" if chg >= 0 else "") + _fmt_num_br(chg, 2) + "%")
        direction = "up" if chg is not None and chg >= 0 else "down" if chg is not None else "flat"
        quotes.append({"label": nome, "val": val, "pct": pct, "dir": direction})

    base_cards = [{"label": "CDI (App)", "val": f"{_fmt_num_br(cdi_aa,2)}%", "pct": "", "dir":"flat"},
                  {"label": "IPCA (App)", "val": f"{_fmt_num_br(ipca_aa,2)}%", "pct": "", "dir":"flat"}]
    if selic_aa is not None:
        base_cards.append({"label": "Selic (App)", "val": f"{_fmt_num_br(selic_aa,2)}%", "pct": "", "dir":"flat"})

    items = base_cards + quotes
    style = """<style>.tstrip-wrap{background:#0b1221;border-radius:12px;padding:8px 10px;margin:4px 0 8px;border:1px solid #182235;}.tstrip-row{display:flex;gap:10px;overflow-x:auto;white-space:nowrap;scrollbar-width:thin}.tstrip-item{display:inline-flex;align-items:baseline;gap:6px;padding:6px 10px;border-radius:999px;background:#0f172a;border:1px solid #1f2937}.tstrip-label{font-size:12px;color:#94a3b8}.tstrip-val{font-size:13px;font-weight:600;color:#e5e7eb}.tstrip-pct{font-size:12px;font-weight:600}.tstrip-pct.up{color:#16a34a}.tstrip-pct.down{color:#dc2626}.tstrip-pct.flat{color:#94a3b8}</style>"""
    chips = [f"""<div class="tstrip-item"><span class="tstrip-label">{it['label']}</span><span class="tstrip-val">{it['val']}</span><span class="tstrip-pct {it['dir']}">{it['pct']}</span></div>""" for it in items]
    html_block = style + f"""<div class="tstrip-wrap"><div class="tstrip-row">{''.join(chips)}</div></div>"""
    st.markdown("### Panorama de Mercado")
    st.markdown(html_block, unsafe_allow_html=True)

# =========================
# PDF: CARREGAR & PARSE
# =========================
def load_pdf_bytes(uploaded_file, default_path: Optional[str]) -> Tuple[Optional[bytes], str]:
    if uploaded_file: return uploaded_file.read(), "PDF carregado por upload."
    if default_path and os.path.exists(default_path):
        with open(default_path, "rb") as f: return f.read(), f"PDF carregado do caminho local."
    return None, "Nenhum PDF carregado (usando configurações padrão)."

_CLASSE_NORMALIZAR = {r"renda fixa pós.*fixada": "Renda Fixa Pós-Fixada", r"p[oó]s[\s\-]*cdi|cdi": "Renda Fixa Pós-Fixada", r"renda fixa infla[cç][aã]o|ipca\+?": "Renda Fixa Inflação", r"cr[eé]dito privado|deb[eê]ntures|cra|cri": "Crédito Privado", r"fundos imobili[aá]rios|fii": "Fundos Imobiliários", r"a[cç][oõ]es.*[íi]ndice|etf": "Ações e Fundos de Índice", r"previd[eê]ncia": "Previdência Privada"}
_PERFIS = ["Conservador", "Moderado", "Arrojado"]

def _normalizar_classe(label: str) -> Optional[str]:
    l = label.lower()
    for pat, out in _CLASSE_NORMALIZAR.items():
        if re.search(pat, l, flags=re.I): return out
    return None

@st.cache_data(show_spinner=False)
def extrair_carteiras_do_pdf_cached(pdf_bytes: bytes) -> Dict[str, Dict]:
    if not (pdf_bytes and HAS_PDFPLUMBER): return DEFAULT_CARTEIRAS
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages: text += "\n" + (page.extract_text() or "")
        
        blocos, carteiras = {}, {}
        for i, perfil in enumerate(_PERFIS):
            start = re.search(perfil, text, re.I)
            if not start: continue
            end_idx = len(text)
            for j in range(i + 1, len(_PERFIS)):
                nxt = re.search(_PERFIS[j], text, re.I)
                if nxt: end_idx = min(end_idx, nxt.start())
            blocos[perfil] = text[start.start():end_idx]

        for perfil, bloco in blocos.items():
            pairs = {}
            for line in bloco.splitlines():
                m = re.search(r"([A-Za-zÀ-ÿ \-\+\/]+?)\s+(\d{1,3})\s*%", line.strip())
                if m:
                    classe = _normalizar_classe(m.group(1).strip())
                    if classe: pairs[classe] = pairs.get(classe, 0.0) + (float(m.group(2)) / 100.0)
            if pairs:
                soma = sum(pairs.values()) or 1.0
                carteiras[perfil] = {"alocacao": {k: v / soma for k, v in pairs.items()}}
        return carteiras if carteiras else DEFAULT_CARTEIRAS
    except Exception:
        return DEFAULT_CARTEIRAS

# =========================
# FINANCE HELPERS
# =========================
def aa_to_am(taxa_aa: float) -> float: return (1 + taxa_aa)**(1/12) - 1

def calcular_projecao(v_ini, aportes_m, taxa_m, prazo_m):
    vals = [v_ini]
    for _ in range(prazo_m): vals.append((vals[-1] + aportes_m) * (1 + taxa_m))
    return vals

def criar_grafico_projecao(df, title):
    fig = px.line(df, x='Mês', y=[c for c in df.columns if c != 'Mês'], title=title, labels={'value': 'Patrimônio (R$)', 'variable': 'Cenário'}, markers=False, color_discrete_sequence=PALETA, template=TEMPLATE)
    fig.update_layout(legend_title_text='Cenários', yaxis_title='Patrimônio (R$)', xaxis_title='Meses')
    return fig

def criar_grafico_alocacao(df, title):
    if df is None or df.empty or df['Valor (R$)'].sum() <= 0: return go.Figure()
    nomes = 'Descrição' if 'Descrição' in df.columns else 'Classe de Ativo'
    fig = px.pie(df, values='Valor (R$)', names=nomes, title=title, hole=.35, color_discrete_sequence=PALETA, template=TEMPLATE)
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.02]*len(df))
    return fig

# =========================
# SESSION STATE INIT
# =========================
# Inicializa os DataFrames de portfólio no estado da sessão se ainda não existirem
for key in ['portfolio_atual', 'portfolio_personalizado']:
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame(columns=["Tipo", "Descrição", "Indexador", "Parâmetro Indexação (% a.a.)", "IR (%)", "Isento", "Rent. 12M (%)", "Rent. 6M (%)", "Alocação (%)"])

# Inicializa os estados de confirmação de exclusão
for key in ['confirming_delete_atual', 'confirming_delete_personalizado', 'asset_to_delete_atual', 'asset_to_delete_personalizado']:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================
# FOCUS HELPERS (BCB)
# =========================
@st.cache_data(ttl=3600, show_spinner="Buscando dados de mercado...")
def _fetch_focus_aa() -> dict:
    if not HAS_BCB: return {}
    try:
        em = BCBExpectativas()
        ep = em.get_endpoint("ExpectativasMercadoAnuais")
        ano = pd.Timestamp.today().year
        
        # Busca dados para o ano corrente e o próximo, para garantir que haja um resultado
        df = ep.query().filter(ep.Indicador.isin(["IPCA", "Selic"]), ep.DataReferencia >= ano).select(ep.Data, ep.DataReferencia, ep.Mediana, ep.Indicador).collect()
        if df.empty: return {}

        # Pega a estimativa mais recente para cada indicador do ano mais próximo
        df = df.sort_values("Data").drop_duplicates(subset=['Indicador', 'DataReferencia'], keep='last')
        
        ipca_df = df[df.Indicador == 'IPCA']
        selic_df = df[df.Indicador == 'Selic']

        ipca_aa = ipca_df['Mediana'].iloc[0] if not ipca_df.empty else None
        selic_aa = selic_df['Mediana'].iloc[0] if not selic_df.empty else None

        out = {}
        if ipca_aa is not None: out["ipca_aa"] = float(ipca_aa)
        if selic_aa is not None: out["selic_aa"] = float(selic_aa)
        return out
    except Exception:
        return {}

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("""<div style="display:flex;align-items:center;gap:10px;margin-bottom:-6px">
                   <div style="font-size:46px;line-height:1">📊</div>
                   <div style="font-weight:600;font-size:18px">Parâmetros do Cliente</div>
                   </div>""", unsafe_allow_html=True)
    st.markdown("---")
    nome_cliente = st.text_input("Nome do Cliente", "Cliente Exemplo")

    st.subheader("Carteiras Sugeridas (PDF)")
    pdf_upload = st.file_uploader("Anexar PDF", type=["pdf"])
    # Caminho do PDF foi removido para portabilidade. O usuário deve fazer o upload.
    pdf_bytes, pdf_msg = load_pdf_bytes(pdf_upload, None)
    st.caption(pdf_msg)

    st.subheader("Parâmetros de Mercado (a.a.)")
    # Busca dados do Focus e define valores padrão
    focus_data = _fetch_focus_aa()
    cdi_default = focus_data.get('selic_aa', 10.5)  # Selic como proxy
    ipca_default = focus_data.get('ipca_aa', 3.8)
    selic_default = focus_data.get('selic_aa', 10.5)
    if focus_data: st.caption("Valores pré-preenchidos via Boletim Focus (BCB).")

    # Inputs numéricos usando os padrões
    cdi_aa = number_input_allow_blank("CDI esperado (% a.a.)", cdi_default, key="cdi_aa", help="Usado para 'Pós CDI'")
    ipca_aa = number_input_allow_blank("IPCA esperado (% a.a.)", ipca_default, key="ipca_aa", help="Usado para 'IPCA+'")
    selic_aa = number_input_allow_blank("Selic esperada (% a.a.)", selic_default, key="selic_aa", help="Apenas para exibição.")

    carteiras_from_pdf = extrair_carteiras_do_pdf_cached(pdf_bytes) if pdf_bytes else DEFAULT_CARTEIRAS
    perfil_investimento = st.selectbox("Perfil de Investimento", list(carteiras_from_pdf.keys()))

    st.subheader("Opções da Carteira Sugerida")
    incluir_credito_privado = st.checkbox("Incluir Crédito Privado", True)
    incluir_previdencia = st.checkbox("Incluir Previdência", False)
    incluir_fundos_imobiliarios = st.checkbox("Incluir Fundos Imobiliários", True)
    incluir_acoes_indice = st.checkbox("Incluir Ações e Fundos de Índice", True)
    st.markdown("---")

    st.subheader("Projeção — Parâmetros")
    valor_inicial = number_input_allow_blank("Valor Inicial (R$)", 50000.0, key="valor_inicial_input")
    aportes_mensais = number_input_allow_blank("Aportes Mensais (R$)", 1000.0, key="aportes_mensais_input")
    prazo_meses = st.slider("Prazo (meses)", 1, 120, 60)
    meta_financeira = number_input_allow_blank("Meta a Atingir (R$)", 500000.0, key="meta_financeira_input")


# =========================
# HEADER + STRIP DE MERCADO
# =========================
st.title(f"💹 Análise de Portfólios — {nome_cliente}")
st.caption(f"Perfil: **{perfil_investimento}** | Prazo: **{prazo_meses} meses** | Investimento Inicial: **{fmt_brl(valor_inicial)}**")
render_market_strip(cdi_aa=cdi_aa, ipca_aa=ipca_aa, selic_aa=selic_aa)

# =========================
# LÓGICA DA CARTEIRA SUGERIDA
# =========================
carteira_base = carteiras_from_pdf.get(perfil_investimento, DEFAULT_CARTEIRAS["Moderado"])
aloc_sugerida = carteira_base["alocacao"].copy()

toggle_flags = {"Crédito Privado": incluir_credito_privado, "Fundos Imobiliários": incluir_fundos_imobiliarios, "Ações e Fundos de Índice": incluir_acoes_indice, "Previdência Privada": incluir_previdencia}
for classe, flag in toggle_flags.items():
    if not flag: aloc_sugerida.pop(classe, None)

tot = sum(aloc_sugerida.values()) or 1.0
aloc_sugerida = {k: v/tot for k, v in aloc_sugerida.items()}

df_sugerido = pd.DataFrame(list(aloc_sugerida.items()), columns=["Classe de Ativo", "Alocação (%)"])
df_sugerido["Alocação (%)"] = (df_sugerido["Alocação (%)"] * 100).round(2)
df_sugerido["Valor (R$)"] = (valor_inicial * df_sugerido["Alocação (%)"] / 100.0).round(2)
rent_aa_sugerida = carteira_base.get("rentabilidade_esperada_aa", 0.10)
rent_am_sugerida = aa_to_am(rent_aa_sugerida)

# =========================
# FORMULÁRIO DE PORTFÓLIO (REUTILIZÁVEL)
# =========================
TIPOS_ATIVO_BASE = ["Debêntures", "CRA", "CRI", "Tesouro Direto", "Ações", "Fundos de Índice (ETF)", "Fundos Imobiliários (FII)", "CDB", "LCA", "LCI", "Renda Fixa Pós-Fixada", "Renda Fixa Inflação", "Crédito Privado", "Previdência Privada", "Sintético", "Outro"]
INDEXADORES = ["Pós CDI", "Prefixado", "IPCA+"]

def form_portfolio(portfolio_key: str, titulo: str):
    st.subheader(titulo)
    dfp = st.session_state[portfolio_key]
    
    # Define chaves únicas para o estado de confirmação
    confirm_key = f'confirming_delete_{"atual" if "atual" in portfolio_key else "personalizado"}'
    asset_key = f'asset_to_delete_{"atual" if "atual" in portfolio_key else "personalizado"}'
    
    # Mostra a caixa de confirmação se a flag estiver ativa
    if st.session_state.get(confirm_key, False):
        st.warning(f"**Você tem certeza que deseja excluir o ativo '{st.session_state[asset_key]}'?**")
        c1, c2, _ = st.columns([1, 1, 4])
        if c1.button("Sim, excluir", key=f"confirm_del_{portfolio_key}", type="primary"):
            df_original = st.session_state[portfolio_key]
            st.session_state[portfolio_key] = df_original[df_original['Descrição'] != st.session_state[asset_key]].copy()
            st.session_state[confirm_key] = False
            st.session_state[asset_key] = None
            st.rerun()
        if c2.button("Cancelar", key=f"cancel_del_{portfolio_key}"):
            st.session_state[confirm_key] = False
            st.session_state[asset_key] = None
            st.rerun()

    with st.expander("Adicionar Novo Ativo", expanded=dfp.empty):
        c = st.columns(4)
        tipo = c[0].selectbox("Tipo", TIPOS_ATIVO_BASE, key=f"tipo_{portfolio_key}")
        desc = c[1].text_input("Descrição", key=f"desc_{portfolio_key}")
        indexador = c[2].selectbox("Indexador", INDEXADORES, key=f"idx_{portfolio_key}")
        aloc = c[3].number_input("Alocação (%)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, key=f"aloc_{portfolio_key}")

        if st.button("Adicionar Ativo", key=f"add_{portfolio_key}"):
            if desc.strip():
                # Simplificado para adicionar apenas os campos essenciais inicialmente
                novo_ativo = pd.DataFrame([{"Tipo": tipo, "Descrição": desc.strip(), "Indexador": indexador, "Alocação (%)": aloc}])
                st.session_state[portfolio_key] = pd.concat([dfp, novo_ativo], ignore_index=True)
                st.rerun()
            else:
                st.warning("A **Descrição** do ativo é obrigatória.")

    if not dfp.empty:
        soma = dfp["Alocação (%)"].sum()
        dfp["Alocação Normalizada (%)"] = (dfp["Alocação (%)"] / soma * 100.0).round(2)
        dfp["Valor (R$)"] = (valor_inicial * dfp["Alocação Normalizada (%)"] / 100.0).round(2)
        
        st.write("---")
        st.subheader("Ativos da Carteira")
        
        if HAS_AGGRID:
            gb = GridOptionsBuilder.from_dataframe(dfp)
            gb.configure_selection('single', use_checkbox=True)
            grid_response = AgGrid(dfp, gb.build(), update_mode=GridUpdateMode.SELECTION_CHANGED, key=f"grid_{portfolio_key}", fit_columns_on_grid_load=True, theme='streamlit', allow_unsafe_jscode=True)
            
            selected_rows = grid_response.get('selected_rows', [])
            if selected_rows:
                desc_selecionada = selected_rows[0]['Descrição']
                st.info(f"Ações para: **{desc_selecionada}**")
                # Botão para iniciar o processo de exclusão
                if st.button("🗑️ Excluir Ativo Selecionado", key=f"init_del_{portfolio_key}"):
                    st.session_state[confirm_key] = True
                    st.session_state[asset_key] = desc_selecionada
                    st.rerun()
        else: # Fallback se AgGrid não estiver instalado
            st.dataframe(dfp.style.format({"Valor (R$)": fmt_brl, "Alocação (%)": fmt_pct100_br, "Alocação Normalizada (%)": fmt_pct100_br}), use_container_width=True, hide_index=True)
            ativo_selecionado = st.selectbox("Selecione um ativo para excluir:", ["(Nenhum)"] + dfp["Descrição"].tolist(), key=f"fallback_sel_{portfolio_key}")
            if ativo_selecionado != "(Nenhum)":
                if st.button("🗑️ Excluir Ativo", key=f"fallback_init_del_{portfolio_key}"):
                    st.session_state[confirm_key] = True
                    st.session_state[asset_key] = ativo_selecionado
                    st.rerun()

        st.subheader("Visualização da Carteira")
        fig = criar_grafico_alocacao(dfp, f"Alocação — {titulo}")
        st.plotly_chart(fig, use_container_width=True)

        if not (99.9 < soma < 100.1):
            st.warning(f"A soma da alocação é {_fmt_num_br(soma)}%. Os valores foram normalizados para 100%.")

        if st.button(f"Limpar Todos os Ativos de {titulo}", key=f"clear_{portfolio_key}"):
            st.session_state[portfolio_key] = pd.DataFrame(columns=dfp.columns)
            st.rerun()
    
    return dfp


# =========================
# ABAS DE NAVEGAÇÃO
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Projeção Sugerida", "💼 Portfólio Atual", "🎨 Personalizar Carteira", "📊 Comparativos", "📋 Relatório"])

with tab1:
    st.subheader("Projeção da Carteira Sugerida")
    proj_sugerida = calcular_projecao(valor_inicial, aportes_mensais, rent_am_sugerida, prazo_meses)
    df_proj = pd.DataFrame({"Mês": range(prazo_meses + 1), "Carteira Sugerida": proj_sugerida})
    fig_proj = criar_grafico_projecao(df_proj, "Projeção de Crescimento do Patrimônio")
    fig_proj.add_hline(y=meta_financeira, line_dash="dash", line_color="red", annotation_text="Meta Financeira", annotation_position="top left")
    st.plotly_chart(fig_proj, use_container_width=True)

    st.subheader(f"Alocação Sugerida — Perfil {perfil_investimento}")
    st.dataframe(df_sugerido.style.format({"Valor (R$)": fmt_brl, "Alocação (%)": fmt_pct100_br}), use_container_width=True, hide_index=True)
    fig_aloc_sugerida = criar_grafico_alocacao(df_sugerido.rename(columns={"Classe de Ativo":"Descrição"}), "Alocação da Carteira Sugerida")
    st.plotly_chart(fig_aloc_sugerida, use_container_width=True)

with tab2:
    df_atual = form_portfolio('portfolio_atual', "Portfólio Atual")

with tab3:
    df_personalizado = form_portfolio('portfolio_personalizado', "Portfólo Personalizado")

with tab4:
    st.subheader("Em desenvolvimento")
    st.info("A aba de comparativos será implementada em uma próxima versão.")

with tab5:
    st.subheader("Em desenvolvimento")
    st.info("A aba de relatórios será implementada em uma próxima versão.")

# =========================
# RODAPÉ
# =========================
st.markdown("---")
with st.expander("Avisos Importantes", expanded=False):
    st.warning("""
    Os resultados simulados são meramente ilustrativos e não configuram garantia de rentabilidade futura.
    Fundos de investimento não contam com garantia do FGC. Leia os documentos dos produtos antes de investir.
    """)