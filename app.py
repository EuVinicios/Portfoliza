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
# HELPERS NUMÉRICOS
# =========================
def _parse_float(txt: str, default: float=0.0) -> float:
    if txt is None:
        return default
    s = str(txt).strip().replace(".", "").replace(",", ".")  # aceita 10.000,50
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default

def number_input_allow_blank(label: str, default: float, key: str, help: Optional[str]=None):
    """Input que permite apagar (vazio => 0)."""
    placeholder = f"{default:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    val_str = st.text_input(label, value=placeholder, key=key, help=help)
    return _parse_float(val_str, default=0.0)

# =========================
# HELPERS DE FORMATAÇÃO (pt-BR)
# =========================
def _fmt_num_br(v: float, nd: int = 2) -> str:
    try:
        return f"{float(v):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(v)

def fmt_brl(v: float) -> str:
    try:
        return "R$ " + _fmt_num_br(float(v), 2)
    except Exception:
        return f"R$ {v}"

def fmt_pct_br(frac: float) -> str:
    """Recebe fração (0.1234) e exibe '12,34 %'."""
    try:
        return _fmt_num_br(float(frac) * 100.0, 2) + " %"
    except Exception:
        return str(frac)

def fmt_pct100_br(pct: float) -> str:
    """Recebe valor já em % (12.34) e exibe '12,34 %'."""
    try:
        return _fmt_num_br(float(pct), 2) + " %"
    except Exception:
        return str(pct)

def style_df_br(
    df: pd.DataFrame,
    money_cols: Optional[List[str]] = None,
    pct_cols: Optional[List[str]] = None,      # espera fração (0-1)
    pct100_cols: Optional[List[str]] = None,   # espera 0-100
    num_cols: Optional[List[str]] = None,
):
    money_cols = money_cols or []
    pct_cols = pct_cols or []
    pct100_cols = pct100_cols or []
    num_cols = num_cols or []

    fmt_map = {}
    for c in money_cols:
        if c in df.columns: fmt_map[c] = fmt_brl
    for c in pct_cols:
        if c in df.columns: fmt_map[c] = fmt_pct_br
    for c in pct100_cols:
        if c in df.columns: fmt_map[c] = fmt_pct100_br
    for c in num_cols:
        if c in df.columns: fmt_map[c] = lambda x: _fmt_num_br(x, 2)

    try:
        return df.style.format(fmt_map)
    except Exception:
        dff = df.copy()
        for c, f in fmt_map.items():
            dff[c] = dff[c].map(f)
        return dff

def maybe_hide_index(styled_or_df):
    try:
        return styled_or_df.hide(axis="index")
    except Exception:
        return styled_or_df

# =========================
# YAHOO FINANÇAS HELPERS
# =========================
YF_TICKERS = {
    "Dólar (USD/BRL)": ["USDBRL=X", "BRL=X"],
    "Euro (EUR/BRL)": ["EURBRL=X"],
    "Ibovespa": ["^BVSP"],
    "IFIX (aprox.)": ["IFIX.SA", "^IFIX"],
    "S&P 500": ["^GSPC"],
    "Nasdaq": ["^IXIC"],
    "Bitcoin": ["BTC-USD"],
    "Ouro (Comex)": ["GC=F"],
    "Petróleo WTI": ["CL=F"],
    "US 10Y": ["^TNX"],  # valor em 'yield' * 10
}

if HAS_YF:
    @st.cache_data(ttl=900, show_spinner=False)
    def _yf_download_cached(symbol: str, period: str="5d", interval: str="1d") -> Optional[pd.DataFrame]:
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                return df
        except Exception:
            return None
        return None

def _yf_last_close_change(symbols: List[str]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if not HAS_YF:
        return None, None, None
    for s in symbols:
        df = _yf_download_cached(s)
        if df is not None and not df.empty:
            close_series = df["Close"].dropna()
            if len(close_series) >= 1:
                last = float(close_series.iloc[-1])
                prev = float(close_series.iloc[-2]) if len(close_series) >= 2 else np.nan
                chg = None if np.isnan(prev) else (last/prev - 1.0) * 100.0
                return last, chg, s
    return None, None, None

def render_market_strip(cdi_aa: float, ipca_aa: float, selic_aa: Optional[float]=None):
    quotes = []
    for nome, syms in YF_TICKERS.items():
        px, chg, used = _yf_last_close_change(syms)
        if px is None:
            continue
        if "USD/BRL" in nome or "BRL" in nome or "Euro" in nome:
            val = "R$ " + _fmt_num_br(px, 4)
        elif "Bitcoin" in nome:
            val = "US$ " + _fmt_num_br(px, 0)
        elif "US 10Y" in nome:
            val = _fmt_num_br(px/10, 2) + "%"
        else:
            val = _fmt_num_br(px, 2)
        pct = "" if chg is None else (("+" if chg >= 0 else "") + _fmt_num_br(chg, 2) + "%")
        direction = "flat"
        if chg is not None:
            direction = "up" if chg >= 0 else "down"
        quotes.append({"label": nome, "val": val, "pct": pct, "dir": direction})

    base_cards = [
        {"label": "CDI (App)", "val": f"{_fmt_num_br(cdi_aa,2)}%", "pct": "", "dir":"flat"},
        {"label": "IPCA (App)", "val": f"{_fmt_num_br(ipca_aa,2)}%", "pct": "", "dir":"flat"},
    ]
    if selic_aa is not None:
        base_cards.append({"label": "Selic (App)", "val": f"{_fmt_num_br(selic_aa,2)}%", "pct": "", "dir":"flat"})

    items = base_cards + quotes

    style = """
    <style>
      .tstrip-wrap{background:#0b1221;border-radius:12px;padding:8px 10px;margin:4px 0 8px;border:1px solid #182235;}
      .tstrip-row{display:flex;gap:10px;overflow-x:auto;white-space:nowrap;scrollbar-width:thin}
      .tstrip-item{display:inline-flex;align-items:baseline;gap:6px;padding:6px 10px;border-radius:999px;
                   background:#0f172a;border:1px solid #1f2937}
      .tstrip-label{font-size:12px;color:#94a3b8}
      .tstrip-val{font-size:13px;font-weight:600;color:#e5e7eb}
      .tstrip-pct{font-size:12px;font-weight:600}
      .tstrip-pct.up{color:#16a34a}
      .tstrip-pct.down{color:#dc2626}
      .tstrip-pct.flat{color:#94a3b8}
    </style>
    """
    chips = []
    for it in items:
        chips.append(
            f"""<div class="tstrip-item">
                   <span class="tstrip-label">{it['label']}</span>
                   <span class="tstrip-val">{it['val']}</span>
                   <span class="tstrip-pct {it['dir']}">{it['pct']}</span>
                </div>"""
        )
    html_block = style + f"""<div class="tstrip-wrap"><div class="tstrip-row">{''.join(chips)}</div></div>"""
    st.markdown("### Panorama de Mercado")
    st.markdown(html_block, unsafe_allow_html=True)

# =========================
# PDF: CARREGAR & PARSE (somente leitura, sem exibir)
# =========================
def load_pdf_bytes(uploaded_file, default_path: Optional[str]) -> Tuple[Optional[bytes], str]:
    if uploaded_file is not None:
        return uploaded_file.read(), "PDF carregado por upload."
    if default_path and os.path.exists(default_path):
        with open(default_path, "rb") as f:
            return f.read(), f"PDF carregado do caminho local: {default_path}"
    return None, "Nenhum PDF carregado (usando configurações padrão)."

_CLASSE_NORMALIZAR = {
    r"renda fixa pós.*fixada": "Renda Fixa Pós-Fixada",
    r"p[oó]s[\s\-]*cdi|cdi": "Renda Fixa Pós-Fixada",
    r"renda fixa infla[cç][aã]o|ipca\+?": "Renda Fixa Inflação",
    r"cr[eé]dito privado|deb[eê]ntures|cra|cri": "Crédito Privado",
    r"fundos imobili[aá]rios|fii": "Fundos Imobiliários",
    r"a[cç][oõ]es.*[íi]ndice|etf|fundos de [íi]ndice|fundos de indice": "Ações e Fundos de Índice",
    r"previd[eê]ncia": "Previdência Privada",
}
_PERFIS = ["Conservador", "Moderado", "Arrojado"]

def _normalizar_classe(label: str) -> Optional[str]:
    l = label.lower()
    for pat, out in _CLASSE_NORMALIZAR.items():
        if re.search(pat, l, flags=re.I):
            return out
    return None

@st.cache_data(show_spinner=False)
def extrair_carteiras_do_pdf_cached(pdf_bytes: bytes) -> Dict[str, Dict]:
    if not (pdf_bytes and HAS_PDFPLUMBER):
        return DEFAULT_CARTEIRAS
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tx = page.extract_text() or ""
                text += "\n" + tx

        blocos = {}
        for i, perfil in enumerate(_PERFIS):
            start = re.search(perfil, text, flags=re.I)
            if not start:
                continue
            start_idx = start.start()
            end_idx = len(text)
            for j in range(i+1, len(_PERFIS)):
                nxt = re.search(_PERFIS[j], text, flags=re.I)
                if nxt:
                    end_idx = min(end_idx, nxt.start())
            blocos[perfil] = text[start_idx:end_idx]

        carteiras: Dict[str, Dict] = {}
        for perfil, bloco in blocos.items():
            pairs: Dict[str, float] = {}
            for line in bloco.splitlines():
                m = re.search(r"([A-Za-zÀ-ÿ \-\+\/]+?)\s+(\d{1,3})\s*%", line.strip())
                if m:
                    rotulo = m.group(1).strip()
                    pct = float(m.group(2)) / 100.0
                    classe = _normalizar_classe(rotulo)
                    if classe:
                        pairs[classe] = pairs.get(classe, 0.0) + pct
            if pairs:
                soma = sum(pairs.values()) or 1.0
                pairs = {k: v/soma for k, v in pairs.items()}
                rent = {"Conservador": 0.08, "Moderado": 0.10, "Arrojado": 0.12}.get(perfil, 0.10)
                carteiras[perfil] = {"rentabilidade_esperada_aa": rent, "alocacao": pairs}

        return carteiras if carteiras else DEFAULT_CARTEIRAS
    except Exception:
        return DEFAULT_CARTEIRAS

# =========================
# FINANCE HELPERS
# =========================
def aa_to_am(taxa_aa: float) -> float:
    return (1 + taxa_aa) ** (1/12) - 1

def safe_aa_to_am(taxa_aa: float) -> float:
    """Converte anual→mensal; se vier inválida/NaN, retorna 0.0 a.m. (evita sumiço do traço)."""
    try:
        x = float(taxa_aa)
        if not np.isfinite(x):
            return 0.0
        return aa_to_am(x)
    except Exception:
        return 0.0

def calcular_projecao(valor_inicial, aportes_mensais, taxa_mensal, prazo_meses: int):
    vals = [valor_inicial]
    for _ in range(prazo_meses):
        vals.append((vals[-1] + aportes_mensais) * (1 + float(taxa_mensal if np.isfinite(taxa_mensal) else 0.0)))
    return vals

def criar_grafico_projecao(df, title="Projeção de Crescimento"):
    fig = px.line(
        df, x='Mês', y=[c for c in df.columns if c != 'Mês'],
        title=title, labels={'value': 'Patrimônio (R$)', 'variable': 'Cenário'},
        markers=True, color_discrete_sequence=PALETA, template=TEMPLATE
    )
    fig.update_layout(legend_title_text='Cenários', yaxis_title='Patrimônio (R$)', xaxis_title='Meses')
    return fig

def criar_grafico_alocacao(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        return go.Figure()

    df = df.copy()
    if "Valor (R$)" not in df.columns:
        base = float(globals().get("valor_inicial", 0.0) or 0.0)
        if "Alocação Normalizada (%)" in df.columns:
            df["Valor (R$)"] = (base * df["Alocação Normalizada (%)"] / 100.0).round(2) if base > 0 else df["Alocação Normalizada (%)"]
        elif "Alocação (%)" in df.columns:
            df["Valor (R$)"] = (base * df["Alocação (%)"] / 100.0).round(2) if base > 0 else df["Alocação (%)"]
        elif "Valor" in df.columns:
            df["Valor (R$)"] = df["Valor"]
        else:
            df["Valor (R$)"] = 1.0

    df = df[df["Valor (R$)"].fillna(0) >= 0]
    if df["Valor (R$)"].sum() <= 0:
        return go.Figure()

    if "Descrição" in df.columns:
        nomes = "Descrição"
    elif "Classe de Ativo" in df.columns:
        nomes = "Classe de Ativo"
    elif "Classe" in df.columns:
        nomes = "Classe"
    else:
        df = df.reset_index(drop=True)
        df["Item"] = [f"Item {i+1}" for i in range(len(df))]
        nomes = "Item"

    fig = px.pie(
        df, values="Valor (R$)", names=nomes, title=title,
        hole=.35, color_discrete_sequence=PALETA, template=TEMPLATE
    )
    fig.update_traces(textinfo='percent+label', pull=[0.02]*len(df))
    fig.update_layout(legend_title_text='Classe de Ativo', margin=dict(t=40, b=20, l=0, r=0), showlegend=True)
    return fig

# ========= SUBSTITUIÇÃO DO KALEIDO =========
def _fig_placeholder_div(fig, alt: str) -> str:
    if fig is None:
        return ('<div style="padding:8px;border:1px dashed #ccc;border-radius:8px;color:#666">'
                'Sem dados para o gráfico.</div>')
    dom_id = f"figwrap_{uuid.uuid4().hex}"
    fig_json = fig.to_json()
    return f"""
    <div class="figwrap" id="{dom_id}">
        <script type="application/json" class="figspec">{fig_json}</script>
        <div class="ph" style="color:#666">Gerando gráfico…</div>
        <noscript>Ative o JavaScript para visualizar este gráfico.</noscript>
    </div>
    """

def fig_to_img_html(fig, alt: str) -> str:
    return _fig_placeholder_div(fig, alt)

# =========================
# TOGGLES → TIPOS (PERSONALIZAR)
# =========================
TIPOS_ATIVO_BASE = [
    "Debêntures","CRA","CRI","Tesouro Direto","Ações",
    "Fundos de Índice (ETF)","Fundos Imobiliários (FII)",
    "CDB","LCA","LCI","Renda Fixa Pós-Fixada","Renda Fixa Inflação",
    "Crédito Privado","Previdência Privada","Sintético","Outro"
]

TOGGLE_MAP = {
    "Crédito Privado": {"Debêntures","CRA","CRI","Crédito Privado"},
    "Previdência Privada": {"Previdência Privada"},
    "Fundos Imobiliários": {"Fundos Imobiliários (FII)"},
    "Ações e Fundos de Índice": {"Ações","Fundos de Índice (ETF)"},
}
TOGGLE_ALL = set().union(*TOGGLE_MAP.values())

def tipos_permitidos_por_toggles(incluir_credito_privado: bool,
                                 incluir_previdencia: bool,
                                 incluir_fii: bool,
                                 incluir_acoes_indice: bool) -> set:
    allowed = set(TIPOS_ATIVO_BASE)
    if not incluir_credito_privado:
        allowed -= TOGGLE_MAP["Crédito Privado"]
    if not incluir_previdencia:
        allowed -= TOGGLE_MAP["Previdência Privada"]
    if not incluir_fii:
        allowed -= TOGGLE_MAP["Fundos Imobiliários"]
    if not incluir_acoes_indice:
        allowed -= TOGGLE_MAP["Ações e Fundos de Índice"]
    return allowed

def filtrar_df_por_toggles(df: pd.DataFrame, allowed_types: set) -> Tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    mask = df["Tipo"].isin(list(allowed_types))
    removed = int((~mask).sum())
    return df.loc[mask].copy(), removed

# =========================
# SESSION STATE
# =========================
if 'portfolio_atual' not in st.session_state:
    st.session_state.portfolio_atual = pd.DataFrame(columns=[
        "Tipo", "Descrição", "Indexador", "Parâmetro Indexação (% a.a.)",
        "IR (%)", "Isento", "Rent. 12M (%)", "Rent. 6M (%)", "Alocação (%)"
    ])
if 'portfolio_personalizado' not in st.session_state:
    st.session_state.portfolio_personalizado = st.session_state.portfolio_atual.copy()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-top:-8px;margin-bottom:-6px">
          <div style="font-size:46px;line-height:1">📊</div>
          <div style="font-weight:600;font-size:18px">Parâmetros do Cliente</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    nome_cliente = st.text_input("Nome do Cliente", "Cliente Exemplo")

    st.subheader("Carteiras Sugeridas (PDF)")
    pdf_upload = st.file_uploader("Anexar PDF", type=["pdf"], help="Opcional: anexe o PDF de carteiras sugeridas.")
    default_pdf_path = "/Users/macvini/Library/CloudStorage/OneDrive-Pessoal/Repos/Portfoliza/Materiais/CarteiraSugeridaBB.pdf"
    pdf_bytes, pdf_msg = load_pdf_bytes(pdf_upload, default_pdf_path)
    st.caption(pdf_msg)

    st.subheader("Parâmetros de Mercado (a.a.)")
    cdi_aa = number_input_allow_blank("CDI esperado (% a.a.)", 12.0, key="cdi_aa", help="Usado para 'Pós CDI'")
    ipca_aa = number_input_allow_blank("IPCA esperado (% a.a.)", 4.0, key="ipca_aa", help="Usado para 'IPCA+'")
    selic_aa = number_input_allow_blank("Selic esperada (% a.a.)", 12.0, key="selic_aa", help="Exibição (não altera cálculos).")

    carteiras_from_pdf = extrair_carteiras_do_pdf_cached(pdf_bytes) if pdf_bytes else DEFAULT_CARTEIRAS
    perfil_investimento = st.selectbox("Perfil de Investimento", list(carteiras_from_pdf.keys()))

    st.subheader("Opções da Carteira Sugerida")
    incluir_credito_privado = st.checkbox("Incluir Crédito Privado", True)
    incluir_previdencia = st.checkbox("Incluir Previdência", False)
    incluir_fundos_imobiliarios = st.checkbox("Incluir Fundos Imobiliários", True)
    incluir_acoes_indice = st.checkbox("Incluir Ações e Fundos de Índice (ETF)", True)
    st.markdown("---")

    st.subheader("Projeção — Parâmetros")
    valor_inicial = number_input_allow_blank("Valor Inicial do Investimento (R$)", 50000.0, key="valor_inicial")
    aportes_mensais = number_input_allow_blank("Aportes Mensais (R$)", 1000.0, key="aportes_mensais")
    prazo_meses = st.slider("Prazo de Permanência (meses)", 1, 120, 60)
    meta_financeira = number_input_allow_blank("Meta a Atingir (R$)", 500000.0, key="meta_financeira")
    ir_eq_sugerida = st.number_input(
        "IR equivalente p/ Carteira Sugerida (%)",
        min_value=0.0, max_value=100.0, value=15.0, step=0.5,
        help="Usado para estimar retorno LÍQUIDO da Carteira Sugerida (aproximação)."
    )
    ir_cdi = st.number_input(
        "IR p/ CDI (%) (linha de referência)",
        min_value=0.0, max_value=100.0, value=15.0, step=0.5,
        help="Traça a linha de 'CDI líquido de IR' nos comparativos."
    )

# =========================
# HEADER + STRIP DE MERCADO
# =========================
st.title(f"💹 Análise de Portfólio — {nome_cliente}")
st.caption(f"Perfil selecionado: **{perfil_investimento}** • Prazo: **{prazo_meses} meses**")
render_market_strip(cdi_aa=cdi_aa, ipca_aa=ipca_aa, selic_aa=selic_aa)

# =========================
# CARTEIRA SUGERIDA (com toggles)
# =========================
carteira_base = carteiras_from_pdf[perfil_investimento]
aloc_sugerida = carteira_base["alocacao"].copy()

toggle_flags = {
    "Crédito Privado": incluir_credito_privado,
    "Fundos Imobiliários": incluir_fundos_imobiliarios,
    "Ações e Fundos de Índice": incluir_acoes_indice,
    "Previdência Privada": incluir_previdencia,
}
for classe, flag in toggle_flags.items():
    if flag:
        if classe == "Previdência Privada" and classe not in aloc_sugerida:
            aloc_sugerida[classe] = 0.10
    else:
        aloc_sugerida.pop(classe, None)

tot = sum(aloc_sugerida.values()) or 1.0
aloc_sugerida = {k: v/tot for k, v in aloc_sugerida.items()}

df_sugerido = pd.DataFrame(list(aloc_sugerida.items()), columns=["Classe de Ativo", "Alocação (%)"])
df_sugerido["Alocação (%)"] = (df_sugerido["Alocação (%)"] * 100).round(2)
df_sugerido["Valor (R$)"] = (valor_inicial * df_sugerido["Alocação (%)"] / 100.0).round(2)

rent_aa_sugerida = carteira_base.get("rentabilidade_esperada_aa", 0.10)
rent_am_sugerida = aa_to_am(rent_aa_sugerida)

ALLOWED_TYPES = tipos_permitidos_por_toggles(
    incluir_credito_privado, incluir_previdencia, incluir_fundos_imobiliarios, incluir_acoes_indice
)

# =========================
# ABAS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Projeção & Carteira Sugerida",
    "💼 Portfólio Atual",
    "🎨 Personalizar Carteira",
    "📊 Comparativos",
    "📋 Relatório"
])

# =========================
# ABA 1
# =========================
with tab1:
    st.subheader("Projeção da Carteira Sugerida")
    proj_sugerida = calcular_projecao(valor_inicial, aportes_mensais, rent_am_sugerida, prazo_meses)
    df_proj = pd.DataFrame({"Mês": list(range(prazo_meses + 1)), "Carteira Sugerida": proj_sugerida})
    fig_proj = criar_grafico_projecao(df_proj, "Projeção de Crescimento do Patrimônio")
    fig_proj.add_hline(y=meta_financeira, line_dash="dash", line_color="red",
                       annotation_text="Meta Financeira", annotation_position="top left")
    st.plotly_chart(fig_proj, use_container_width=True)

    st.subheader(f"Alocação Sugerida — Perfil {perfil_investimento}")
    styled_sug = style_df_br(
        df_sugerido,
        money_cols=["Valor (R$)"],
        pct100_cols=["Alocação (%)"]
    )
    st.dataframe(maybe_hide_index(styled_sug), use_container_width=True)
    fig_aloc_sugerida = criar_grafico_alocacao(
        df_sugerido.rename(columns={"Classe de Ativo":"Descrição"}), "Alocação da Carteira Sugerida"
    )
    st.plotly_chart(fig_aloc_sugerida, use_container_width=True)

# =========================
# FORM DINÂMICO DO INDEXADOR
# =========================
INDEXADORES = ["Pós CDI", "Prefixado", "IPCA+"]

def param_indexador_input(indexador: str, portfolio_key: str):
    dyn_key = f"par_{portfolio_key}_{indexador.replace(' ', '_')}"
    if indexador == "Pós CDI":
        return st.number_input("% do CDI (% a.a.)", min_value=0.0, value=110.0, step=1.0, key=dyn_key)
    elif indexador == "Prefixado":
        return st.number_input("Taxa Prefixada (% a.a.)", min_value=0.0, value=14.0, step=0.1, key=dyn_key)
    else:
        return st.number_input("Taxa sobre IPCA (% a.a.)", min_value=0.0, value=5.0, step=0.1, key=dyn_key)

def taxa_inputs_group(indexador: str, portfolio_key: str, prefix: str = "") -> float:
    kb = f"{prefix}{portfolio_key}"
    v_cdi  = st.number_input("% do CDI (% a.a.)",        min_value=0.0, value=110.0, step=1.0,  key=f"par_cdi_{kb}",  disabled=(indexador!="Pós CDI"))
    v_pre  = st.number_input("Taxa Prefixada (% a.a.)",  min_value=0.0, value=14.0, step=0.1,  key=f"par_pre_{kb}",  disabled=(indexador!="Prefixado"))
    v_ipca = st.number_input("Taxa sobre IPCA (% a.a.)", min_value=0.0, value=5.0,  step=0.1,  key=f"par_ipca_{kb}", disabled=(indexador!="IPCA+"))
    return v_cdi if indexador=="Pós CDI" else (v_pre if indexador=="Prefixado" else v_ipca)

def ir_inputs_group(portfolio_key: str, col_sel, col_custom):
    with col_sel:
        ir_opt = st.selectbox(
            "IR",
            ["Isento", "15", "17.5", "20", "22.5", "Outro"],
            key=f"ir_{portfolio_key}"
        )
    with col_custom:
        ir_custom = st.number_input(
            "IR personalizado (%)",
            min_value=0.0, max_value=100.0, value=15.0, step=0.5,
            key=f"irv_{portfolio_key}",
            disabled=(ir_opt != "Outro")
        )
    isento = (ir_opt == "Isento")
    if isento:
        ir_pct = 0.0
    elif ir_opt == "Outro":
        ir_pct = ir_custom
    else:
        ir_pct = float(ir_opt)
    return ir_pct, isento

def form_portfolio(portfolio_key: str, titulo: str, allowed_types: set):
    st.subheader(titulo)
    tipos_visiveis = [t for t in TIPOS_ATIVO_BASE if (t in allowed_types) or (t not in TOGGLE_ALL)]
    dfp = st.session_state[portfolio_key]

    with st.expander("Adicionar/Remover Ativos", expanded=True if dfp.empty else False):
        c = st.columns(9)
        tipo      = c[0].selectbox("Tipo", tipos_visiveis, key=f"tipo_{portfolio_key}")
        desc      = c[1].text_input("Descrição", key=f"desc_{portfolio_key}")
        indexador = c[2].selectbox("Indexador", ["Pós CDI","Prefixado","IPCA+"], key=f"idx_{portfolio_key}")

        with c[3]:
            par_idx = taxa_inputs_group(indexador, portfolio_key)
            st.caption("O campo de taxa habilitado depende do indexador.")

        ir_pct, isento = ir_inputs_group(portfolio_key, c[4], c[5])
        r12  = c[6].number_input("Rent. 12M (%)", min_value=0.0, value=0.0, step=0.1, key=f"r12_{portfolio_key}")
        r6   = c[7].number_input("Rent. 6M (%)", min_value=0.0, value=0.0, step=0.1, key=f"r6_{portfolio_key}")
        aloc = c[8].number_input("Alocação (%)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, key=f"aloc_{portfolio_key}")

        if st.button("Adicionar Ativo", key=f"add_{portfolio_key}"):
            if desc.strip():
                novo = pd.DataFrame([{
                    "Tipo": tipo,
                    "Descrição": desc.strip(),
                    "Indexador": indexador,
                    "Parâmetro Indexação (% a.a.)": par_idx,
                    "IR (%)": ir_pct,
                    "Isento": isento,
                    "Rent. 12M (%)": r12,
                    "Rent. 6M (%)": r6,
                    "Alocação (%)": aloc
                }])
                st.session_state[portfolio_key] = pd.concat([st.session_state[portfolio_key], novo], ignore_index=True)
                st.rerun()
            else:
                st.warning("Informe a **Descrição** antes de adicionar.")

        dfp = st.session_state[portfolio_key]
        dfp_filt, removed = filtrar_df_por_toggles(dfp, allowed_types)
        if removed > 0:
            st.info(f"{removed} ativo(s) ocultado(s) por configuração da barra lateral.")

        if not dfp_filt.empty:
            soma = dfp_filt["Alocação (%)"].sum()
            dfp_filt["Alocação Normalizada (%)"] = (dfp_filt["Alocação (%)"]/soma*100.0).round(2)
            dfp_filt["Valor (R$)"] = (valor_inicial * dfp_filt["Alocação Normalizada (%)"]/100.0).round(2)

            if HAS_AGGRID:
                gob = GridOptionsBuilder.from_dataframe(dfp_filt)
                gob.configure_selection('single', use_checkbox=True)
                gob.configure_grid_options(domLayout='autoHeight')

                money_cols_grid = [c for c in ["Valor (R$)"] if c in dfp_filt.columns]
                pct100_cols_grid = [c for c in ["IR (%)","Rent. 12M (%)","Alocação Normalizada (%)","Alocação (%)"] if c in dfp_filt.columns]
                num_cols_grid = [c for c in ["Parâmetro Indexação (% a.a.)"] if c in dfp_filt.columns]

                for c in money_cols_grid:
                    gob.configure_column(c, type=["numericColumn"],
                        valueFormatter='(value==null? "": new Intl.NumberFormat("pt-BR",{style:"currency",currency:"BRL"}).format(Number(value)))')
                for c in pct100_cols_grid:
                    gob.configure_column(c, type=["numericColumn"],
                        valueFormatter='(value==null? "": (Number(value).toLocaleString("pt-BR",{minimumFractionDigits:2, maximumFractionDigits:2}) + " %"))')
                for c in num_cols_grid:
                    gob.configure_column(c, type=["numericColumn"],
                        valueFormatter='(value==null? "": Number(value).toLocaleString("pt-BR",{minimumFractionDigits:2, maximumFractionDigits:2}))')

                grid = AgGrid(
                    dfp_filt, gridOptions=gob.build(),
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    theme='streamlit', fit_columns_on_grid_load=True
                )
                sel = grid["selected_rows"]
                if sel:
                    st.info(f"Editar: **{sel[0].get('Descrição','')}**")
                    with st.form(f"edit_{portfolio_key}"):
                        novo_tipo = st.selectbox("Tipo", tipos_visiveis, index=tipos_visiveis.index(sel[0]["Tipo"]))
                        novo_indexador = st.selectbox("Indexador", INDEXADORES, index=INDEXADORES.index(sel[0]["Indexador"]))
                        novo_par = param_indexador_input(novo_indexador, f"edit_{portfolio_key}")
                        novo_ir = st.number_input("IR (%) (0 para Isento)", min_value=0.0, max_value=100.0, step=0.5, value=float(sel[0]["IR (%)"]))
                        nova_aloc = st.number_input("Alocação (%)", min_value=0.1, max_value=100.0, step=0.1, value=float(sel[0]["Alocação (%)"]))
                        sub_edit = st.form_submit_button("Aplicar")
                        if sub_edit:
                            desc_sel = sel[0]["Descrição"]
                            real_idx = st.session_state[portfolio_key].index[st.session_state[portfolio_key]["Descrição"] == desc_sel][0]
                            st.session_state[portfolio_key].loc[real_idx, ["Tipo","Indexador","Parâmetro Indexação (% a.a.)","IR (%)","Isento","Alocação (%)"]] = [
                                novo_tipo, novo_indexador, novo_par, novo_ir, (novo_ir==0.0), nova_aloc
                            ]
                            st.rerun()
            else:
                cols_view = ["Tipo","Descrição","Indexador","Parâmetro Indexação (% a.a.)","IR (%)","Isento","Rent. 12M (%)","Alocação Normalizada (%)","Valor (R$)"]
                cols_view = [c for c in cols_view if c in dfp_filt.columns]
                styled = style_df_br(
                    dfp_filt[cols_view],
                    money_cols=["Valor (R$)"],
                    pct100_cols=[c for c in ["IR (%)","Rent. 12M (%)","Alocação Normalizada (%)"] if c in cols_view],
                    num_cols=["Parâmetro Indexação (% a.a.)"]
                )
                st.dataframe(maybe_hide_index(styled), use_container_width=True)

            fig = criar_grafico_alocacao(
                dfp_filt.rename(columns={"Tipo":"Classe","Descrição":"Descrição"}), f"Alocação — {titulo}"
            )
            st.plotly_chart(fig, use_container_width=True)

            if soma > 100.1 or soma < 99.9:
                st.warning(f"A soma da alocação é {_fmt_num_br(soma,2)}%. Os valores foram normalizados para 100%.")

            colb = st.columns(2)
            with colb[0]:
                if st.button(f"Limpar {titulo}", key=f"clear_{portfolio_key}"):
                    st.session_state[portfolio_key] = pd.DataFrame(columns=st.session_state[portfolio_key].columns)
                    st.rerun()
            with colb[1]:
                if not HAS_AGGRID:
                    st.caption("Dica: instale `streamlit-aggrid` para clique direto na linha.")

        # Retorna a visão filtrada (ou DataFrame vazio)
        return dfp_filt if 'dfp_filt' in locals() else pd.DataFrame()

# =========================
# ABA 2 — PORTFÓLIO ATUAL
# =========================
with tab2:
    df_atual = form_portfolio('portfolio_atual', "Portfólio Atual", allowed_types=TIPOS_ATIVO_BASE)

# =========================
# ABA 3 — PERSONALIZAR
# =========================
with tab3:
    df_personalizado = form_portfolio('portfolio_personalizado', "Portfólio Personalizado", allowed_types=ALLOWED_TYPES)

# =========================
# TAXAS A PARTIR DO INDEXADOR
# =========================
def taxa_aa_from_indexer(indexador: str, par_idx: float, cdi_aa: float, ipca_aa: float) -> float:
    if indexador == "Pós CDI":
        return (par_idx/100.0) * (cdi_aa/100.0)
    elif indexador == "Prefixado":
        return par_idx/100.0
    else:
        return (ipca_aa/100.0) + (par_idx/100.0)

def df_normalizar_pesos(df: pd.DataFrame) -> pd.DataFrame:
    """Garante coluna 'Alocação Normalizada (%)' a partir de 'Alocação (%)'."""
    out = df.copy()
    if out.empty:
        return out
    if "Alocação Normalizada (%)" not in out.columns:
        if "Alocação (%)" in out.columns and out["Alocação (%)"].sum() > 0:
            soma = out["Alocação (%)"].sum()
            out["Alocação Normalizada (%)"] = out["Alocação (%)"]/soma*100.0
    return out

def taxa_portfolio_aa(df: pd.DataFrame, cdi_aa: float, ipca_aa: float,
                      apply_tax: bool=False) -> float:
    if df is None or df.empty:
        return 0.0

    rows_taxas_pesos = []
    for _, r in df.iterrows():
        w = None
        if pd.notna(r.get("Alocação Normalizada (%)", np.nan)):
            try:
                w = float(r["Alocação Normalizada (%)"]) / 100.0
            except Exception:
                w = None
        elif pd.notna(r.get("Alocação (%)", np.nan)):
            try:
                w = float(r["Alocação (%)"]) / 100.0
            except Exception:
                w = None
        if w is None or not np.isfinite(w) or w <= 0:
            continue

        idx = str(r.get("Indexador", "Pós CDI") or "Pós CDI")
        par_raw = r.get("Parâmetro Indexação (% a.a.)", 0.0)
        try:
            par = float(par_raw)
        except Exception:
            par = 0.0
        if pd.isna(par) or not np.isfinite(par):
            par = 0.0

        taxa = taxa_aa_from_indexer(idx, par, cdi_aa, ipca_aa)

        if apply_tax and not bool(r.get("Isento", False)):
            ir_raw = r.get("IR (%)", 0.0)
            try:
                ir = float(ir_raw)
            except Exception:
                ir = 0.0
            if np.isfinite(ir) and ir > 0:
                taxa = taxa * (1 - ir/100.0)

        if pd.isna(taxa) or not np.isfinite(taxa):
            continue

        rows_taxas_pesos.append((taxa, w))

    if not rows_taxas_pesos:
        return 0.0

    taxas, pesos = zip(*rows_taxas_pesos)
    return float(np.average(np.array(taxas), weights=np.array(pesos)))

# =========================
# Preparos para COMPARATIVOS (sempre a partir do estado)
# =========================
# 1) Portfólio Atual (líquido)
df_atual_state = st.session_state.get('portfolio_atual', pd.DataFrame())
df_atual_for_rate = df_normalizar_pesos(df_atual_state)
rent_atual_aa_liq = taxa_portfolio_aa(df_atual_for_rate, cdi_aa, ipca_aa, apply_tax=True)

# 2) Portfólio Personalizado (líquido)
df_pers_state = st.session_state.get('portfolio_personalizado', pd.DataFrame())
df_pers_for_rate = df_normalizar_pesos(df_pers_state)
rent_pers_aa_liq = taxa_portfolio_aa(df_pers_for_rate, cdi_aa, ipca_aa, apply_tax=True)

# 3) Carteira Sugerida (líquida, via IR equivalente)
rent_sugerida_aa_liq = rent_aa_sugerida * (1 - ir_eq_sugerida/100.0)

# =========================
# ABA 4 — COMPARATIVOS (líquido)
# =========================
with tab4:
    st.subheader("Comparativo de Projeção (líquido de IR)")

    cdi_liq_aa = (cdi_aa/100.0) * (1 - ir_cdi/100.0)

    monthly_rates = {
        "Carteira Sugerida (líquida)": safe_aa_to_am(rent_sugerida_aa_liq),
        "Portfólio Atual (líquido)":    safe_aa_to_am(rent_atual_aa_liq),
        "Portfólio Personalizado (líquido)": safe_aa_to_am(rent_pers_aa_liq),
        "CDI líquido de IR":            safe_aa_to_am(cdi_liq_aa),
    }

    df_comp = pd.DataFrame({'Mês': range(25)})
    for nome, taxa_m in monthly_rates.items():
        df_comp[nome] = calcular_projecao(valor_inicial, aportes_mensais, taxa_m, 24)

    fig_comp = criar_grafico_projecao(df_comp, "Projeção — Líquido de Impostos (24 meses)")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("Resumo 12 meses (líquido)")
    linhas = []
    for nome, taxa_m in monthly_rates.items():
        taxa_aa = (1 + float(taxa_m)) ** 12 - 1
        valor_12m = valor_inicial * (1 + taxa_aa)
        linhas.append({
            "Cenário": nome,
            "Rent. 12M (a.a.) Líquida": taxa_aa,
            "Resultado estimado em 12M (R$)": valor_12m
        })
    df_resumo = pd.DataFrame(linhas)
    styled_resumo = style_df_br(
        df_resumo,
        money_cols=["Resultado estimado em 12M (R$)"],
        pct_cols=["Rent. 12M (a.a.) Líquida"]
    )
    st.dataframe(maybe_hide_index(styled_resumo), use_container_width=True)

# =========================
# RELATÓRIO (HTML)
# =========================
def build_html_report(nome: str, perfil: str, prazo_meses: int, valor_inicial: float, aportes: float,
                      meta: float, df_sug_classe: pd.DataFrame, df_produtos: pd.DataFrame,
                      email_text_html: str,
                      fig_comp_placeholder: str,
                      fig_aloc_atual_placeholder: str,
                      fig_aloc_pers_placeholder: str,
                      fig_aloc_sug_placeholder: str) -> str:
    # Classes sugeridas formatadas
    df_sug = df_sug_classe.copy()
    if "Valor (R$)" in df_sug.columns:
        df_sug["Valor (R$)"] = df_sug["Valor (R$)"].map(fmt_brl)
    if "Alocação (%)" in df_sug.columns:
        df_sug["Alocação (%)"] = df_sug["Alocação (%)"].map(fmt_pct100_br)

    cols_prod_all = [
        "Tipo","Descrição","Indexador","Parâmetro Indexação (% a.a.)","IR (%)","Isento",
        "Rent. 12M (%)","Rent. 6M (%)","Alocação (%)","Alocação Normalizada (%)","Valor (R$)"
    ]
    cols_prod = [c for c in cols_prod_all if c in df_produtos.columns]
    df_prod = (df_produtos[cols_prod].copy() if cols_prod else pd.DataFrame())
    for c in ["IR (%)","Rent. 12M (%)","Rent. 6M (%)","Alocação (%)","Alocação Normalizada (%)"]:
        if c in df_prod.columns:
            df_prod[c] = df_prod[c].map(fmt_pct100_br)
    if "Parâmetro Indexação (% a.a.)" in df_prod.columns:
        df_prod["Parâmetro Indexação (% a.a.)"] = df_prod["Parâmetro Indexação (% a.a.)"].map(lambda x: _fmt_num_br(x, 2))
    if "Valor (R$)" in df_prod.columns:
        df_prod["Valor (R$)"] = df_prod["Valor (R$)"].map(fmt_brl)

    tabela_sug_classe = df_sug[["Classe de Ativo","Alocação (%)","Valor (R$)"]].to_html(index=False, border=0)
    tabela_produtos = df_prod.to_html(index=False, border=0)

    style = """
    <style>
      body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; color:#222; padding: 12px; }
      .card { border:1px solid #e5e5e5; border-radius:12px; padding:16px; margin:12px 0; }
      h1,h2,h3 { margin: 0.2rem 0 0.6rem; }
      table { width:100%; border-collapse: collapse; table-layout:auto; }
      th, td {
        padding: 8px 10px; border-bottom: 1px solid #eee; text-align:left;
        white-space: normal; word-break: break-word; overflow-wrap: anywhere; hyphens:auto; font-size:14px;
      }
      .muted { color:#666; font-size: 0.9rem; }
      .tag { background:#f3f4f6; border:1px solid #e5e7eb; padding:3px 8px; border-radius:999px; font-size:0.85rem; }
      .grid { display:grid; grid-template-columns: 1fr; gap: 16px; }
      @media (min-width: 1024px) { .grid-2 { grid-template-columns: 1fr 1fr; } }
      .highlight { background:#fff7ed; border-color:#fdba74; }
      .imgwrap { text-align:center; }
      .imgwrap img { max-width:640px; width:640px; height:auto; }
      .notice { color:#0b1221; background:#e6f4ff; border:1px solid #b6e0ff; border-radius:8px; padding:8px 10px; font-size:0.9rem; }
    </style>
    """
    html_report = f"""
    {style}
    <div id="report-root">
      <h1>Relatório — Análise de Portfólio</h1>
      <div class="muted">Cliente: <span class="tag">{nome}</span> • Perfil: <span class="tag">{perfil}</span> • Prazo: <span class="tag">{prazo_meses} meses</span></div>

      {"<div class='card'><h2>Mensagem do Assessor</h2><div class='muted'>Conteúdo preparado para e-mail</div><div style='margin-top:6px'>" + email_text_html + "</div></div>" if email_text_html else ""}

      <div class="card">
          <h2>Dados Iniciais</h2>
          <ul>
              <li><b>Valor Inicial:</b> {fmt_brl(valor_inicial)}</li>
              <li><b>Aportes Mensais:</b> {fmt_brl(aportes)}</li>
              <li><b>Meta Financeira:</b> {fmt_brl(meta)}</li>
          </ul>
      </div>

      <div class="card highlight">
          <h2>Carteira Sugerida — Alocação por Classe (Destaque)</h2>
          {tabela_sug_classe}
      </div>

      <div class="card">
          <h2>Produtos Selecionados (Portfólio Sugerido ao Cliente)</h2>
          {tabela_produtos}
      </div>

      <div class="card">
          <h2>Comparativo de Projeção (líquido de IR)</h2>
          <div class="imgwrap">{fig_comp_placeholder}</div>
      </div>

      <div class="card">
          <h2>Alocações — Antes e Depois</h2>
          <div class="grid grid-2">
              <div>
                  <h3>Portfólio Atual</h3>
                  <div class="imgwrap">{fig_aloc_atual_placeholder}</div>
              </div>
              <div>
                  <h3>Portfólio Personalizado</h3>
                  <div class="imgwrap">{fig_aloc_pers_placeholder}</div>
              </div>
          </div>
          <div style="margin-top:12px">
              <h3>Carteira Sugerida</h3>
              <div class="imgwrap">{fig_aloc_sug_placeholder}</div>
          </div>
      </div>

      <div class="card">
          <h3>Avisos Importantes</h3>
          <p class="muted">Os resultados simulados são ilustrativos, não configuram garantia de rentabilidade futura.
          As projeções foram consideradas líquidas de IR conforme parâmetros informados/estimados no aplicativo.
          Leia os documentos dos produtos antes de investir.</p>
      </div>
    </div>
    """
    return html_report

# Figuras para o relatório
fig_aloc_atual_rep = criar_grafico_alocacao(
    st.session_state.get('portfolio_atual', pd.DataFrame()).rename(columns={"Tipo":"Classe","Descrição":"Descrição"}),
    "Alocação — Portfólio Atual"
)
fig_aloc_pers_rep = criar_grafico_alocacao(
    st.session_state.get('portfolio_personalizado', pd.DataFrame()).rename(columns={"Tipo":"Classe","Descrição":"Descrição"}),
    "Alocação — Portfólio Personalizado"
)
fig_aloc_sug_rep = criar_grafico_alocacao(
    df_sugerido.rename(columns={"Classe de Ativo":"Descrição"}),
    "Alocação — Carteira Sugerida"
)

# Placeholders (cada um contém o JSON do figure)
comp_img = fig_to_img_html(globals().get("fig_comp", None), "Projeção Líquida")
atual_img = fig_to_img_html(fig_aloc_atual_rep, "Alocação — Portfólio Atual")
pers_img  = fig_to_img_html(fig_aloc_pers_rep, "Alocação — Portfólio Personalizado")
sug_img   = fig_to_img_html(fig_aloc_sug_rep, "Alocação — Carteira Sugerida")

with tab5:
    st.subheader("Relatório (copiar conteúdo formatado)")
    st.caption("Os gráficos são convertidos automaticamente em imagens PNG no momento da cópia.")

    email_msg = st.text_area(
        "Mensagem do e-mail (edite aqui)",
        value="Olá, tudo bem? Segue abaixo a análise e a sugestão de carteira preparada conforme seu perfil e objetivos.",
        height=140,
        help="Este conteúdo vai junto no relatório copiado para colar no e-mail."
    )
    email_msg_html = "<br>".join(html.escape(l) for l in email_msg.splitlines())

    df_prod_report = st.session_state.get('portfolio_personalizado', pd.DataFrame()).copy()
    if not df_prod_report.empty:
        if "Alocação Normalizada (%)" not in df_prod_report.columns and "Alocação (%)" in df_prod_report.columns:
            soma_p = df_prod_report["Alocação (%)"].sum() or 1.0
            df_prod_report["Alocação Normalizada (%)"] = (df_prod_report["Alocação (%)"]/soma_p*100.0).round(2)
        if "Valor (R$)" not in df_prod_report.columns and "Alocação Normalizada (%)" in df_prod_report.columns:
            df_prod_report["Valor (R$)"] = (valor_inicial * df_prod_report["Alocação Normalizada (%)"]/100.0).round(2)

    html_report = build_html_report(
        nome_cliente, perfil_investimento, prazo_meses, valor_inicial, aportes_mensais, meta_financeira,
        df_sugerido, df_prod_report, email_msg_html,
        fig_comp_placeholder=comp_img,
        fig_aloc_atual_placeholder=atual_img,
        fig_aloc_pers_placeholder=pers_img,
        fig_aloc_sug_placeholder=sug_img
    )

    copy_block = f"""
    <div>
      {html_report}
      <div style="margin-top:10px">
        <button id="cpy" style="background:#0b1221;color:#e5e7eb;border:1px solid #1f2937;padding:10px 14px;border-radius:8px;cursor:pointer">
          Copiar conteúdo formatado
        </button>
        <span id="cpyst" style="margin-left:10px;color:#10b981"></span>
      </div>
    </div>
    <script>
      (function(){{
        function ensurePlotly() {{
          return new Promise(function(resolve, reject){{
            if (window.Plotly) return resolve();
            var s = document.createElement('script');
            s.src = 'https://cdn.plot.ly/plotly-2.35.3.min.js';
            s.onload = function(){{ resolve(); }};
            s.onerror = function(){{ reject(new Error('Falha ao carregar plotly.js')); }};
            document.head.appendChild(s);
          }});
        }}

        let rendered = false;

        async function renderAll(){{
          await ensurePlotly();
          const wraps = Array.from(document.querySelectorAll('.figwrap'));
          for (const w of wraps){{
            const specEl = w.querySelector('.figspec');
            if (!specEl) continue;
            let fig;
            try {{
              fig = JSON.parse(specEl.textContent);
            }} catch (e) {{
              w.innerHTML = '<div style="padding:8px;border:1px dashed #ccc;border-radius:8px;color:#666">Erro ao ler gráfico.</div>';
              continue;
            }}
            const div = document.createElement('div');
            div.style.width = '100%';
            div.style.maxWidth = '640px';
            div.style.margin = '0 auto';
            w.innerHTML = '';
            w.appendChild(div);
            try {{
              await Plotly.newPlot(div, fig.data || [], fig.layout || {{}}, {{staticPlot:true, displayModeBar:false}});
              const url = await Plotly.toImage(div, {{format:'png', scale:2}});
              w.innerHTML = '<img src=\"'+url+'\" style=\"max-width:100%;height:auto;border:1px solid #eee;border-radius:12px\" />';
            }} catch (e) {{
              w.innerHTML = '<div style="padding:8px;border:1px dashed #ccc;border-radius:8px;color:#666">Falha ao gerar imagem.</div>';
            }}
          }}
          rendered = true;
        }}

        async function copyHtml(){{
          if (!rendered) await renderAll();
          const root = document.getElementById('report-root');
          if (!root) return;
          const html = root.outerHTML;
          try {{
            if (navigator.clipboard && window.ClipboardItem) {{
              const item = new ClipboardItem({{
                'text/html': new Blob([html], {{type:'text/html'}}),
                'text/plain': new Blob([root.innerText], {{type:'text/plain'}})
              }});
              await navigator.clipboard.write([item]);
            }} else {{
              const sel = window.getSelection();
              const range = document.createRange();
              range.selectNode(root);
              sel.removeAllRanges();
              sel.addRange(range);
              document.execCommand('copy');
              sel.removeAllRanges();
            }}
            document.getElementById('cpyst').textContent = 'Conteúdo copiado para a área de transferência!';
            setTimeout(()=> document.getElementById('cpyst').textContent = '', 3000);
          }} catch(e) {{
            document.getElementById('cpyst').textContent = 'Falha ao copiar';
          }}
        }}

        renderAll();
        document.getElementById('cpy').addEventListener('click', copyHtml);
      }})();
    </script>
    """
    st.components.v1.html(copy_block, height=1200, scrolling=True)

# =========================
# RODAPÉ
# =========================
st.markdown("---")
with st.expander("Avisos Importantes", expanded=False):
    st.warning("""
Os resultados simulados são meramente ilustrativos, não configurando garantia de rentabilidade futura ou promessa de retorno.
As projeções do comparativo estão líquidas de IR conforme parâmetros informados/estimados.
Fundos de investimento não contam com garantia do FGC. Leia os documentos dos produtos antes de investir.
    """)
