# app.py
from __future__ import annotations
import io, os, re, base64, json, uuid, html
from typing import Dict, Tuple, Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(page_title="Gerador de Portfólios de Investimento", page_icon="💹", layout="wide")

# =========================
# IMPORTS OPCIONAIS
# =========================
HAS_PDFPLUMBER = HAS_AGGRID = HAS_YF = HAS_BCB = False
try:
    import pdfplumber; HAS_PDFPLUMBER = True
except Exception:
    pass
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode; HAS_AGGRID = True
except Exception:
    pass
try:
    import yfinance as yf; HAS_YF = True
except Exception:
    pass
# --- BCB / Focus / SGS
try:
    from bcb import Expectativas as BCBExpectativas
    from bcb import sgs
    HAS_BCB = True
except Exception:
    HAS_BCB = False

# =========================
# TEMA DE GRÁFICOS
# =========================
PALETA = px.colors.qualitative.Vivid
TEMPLATE = "plotly_white"

# === DEBUG SWITCH (oculta diagnósticos do usuário) ===
try:
    _qp = st.experimental_get_query_params()
except Exception:
    _qp = {}
DEBUG_MODE = (str(st.secrets.get("DEBUG", "0")) == "1") or ("debug" in _qp)


# =========================
# MOCK DEFAULT (FALLBACK)
# =========================
DEFAULT_CARTEIRAS = {
    "Conservador": {"rentabilidade_esperada_aa": 0.08, "alocacao": {
        "Renda Fixa Pós-Fixada": 0.70, "Renda Fixa Inflação": 0.20, "Crédito Privado": 0.10}},
    "Moderado": {"rentabilidade_esperada_aa": 0.10, "alocacao": {
        "Renda Fixa Pós-Fixada": 0.40, "Renda Fixa Inflação": 0.25, "Crédito Privado": 0.15,
        "Fundos Imobiliários": 0.10, "Ações e Fundos de Índice": 0.10}},
    "Arrojado": {"rentabilidade_esperada_aa": 0.12, "alocacao": {
        "Renda Fixa Pós-Fixada": 0.20, "Renda Fixa Inflação": 0.10, "Crédito Privado": 0.20,
        "Fundos Imobiliários": 0.20, "Ações e Fundos de Índice": 0.30}},
}

# =========================
# HELPERS NUMÉRICOS / FORMATAÇÃO
# =========================
def _parse_float(txt: str, default: float=0.0) -> float:
    if txt is None: return default
    s = str(txt).strip()
    # aceita 1.234,56 e 1234.56
    s = s.replace(".", "").replace(",", ".")
    if s == "": return default
    try:
        return float(s)
    except Exception:
        return default

def number_input_allow_blank(label: str, default: float, key: str, help: Optional[str]=None):
    # Mostra sempre com 2 casas em PT-BR
    placeholder = f"{default:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    val_str = st.text_input(label, value=placeholder, key=key, help=help)
    return _parse_float(val_str, default=default)

def _fmt_num_br(v: float, nd: int = 2) -> str:
    try: return f"{float(v):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception: return str(v)

def fmt_brl(v: float) -> str:
    try: return "R$ " + _fmt_num_br(float(v), 2)
    except Exception: return f"R$ {v}"

def fmt_pct_br(frac: float) -> str:
    try: return _fmt_num_br(float(frac) * 100.0, 2) + " %"
    except Exception: return str(frac)

def fmt_pct100_br(pct: float) -> str:
    try: return _fmt_num_br(float(pct), 2) + " %"
    except Exception: return str(pct)

def style_df_br(df: pd.DataFrame, money_cols: Optional[List[str]] = None,
                pct_cols: Optional[List[str]] = None, pct100_cols: Optional[List[str]] = None,
                num_cols: Optional[List[str]] = None):
    money_cols = money_cols or []; pct_cols = pct_cols or []; pct100_cols = pct100_cols or []; num_cols = num_cols or []
    fmt_map = {}
    for c in money_cols:
        if c in df.columns:
            fmt_map[c] = fmt_brl
    for c in pct_cols:
        if c in df.columns:
            fmt_map[c] = fmt_pct_br
    for c in pct100_cols:
        if c in df.columns:
            fmt_map[c] = fmt_pct100_br
    for c in num_cols:
        if c in df.columns:
            fmt_map[c] = lambda x: _fmt_num_br(x, 2)
    try: return df.style.format(fmt_map)
    except Exception:
        dff = df.copy()
        for c, f in fmt_map.items(): dff[c] = dff[c].map(f)
        return dff

def maybe_hide_index(styled_or_df):
    try: return styled_or_df.hide(axis="index")
    except Exception: return styled_or_df

# =========================
# YAHOO FINANÇAS (strip)
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
    "US 10Y": ["^TNX"],
}
if HAS_YF:
    @st.cache_data(ttl=900, show_spinner=False)
    def _yf_download_cached(symbol: str, period: str="5d", interval: str="1d") -> Optional[pd.DataFrame]:
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            if df is not None and not df.empty: return df
        except Exception:
            return None
        return None

def _yf_last_close_change(symbols: List[str]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if not HAS_YF: return None, None, None
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
        px_, chg, _ = _yf_last_close_change(syms)
        if px_ is None: continue
        if "USD/BRL" in nome or "BRL" in nome or "Euro" in nome: val = "R$ " + _fmt_num_br(px_, 4)
        elif "Bitcoin" in nome: val = "US$ " + _fmt_num_br(px_, 0)
        elif "US 10Y" in nome: val = _fmt_num_br(px_/10, 2) + "%"
        else: val = _fmt_num_br(px_, 2)
        pct = "" if chg is None else (("+" if chg >= 0 else "") + _fmt_num_br(chg, 2) + "%")
        direction = "flat" if chg is None else ("up" if chg >= 0 else "down")
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
      .tstrip-item{display:inline-flex;align-items:baseline;gap:6px;padding:6px 10px;border-radius:999px;background:#0f172a;border:1px solid #1f2937}
      .tstrip-label{font-size:12px;color:#94a3b8}.tstrip-val{font-size:13px;font-weight:600;color:#e5e7eb}
      .tstrip-pct{font-size:12px;font-weight:600}.tstrip-pct.up{color:#16a34a}.tstrip-pct.down{color:#dc2626}.tstrip-pct.flat{color:#94a3b8}
    </style>
    """
    chips = [f"""<div class="tstrip-item"><span class="tstrip-label">{it['label']}</span>
                 <span class="tstrip-val">{it['val']}</span>
                 <span class="tstrip-pct {it['dir']}">{it['pct']}</span></div>"""
             for it in items]
    html_block = style + f"""<div class="tstrip-wrap"><div class="tstrip-row">{''.join(chips)}</div></div>"""
    st.markdown("### Panorama de Mercado")
    st.markdown(html_block, unsafe_allow_html=True)

# =========================
# PDF: CARREGAR 1x
# =========================
def load_pdf_bytes_once(uploaded_file, default_path: Optional[str]) -> Tuple[Optional[bytes], str]:
    store = st.session_state.setdefault("__pdf_store__", {"bytes": None, "msg": "Nenhum PDF carregado (usando configurações padrão)."})
    if uploaded_file is not None:
        store["bytes"] = uploaded_file.read(); store["msg"] = "PDF carregado por upload."
    elif store["bytes"] is None and default_path and os.path.exists(default_path):
        with open(default_path, "rb") as f: store["bytes"] = f.read()
        store["msg"] = f"PDF carregado do caminho local: {default_path}"
    return store["bytes"], store["msg"]

# =========================
# FOCUS/BCB (CACHE)
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_focus_aa_cached() -> dict:
    """
    Retorna {'ipca_aa': float, 'selic_aa': float} com as medianas mais recentes (endpoint Anuais).
    Fallback: tenta Mensais (última DataReferencia >= ano atual).
    """
    out = {}
    if not HAS_BCB:
        return out
    try:
        em = BCBExpectativas()
        ep = em.get_endpoint("ExpectativasMercadoAnuais")
        ano_ref = pd.Timestamp.today().year

        df_any = (ep.query()
                    .select(ep.Indicador, ep.Data, ep.DataReferencia, ep.Mediana)
                    .orderby(ep.Data.desc())
                    .collect())

        def _pick(df, indic):
            df_i = df[df["Indicador"].str.upper().str.contains(indic.upper())]
            if df_i.empty: return None
            # pega mediana do ano atual se existir, senão a última
            try:
                df_i["DataReferencia"] = df_i["DataReferencia"].astype(int)
                df_now = df_i[df_i["DataReferencia"] >= int(ano_ref)]
                base = df_now if not df_now.empty else df_i
            except Exception:
                base = df_i
            return float(base.iloc[0]["Mediana"])

        ipca = _pick(df_any, "IPCA")
        seli = _pick(df_any, "SELIC")

        # Fallback: Mensais
        if (ipca is None or seli is None):
            epm = em.get_endpoint("ExpectativasMercadoMensais")
            df_m = (epm.query()
                      .select(epm.Indicador, epm.Data, epm.DataReferencia, epm.Mediana)
                      .orderby(epm.Data.desc())
                      .collect())
            if ipca is None: ipca = _pick(df_m, "IPCA")
            if seli is None: seli = _pick(df_m, "SELIC")

        if ipca is not None:  out["ipca_aa"]  = ipca
        if seli is not None:  out["selic_aa"] = seli
        return out
    except Exception:
        return out

# ==== CDI a partir do SGS (robusto) ====
from datetime import date, timedelta

# Candidatos de séries SGS para CDI (alguns ambientes têm mapeamentos diferentes)
# - 'ad' = % ao dia (vamos anualizar em 252)
# - 'aa' = % ao ano diretamente
_CDI_SERIES = {
    "ad": [4389, 7809, 12],   # tenta na ordem; 12 costuma ser Selic-meta em alguns ambientes, por isso validamos magnitude
    "aa": [4390, 7802]        # anualizadas (se disponíveis)
}
_SELIC_ANN252_SER = 1178      # Selic anualizada (base 252) – usado no método "basis"

def _try_get_sgs_series(series_id: int, last_points: int = 90) -> Optional[pd.Series]:
    try:
        df = sgs.get({str(series_id): series_id}, last=last_points)
        if df is None or df.empty or str(series_id) not in df.columns:
            # fallback por data
            end = date.today()
            start = end - timedelta(days=last_points*3)
            df = sgs.get({str(series_id): series_id}, start=start.strftime("%d/%m/%Y"), end=end.strftime("%d/%m/%Y"))
        if df is None or df.empty or str(series_id) not in df.columns:
            return None
        s = pd.to_numeric(df[str(series_id)], errors="coerce").dropna()
        return s if not s.empty else None
    except Exception:
        return None

def _pick_first_valid_cdi_ad(last_points: int = 60) -> Optional[pd.Series]:
    """
    Tenta séries candidatas de CDI % a.d. e valida magnitude típica (0,01% a 0,20% ao dia).
    """
    for sid in _CDI_SERIES["ad"]:
        s = _try_get_sgs_series(sid, last_points)
        if s is None:
            continue
        s_frac = s/100.0  # % -> fração
        # magnitude diária típica do CDI (~0.03%–0.07% a.d. em anos “normais”)
        if s_frac.tail(30).between(0.0001, 0.0020).mean() > 0.8:  # >80% dos pontos no range
            return s_frac
    return None

def _pick_first_valid_cdi_aa(last_points: int = 60) -> Optional[pd.Series]:
    """
    Tenta séries candidatas de CDI % a.a. e valida magnitude típica (2%–30% a.a.).
    """
    for sid in _CDI_SERIES["aa"]:
        s = _try_get_sgs_series(sid, last_points)
        if s is None:
            continue
        s_frac = s/100.0
        if s_frac.tail(30).between(0.02, 0.30).mean() > 0.8:
            return s_frac
    return None

def _focus_selic_current_year() -> Optional[float]:
    if not HAS_BCB: return None
    try:
        em = BCBExpectativas()
        ep = em.get_endpoint("ExpectativasMercadoAnuais")
        ano = pd.Timestamp.today().year
        df = (ep.query()
                .filter((ep.Indicador=="Selic") | (ep.Indicador=="SELIC"))
                .select(ep.Data, ep.DataReferencia, ep.Mediana)
                .collect()).sort_values(["Data","DataReferencia"])
        row = df[df["DataReferencia"].astype(str) >= str(ano)].tail(1)
        if row.empty: row = df.tail(1)
        return float(row["Mediana"].iloc[0])
    except Exception:
        return None

def _cdi_from_daily(window_days: int = 30) -> Optional[float]:
    """
    CDI anualizado (% a.a.) pela média dos últimos N dias úteis de CDI % a.d. (validação de magnitude).
    """
    if not HAS_BCB: return None
    s_ad = _pick_first_valid_cdi_ad(last_points=window_days+20)
    if s_ad is None:
        return None
    cdi_ad = s_ad.tail(window_days)
    if cdi_ad.empty:
        return None
    cdi_aa = (1.0 + float(cdi_ad.mean()))**252 - 1.0
    return round(cdi_aa*100.0, 4)

def _cdi_from_basis(window_days: int = 60) -> Optional[float]:
    """
    CDI esperado (% a.a.) = Selic Focus (a.a.) + spread_aa, onde
    spread_aa vem da média[(CDI a.d.) – (Selic a.d.)] anualizada.
    """
    if not HAS_BCB: return None
    s_cdi_ad = _pick_first_valid_cdi_ad(last_points=window_days+30)
    if s_cdi_ad is None:
        return None
    # Selic a.a. base 252 -> converter para a.d.
    s_selic_aa = _try_get_sgs_series(_SELIC_ANN252_SER, last_points=window_days+30)
    if s_selic_aa is None:
        return None
    selic_ad = (s_selic_aa/100.0)/252.0
    df = pd.concat([s_cdi_ad.rename("cdi_ad"), selic_ad.rename("selic_ad")], axis=1).dropna().tail(window_days)
    if df.empty:
        return None
    spread_ad = float((df["cdi_ad"] - df["selic_ad"]).mean())
    spread_aa = (1.0 + spread_ad)**252 - 1.0
    selic_focus_aa = (_focus_selic_current_year() or 12.0)/100.0
    cdi_aa = selic_focus_aa + spread_aa
    return round(max(0.0, cdi_aa)*100.0, 4)

@st.cache_data(ttl=6*3600, show_spinner=False)
def _cdi_expected_cached() -> Tuple[Optional[float], dict]:
    meta = {"method": None, "window": None}
    v = _cdi_from_daily(30)
    if v is not None:
        meta.update({"method": "daily_mean", "window": 30})
        return v, meta
    v = _cdi_from_basis(60)
    if v is not None:
        meta.update({"method": "basis", "window": 60})
        return v, meta
    meta.update({"method": "fallback_selic"})
    return None, meta

# =========================
# FINANCE HELPERS
# =========================
def aa_to_am(taxa_aa: float) -> float: return (1 + taxa_aa) ** (1/12) - 1
def safe_aa_to_am(taxa_aa: float) -> float:
    try:
        x = float(taxa_aa)
        if not np.isfinite(x): return 0.0
        return aa_to_am(x)
    except Exception: return 0.0

def calcular_projecao(valor_inicial, aportes_mensais, taxa_mensal, prazo_meses: int):
    vals = [valor_inicial]
    tm = float(taxa_mensal if np.isfinite(taxa_mensal) else 0.0)
    for _ in range(prazo_meses): vals.append((vals[-1] + aportes_mensais) * (1 + tm))
    return vals

def criar_grafico_projecao(df, title="Projeção de Crescimento"):
    fig = px.line(df, x='Mês', y=[c for c in df.columns if c != 'Mês'], title=title,
                  labels={'value':'Patrimônio (R$)','variable':'Cenário'},
                  markers=True, color_discrete_sequence=PALETA, template=TEMPLATE)
    fig.update_layout(legend_title_text='Cenários', yaxis_title='Patrimônio (R$)', xaxis_title='Meses')
    return fig

def criar_grafico_alocacao(df: pd.DataFrame, title: str):
    if df is None or df.empty: return go.Figure()
    df = df.copy()
    if "Valor (R$)" not in df.columns:
        base = float(globals().get("valor_inicial", 0.0) or 0.0)
        if "Alocação Normalizada (%)" in df.columns:
            df["Valor (R$)"] = (base * df["Alocação Normalizada (%)"]/100.0).round(2) if base > 0 else df["Alocação Normalizada (%)"]
        elif "Alocação (%)" in df.columns:
            df["Valor (R$)"] = (base * df["Alocação (%)"]/100.0).round(2) if base > 0 else df["Alocação (%)"]
        elif "Valor" in df.columns:
            df["Valor (R$)"] = df["Valor"]
        else:
            df["Valor (R$)"] = 1.0
    df = df[df["Valor (R$)"].fillna(0) >= 0]
    if df["Valor (R$)"].sum() <= 0: return go.Figure()
    if "Descrição" in df.columns: nomes = "Descrição"
    elif "Classe de Ativo" in df.columns: nomes = "Classe de Ativo"
    elif "Classe" in df.columns: nomes = "Classe"
    else:
        df = df.reset_index(drop=True); df["Item"] = [f"Item {i+1}" for i in range(len(df))]; nomes = "Item"
    fig = px.pie(df, values="Valor (R$)", names=nomes, title=title, hole=.35,
                 color_discrete_sequence=PALETA, template=TEMPLATE)
    fig.update_traces(textinfo='percent+label', pull=[0.02]*len(df))
    fig.update_layout(legend_title_text='Classe de Ativo', margin=dict(t=40,b=20,l=0,r=0), showlegend=True)
    return fig

def fig_to_img_html(fig, alt: str) -> str:
    if fig is None:
        return ('<div style="padding:8px;border:1px dashed #ccc;border-radius:8px;color:#666">Sem dados para o gráfico.</div>')
    dom_id = f"figwrap_{uuid.uuid4().hex}"
    fig_json = fig.to_json()
    return f"""
    <div class="figwrap" id="{dom_id}">
      <script type="application/json" class="figspec">{fig_json}</script>
      <div class="ph" style="color:#666">Gerando gráfico…</div>
      <noscript>Ative o JavaScript para visualizar este gráfico.</noscript>
    </div>"""

# =========================
# TOGGLES
# =========================
TIPOS_ATIVO_BASE = ["Debêntures","CRA","CRI","Tesouro Direto","Ações","Fundos de Índice (ETF)","Fundos Imobiliários (FII)",
                    "CDB","LCA","LCI","Renda Fixa Pós-Fixada","Renda Fixa Inflação","Crédito Privado","Previdência Privada","Sintético","Outro"]
TOGGLE_MAP = {
    "Crédito Privado": {"Debêntures","CRA","CRI","Crédito Privado"},
    "Previdência Privada": {"Previdência Privada"},
    "Fundos Imobiliários": {"Fundos Imobiliários (FII)"},
    "Ações e Fundos de Índice": {"Ações","Fundos de Índice (ETF)"},
}
TOGGLE_ALL = set().union(*TOGGLE_MAP.values())

def tipos_permitidos_por_toggles(incluir_credito_privado: bool, incluir_previdencia: bool,
                                 incluir_fii: bool, incluir_acoes_indice: bool) -> set:
    allowed = set(TIPOS_ATIVO_BASE)
    if not incluir_credito_privado: allowed -= TOGGLE_MAP["Crédito Privado"]
    if not incluir_previdencia:     allowed -= TOGGLE_MAP["Previdência Privada"]
    if not incluir_fii:             allowed -= TOGGLE_MAP["Fundos Imobiliários"]
    if not incluir_acoes_indice:    allowed -= TOGGLE_MAP["Ações e Fundos de Índice"]
    return allowed

def filtrar_df_por_toggles(df: pd.DataFrame, allowed_types: set) -> Tuple[pd.DataFrame, int]:
    if df.empty: return df, 0
    mask = df["Tipo"].isin(list(allowed_types))
    removed = int((~mask).sum())
    return df.loc[mask].copy(), removed

# =========================
# SESSION STATE
# =========================
if 'portfolio_atual' not in st.session_state:
    st.session_state.portfolio_atual = pd.DataFrame(columns=[
        "UID","Tipo","Descrição","Indexador","Parâmetro Indexação (% a.a.)","IR (%)","Isento","Rent. 12M (%)","Rent. 6M (%)","Alocação (%)"])
if 'portfolio_personalizado' not in st.session_state:
    st.session_state.portfolio_personalizado = st.session_state.portfolio_atual.copy()
for _k in ('portfolio_atual','portfolio_personalizado'):
    if "UID" not in st.session_state[_k].columns:
        st.session_state[_k].insert(0, "UID", [uuid.uuid4().hex for _ in range(len(st.session_state[_k]))])

# =========================
# SIDEBAR (ÚNICA)
# =========================
def get_focus_defaults():
    """
    Retorna (cdi_aa, ipca_aa, selic_aa, meta) usando caches Focus/BCB e CDI.
    """
    cdi_aa, meta = _cdi_expected_cached()
    focus = _fetch_focus_aa_cached()
    ipca_aa = focus.get("ipca_aa", 4.0)
    selic_aa = focus.get("selic_aa", 12.0)
    if cdi_aa is None:
        cdi_aa = selic_aa
        meta = {"method": "fallback_selic"}
    return float(cdi_aa), float(ipca_aa), float(selic_aa), meta

def _apply_focus_defaults(*, rerun: bool = False, **_):
    """
    Preenche widgets e estados numéricos com Focus/BCB + CDI (SGS).
    """
    cdi_def, ipca_def, selic_def, _meta = get_focus_defaults()

    # Preenche widgets (strings PT-BR) que o form lê
    st.session_state["cdi_aa_input"]   = _fmt_num_br(cdi_def, 2)
    st.session_state["ipca_aa_input"]  = _fmt_num_br(ipca_def, 2)
    st.session_state["selic_aa_input"] = _fmt_num_br(selic_def, 2)

    # Atualiza valores numéricos
    st.session_state["cdi_aa"]   = float(cdi_def)
    st.session_state["ipca_aa"]  = float(ipca_def)
    st.session_state["selic_aa"] = float(selic_def)

with st.sidebar:
    st.markdown(
        """<div style="display:flex;align-items:center;gap:10px;margin-top:-8px;margin-bottom:-6px">
        <div style="font-size:46px;line-height:1">📊</div>
        <div style="font-weight:600;font-size:18px">Parâmetros do Cliente</div></div>""",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # --- Parâmetros de Mercado (a.a.) ---
    st.subheader("Parâmetros de Mercado (a.a.)")

    # Prefill inicial 1x
    if not st.session_state.get("__focus_prefilled__", False):
        st.session_state["__focus_prefilled__"] = True
        st.session_state.setdefault("__side_use_focus__", True)
        _apply_focus_defaults()

    # Toggle (mantido apenas para exibir/ocultar auto-preenchimento)
    st.checkbox(
        "Usar Focus/BCB para preencher automaticamente",
        key="__side_use_focus__",
        value=st.session_state.get("__side_use_focus__", True),
        on_change=_apply_focus_defaults
    )

    # Botão para atualizar Focus/BCB + CDI agora (limpa caches)
    if st.button("🔄 Atualizar Focus/BCB agora", use_container_width=True):
        try: _fetch_focus_aa_cached.clear()
        except Exception: pass
        try: _cdi_expected_cached.clear()
        except Exception: pass
        _apply_focus_defaults()
        st.success("Parâmetros atualizados com sucesso.")
        st.rerun()

    # ---------- FORM ----------
    with st.form("sidebar_params", clear_on_submit=False):
        # Nome
        nome_cliente_input = st.text_input(
            "Nome do Cliente",
            st.session_state.get("nome_cliente", "Cliente Exemplo")
        )

        # PDF (carrega 1x e guarda na sessão)
        st.subheader("Carteiras Sugeridas (PDF)")
        pdf_upload = st.file_uploader(
            "Anexar PDF", type=["pdf"],
            help="Opcional: anexe o PDF de carteiras sugeridas."
        )
        default_pdf_path = "/Users/macvini/Library/CloudStorage/OneDrive-Pessoal/Repos/Portfoliza/Materiais/CarteiraSugeridaBB.pdf"
        pdf_bytes, pdf_msg = load_pdf_bytes_once(pdf_upload, default_pdf_path)
        st.caption(pdf_msg)

        # Inputs (usam os valores que o callback gravou no session_state)
        # Obs.: number_input_allow_blank trata PT-BR e mantém 2 casas
        cdi_def, ipca_def, selic_def, _meta_debug = get_focus_defaults()

        cdi_aa_input = number_input_allow_blank(
            "CDI esperado (% a.a.)",
            st.session_state.get("cdi_aa", cdi_def),
            key="cdi_aa_input",
            help="Usado para 'Pós CDI'"
        )
        ipca_aa_input = number_input_allow_blank(
            "IPCA esperado (% a.a.)",
            st.session_state.get("ipca_aa", ipca_def),
            key="ipca_aa_input",
            help="Usado para 'IPCA+'"
        )
        selic_aa_input = number_input_allow_blank(
            "Selic esperada (% a.a.)",
            st.session_state.get("selic_aa", selic_def),
            key="selic_aa_input",
            help="Exibição (não altera cálculos)."
        )

        # Botão "Aplicar parâmetros" grava os valores numéricos oficiais
        submit_params = st.form_submit_button("Aplicar parâmetros")
        if submit_params:
            st.session_state["nome_cliente"] = nome_cliente_input
            st.session_state["cdi_aa"]   = float(cdi_aa_input or 0.0)
            st.session_state["ipca_aa"]  = float(ipca_aa_input or 0.0)
            st.session_state["selic_aa"] = float(selic_aa_input or 0.0)

@st.cache_data(ttl=24*3600, show_spinner=False)
def extrair_carteiras_do_pdf_cached(pdf_bytes: Optional[bytes]) -> dict:
    # Fallback imediato
    if not pdf_bytes or not HAS_PDFPLUMBER:
        return DEFAULT_CARTEIRAS

    perfis_validos = {"conservador": "Conservador", "moderado": "Moderado", "arrojado": "Arrojado"}
    classes_validas = {
        "renda fixa pós-fixada": "Renda Fixa Pós-Fixada",
        "renda fixa pos-fixada": "Renda Fixa Pós-Fixada",
        "renda fixa inflação": "Renda Fixa Inflação",
        "renda fixa inflacao": "Renda Fixa Inflação",
        "crédito privado": "Crédito Privado",
        "credito privado": "Crédito Privado",
        "fundos imobiliários": "Fundos Imobiliários",
        "ações e fundos de índice": "Ações e Fundos de Índice",
        "acoes e fundos de indice": "Ações e Fundos de Índice",
        "previdência privada": "Previdência Privada",
        "previdencia privada": "Previdência Privada",
    }

    try:
        import re, io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        if not text.strip():
            return DEFAULT_CARTEIRAS

        # Normaliza texto
        tnorm = re.sub(r"[ \t]+", " ", text.lower())
        tnorm = tnorm.replace("%", " %")

        # Quebra por seções de perfil
        # Aceita variações como "perfil conservador", "conservador", etc.
        sections = {}
        for k_norm, k_title in perfis_validos.items():
            # pega tudo entre o cabeçalho do perfil e o próximo perfil (ou fim)
            pat = rf"(?:perfil\s+)?{k_norm}\b(.+?)(?=(?:perfil\s+)?conservador\b|(?:perfil\s+)?moderado\b|(?:perfil\s+)?arrojado\b|$)"
            m = re.search(pat, tnorm, flags=re.DOTALL)
            if m:
                sections[k_title] = m.group(1)
        if not sections:
            return DEFAULT_CARTEIRAS

        def parse_aloc(sec_text: str) -> dict:
            aloc = {}
            for raw, cname in classes_validas.items():
                # padrões como "renda fixa inflação .... 25 %" ou "renda fixa inflação: 25%"
                pat = rf"{raw}[^0-9]{{0,20}}(\d+(?:[.,]\d+)?)\s*%"
                mm = re.search(pat, sec_text, flags=re.IGNORECASE)
                if mm:
                    v = mm.group(1).replace(".", "").replace(",", ".")
                    try:
                        aloc[cname] = float(v)/100.0
                    except Exception:
                        pass
            # Normaliza se somou > 0
            s = sum(aloc.values())
            if s > 0:
                aloc = {k: v/s for k, v in aloc.items()}
            return aloc

        out = {}
        for perfil, sec in sections.items():
            aloc = parse_aloc(sec)
            if not aloc:
                # Se não achou nada neste perfil, mantém fallback por perfil
                out[perfil] = DEFAULT_CARTEIRAS.get(perfil, DEFAULT_CARTEIRAS["Moderado"])
            else:
                # Se achou, mantém uma rentabilidade esperada padrão coerente, ou tente inferir do PDF se existir
                base_ret = DEFAULT_CARTEIRAS.get(perfil, DEFAULT_CARTEIRAS["Moderado"])["rentabilidade_esperada_aa"]
                out[perfil] = {"rentabilidade_esperada_aa": base_ret, "alocacao": aloc}

        # Garante que todos os três estejam presentes
        for p in ["Conservador","Moderado","Arrojado"]:
            if p not in out:
                out[p] = DEFAULT_CARTEIRAS[p]

        return out
    except Exception:
        return DEFAULT_CARTEIRAS

# ------------------------- VARS USADAS FORA -------------------------
nome_cliente = st.session_state.get("nome_cliente", "Cliente Exemplo")
cdi_aa   = float(st.session_state.get("cdi_aa",   get_focus_defaults()[0]))
ipca_aa  = float(st.session_state.get("ipca_aa",  get_focus_defaults()[1]))
selic_aa = float(st.session_state.get("selic_aa", get_focus_defaults()[2]))

# =========================
# PERFIL DE INVESTIMENTO (EXEMPLO: definir aqui ou obter do usuário)
# =========================
perfil_investimento = st.session_state.get("perfil_investimento", "Moderado")
prazo_meses = st.session_state.get("prazo_meses", 24)  # Defina um valor padrão ou obtenha do usuário

# =========================
# HEADER + STRIP
# =========================
st.title(f"💹 Análise de Portfólio — {nome_cliente}")
st.caption(f"Perfil selecionado: **{perfil_investimento}** • Prazo: **{prazo_meses} meses**")
render_market_strip(cdi_aa=cdi_aa, ipca_aa=ipca_aa, selic_aa=selic_aa)

# =========================
# DIAGNÓSTICO (DEBUG) — oculto para usuários
# =========================
if DEBUG_MODE:
    with st.expander("Diagnóstico BCB/Focus (debug)", expanded=False):
        cdi_used, ipca_used, selic_used, meta_used = get_focus_defaults()
        st.write({
            "CDI_app_%aa": round(cdi_used, 2),
            "CDI_calc_method": meta_used.get("method"),
            "CDI_calc_window": meta_used.get("window"),
            "Focus_Selic_%aa": round(selic_used, 2),
            "Focus_IPCA_%aa": round(ipca_used, 2),
            "HAS_BCB": HAS_BCB
        })
        st.markdown("**Validação:** " + ("✅ CDI ≤ Selic" if cdi_used <= selic_used else "⚠️ CDI > Selic (investigar)"))

# =========================
# CARTEIRA SUGERIDA
# =========================
carteiras_from_pdf = extrair_carteiras_do_pdf_cached(pdf_bytes)
carteira_base = carteiras_from_pdf[perfil_investimento]
aloc_sugerida = carteira_base["alocacao"].copy()
# Defina os toggles de inclusão de classes de ativos (pode ser via sidebar ou valores padrão)
incluir_credito_privado = st.session_state.get("incluir_credito_privado", True)
incluir_fundos_imobiliarios = st.session_state.get("incluir_fundos_imobiliarios", True)
incluir_acoes_indice = st.session_state.get("incluir_acoes_indice", True)
incluir_previdencia = st.session_state.get("incluir_previdencia", True)

# Defina valor_inicial e aportes_mensais e meta_financeira (ajuste os valores padrão conforme necessário)
valor_inicial = st.session_state.get("valor_inicial", 100000.0)
aportes_mensais = st.session_state.get("aportes_mensais", 2000.0)
meta_financeira = st.session_state.get("meta_financeira", 150000.0)

toggle_flags = {
    "Crédito Privado": incluir_credito_privado,
    "Fundos Imobiliários": incluir_fundos_imobiliarios,
    "Ações e Fundos de Índice": incluir_acoes_indice,
    "Previdência Privada": incluir_previdencia,
}
for classe, flag in toggle_flags.items():
    if flag:
        if classe == "Previdência Privada" and classe not in aloc_sugerida: aloc_sugerida[classe] = 0.10
    else:
        aloc_sugerida.pop(classe, None)
tot = sum(aloc_sugerida.values()) or 1.0
aloc_sugerida = {k: v/tot for k, v in aloc_sugerida.items()}

df_sugerido = pd.DataFrame(list(aloc_sugerida.items()), columns=["Classe de Ativo","Alocação (%)"])
df_sugerido["Alocação (%)"] = (df_sugerido["Alocação (%)"] * 100).round(2)
df_sugerido["Valor (R$)"]   = (valor_inicial * df_sugerido["Alocação (%)"]/100.0).round(2)

rent_aa_sugerida = carteira_base.get("rentabilidade_esperada_aa", 0.10)
rent_am_sugerida = aa_to_am(rent_aa_sugerida)

ALLOWED_TYPES = tipos_permitidos_por_toggles(incluir_credito_privado, incluir_previdencia,
                                             incluir_fundos_imobiliarios, incluir_acoes_indice)

# =========================
# CACHE AUX
# =========================
@st.cache_data(show_spinner=False)
def _df_normalizar_pesos_cached(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty: return out
    if "Alocação Normalizada (%)" not in out.columns:
        if "Alocação (%)" in out.columns and out["Alocação (%)"].sum() > 0:
            soma = out["Alocação (%)"].sum(); out["Alocação Normalizada (%)"] = out["Alocação (%)"]/soma*100.0
    return out

def taxa_aa_from_indexer(indexador: str, par_idx: float, cdi_aa: float, ipca_aa: float) -> float:
    if indexador == "Pós CDI": return (par_idx/100.0) * (cdi_aa/100.0)
    elif indexador == "Prefixado": return par_idx/100.0
    else: return (ipca_aa/100.0) + (par_idx/100.0)

def taxa_portfolio_aa(df: pd.DataFrame, cdi_aa: float, ipca_aa: float, apply_tax: bool=False) -> float:
    if df is None or df.empty: return 0.0
    rows_taxas_pesos = []
    for _, r in df.iterrows():
        w = None
        if pd.notna(r.get("Alocação Normalizada (%)", np.nan)):
            try: w = float(r["Alocação Normalizada (%)"])/100.0
            except Exception: w = None
        elif pd.notna(r.get("Alocação (%)", np.nan)):
            try: w = float(r["Alocação (%)"])/100.0
            except Exception: w = None
        if w is None or not np.isfinite(w) or w <= 0: continue

        idx = str(r.get("Indexador","Pós CDI") or "Pós CDI")
        par = r.get("Parâmetro Indexação (% a.a.)", 0.0)
        try: par = float(par)
        except Exception: par = 0.0
        if pd.isna(par) or not np.isfinite(par): par = 0.0

        taxa = taxa_aa_from_indexer(idx, par, cdi_aa, ipca_aa)
        if apply_tax and not bool(r.get("Isento", False)):
            ir_raw = r.get("IR (%)", 0.0)
            try: ir = float(ir_raw)
            except Exception: ir = 0.0
            if np.isfinite(ir) and ir > 0: taxa = taxa * (1 - ir/100.0)
        if pd.isna(taxa) or not np.isfinite(taxa): continue
        rows_taxas_pesos.append((taxa, w))
    if not rows_taxas_pesos: return 0.0
    taxas, pesos = zip(*rows_taxas_pesos)
    return float(np.average(np.array(taxas), weights=np.array(pesos)))

@st.cache_data(show_spinner=False)
def _taxa_portfolio_aa_cached(df: pd.DataFrame, cdi_aa: float, ipca_aa: float, apply_tax: bool=False) -> float:
    return taxa_portfolio_aa(df, cdi_aa, ipca_aa, apply_tax)

# =========================
# FORM DINÂMICO (INDEXADOR)
# =========================
INDEXADORES = ["Pós CDI","Prefixado","IPCA+"]

def taxa_inputs_group(indexador: str, portfolio_key: str, prefix: str = "") -> float:
    kb = f"{prefix}{portfolio_key}"
    v_cdi  = st.number_input("% do CDI (% a.a.)",        min_value=0.0, value=110.0, step=1.0,  key=f"par_cdi_{kb}",  disabled=(indexador!="Pós CDI"))
    v_pre  = st.number_input("Taxa Prefixada (% a.a.)",  min_value=0.0, value=14.0, step=0.1,  key=f"par_pre_{kb}",  disabled=(indexador!="Prefixado"))
    v_ipca = st.number_input("Taxa sobre IPCA (% a.a.)", min_value=0.0, value=5.0,  step=0.1,  key=f"par_ipca_{kb}", disabled=(indexador!="IPCA+"))
    return v_cdi if indexador=="Pós CDI" else (v_pre if indexador=="Prefixado" else v_ipca)

def ir_inputs_group(portfolio_key: str, col_sel, col_custom):
    with col_sel:
        ir_opt = st.selectbox("IR", ["Isento","15","17.5","20","22.5","Outro"], key=f"ir_{portfolio_key}")
    with col_custom:
        ir_custom = st.number_input("IR personalizado (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.5,
                                    key=f"irv_{portfolio_key}", disabled=(ir_opt!="Outro"))
    isento = (ir_opt == "Isento")
    ir_pct = 0.0 if isento else (ir_custom if ir_opt=="Outro" else float(ir_opt))
    return ir_pct, isento

def _market_rates_for_autofill_products(cdi_manual_aa: float, ipca_manual_aa: float) -> Tuple[float, float]:
    cdi_used  = float(st.session_state.get("cdi_aa", cdi_manual_aa))
    ipca_used = float(st.session_state.get("ipca_aa", ipca_manual_aa))
    return cdi_used, ipca_used

# =========================
# UTIL: EXCLUSÃO POR UID
# =========================
def _excluir_por_uids(portfolio_key: str, uids: List[str]):
    base = st.session_state.get(portfolio_key, pd.DataFrame())
    if base.empty or "UID" not in base.columns: return
    mask = ~base["UID"].astype(str).isin([str(u) for u in uids])
    st.session_state[portfolio_key] = base.loc[mask].reset_index(drop=True)

# =========================
# FORMULÁRIO DE PORTFÓLIO
# =========================
def form_portfolio(portfolio_key: str, titulo: str, allowed_types: set):
    st.subheader(titulo)

    tipos_visiveis = [t for t in TIPOS_ATIVO_BASE if (t in allowed_types) or (t not in TOGGLE_ALL)]
    dfp = st.session_state[portfolio_key]

    # garante UID
    if "UID" not in dfp.columns:
        st.session_state[portfolio_key].insert(0, "UID", [uuid.uuid4().hex for _ in range(len(dfp))])
        dfp = st.session_state[portfolio_key]

    with st.expander("Adicionar/Remover Ativos", expanded=True if dfp.empty else False):
        c = st.columns(9)
        tipo      = c[0].selectbox("Tipo", tipos_visiveis, key=f"tipo_{portfolio_key}")
        desc      = c[1].text_input("Descrição", key=f"desc_{portfolio_key}")
        indexador = c[2].selectbox("Indexador", INDEXADORES, key=f"idx_{portfolio_key}")

        with c[3]:
            par_idx = taxa_inputs_group(indexador, portfolio_key)
            st.caption("O campo de taxa habilitado depende do indexador.")

        # Autofill Focus/CDI/IPCA p/ 12M/6M
        try:
            cdi_auto_aa, ipca_auto_aa = _market_rates_for_autofill_products(
                st.session_state.get("cdi_aa", cdi_aa),
                st.session_state.get("ipca_aa", ipca_aa),
            )
        except Exception:
            cdi_auto_aa, ipca_auto_aa = cdi_aa, ipca_aa

        taxa_auto_aa_frac = taxa_aa_from_indexer(indexador, par_idx, cdi_auto_aa, ipca_auto_aa)
        r12_auto = float(np.clip(round(taxa_auto_aa_frac * 100.0, 2), 0.0, None))
        r6_auto  = float(np.clip(round(((1.0 + taxa_auto_aa_frac) ** 0.5 - 1.0) * 100.0, 2), 0.0, None))

        _drv_key = f"__auto_fill_state__{portfolio_key}"
        _drv_val = (indexador, float(par_idx), round(cdi_auto_aa, 4), round(ipca_auto_aa, 4))
        if st.session_state.get(_drv_key) != _drv_val:
            st.session_state[_drv_key] = _drv_val
            st.session_state[f"r12_{portfolio_key}"] = r12_auto
            st.session_state[f"r6_{portfolio_key}"]  = r6_auto

        ir_pct, isento = ir_inputs_group(portfolio_key, c[4], c[5])

        r12  = c[6].number_input(
            "Rent. 12M (%)", min_value=0.0,
            value=float(st.session_state.get(f"r12_{portfolio_key}", r12_auto)),
            step=0.1, key=f"r12_{portfolio_key}"
        )
        r6   = c[7].number_input(
            "Rent. 6M (%)", min_value=0.0,
            value=float(st.session_state.get(f"r6_{portfolio_key}", r6_auto)),
            step=0.1, key=f"r6_{portfolio_key}"
        )
        aloc = c[8].number_input("Alocação (%)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, key=f"aloc_{portfolio_key}")

        if st.button("Adicionar Ativo", key=f"add_{portfolio_key}"):
            if desc.strip():
                novo = pd.DataFrame([{
                    "UID": uuid.uuid4().hex,
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

        # --------- LISTAGEM ----------
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
                gob.configure_selection('multiple', use_checkbox=True)
                gob.configure_grid_options(domLayout='autoHeight')
                if "UID" in dfp_filt.columns:
                    gob.configure_column("UID", hide=True)
                grid = AgGrid(
                    dfp_filt, gridOptions=gob.build(),
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    theme='streamlit', fit_columns_on_grid_load=True,
                    key=f"grid_{portfolio_key}"
                )
                sel = grid.get("selected_rows")
                sel_df = pd.DataFrame(sel) if isinstance(sel, list) else sel
                if sel_df is not None and len(sel_df) > 0 and st.button("Excluir selecionado(s) (AgGrid)", key=f"del_{portfolio_key}"):
                    if "UID" in sel_df.columns:
                        _excluir_por_uids(portfolio_key, sel_df["UID"].astype(str).unique().tolist())
                    else:
                        base = st.session_state[portfolio_key]
                        tgt = base[base["Descrição"].astype(str).isin(sel_df["Descrição"].astype(str).unique().tolist())]["UID"].astype(str).tolist()
                        _excluir_por_uids(portfolio_key, tgt)
                    st.rerun()

            # ---------- Exclusão SEMPRE visível (alternativa) ----------
            st.markdown("**Excluir ativos**")
            _opts = dfp_filt[["UID","Descrição"]].copy() if "UID" in dfp_filt.columns else dfp_filt.assign(UID=dfp_filt["Descrição"])
            _labels = [f"{r['Descrição']}" for _, r in _opts.iterrows()]
            _map_lbl_uid = dict(zip(_labels, _opts["UID"]))
            _pick = st.multiselect("Selecionar ativos para excluir", _labels, key=f"msdel_any_{portfolio_key}")
            if st.button("Excluir selecionados", key=f"btn_del_any_{portfolio_key}") and _pick:
                _uids = [str(_map_lbl_uid[l]) for l in _pick if l in _map_lbl_uid]
                _excluir_por_uids(portfolio_key, _uids)
                st.rerun()

            fig = criar_grafico_alocacao(
                dfp_filt.rename(columns={"Tipo":"Classe","Descrição":"Descrição"}), f"Alocação — {titulo}"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"chart_aloc_{portfolio_key}")

            if soma > 100.1 or soma < 99.9:
                st.warning(f"A soma da alocação é {_fmt_num_br(soma,2)}%. Os valores foram normalizados para 100%.")

            colb = st.columns(2)
            with colb[0]:
                if st.button(f"Limpar {titulo}", key=f"clear_{portfolio_key}"):
                    cols = st.session_state[portfolio_key].columns.tolist()
                    st.session_state[portfolio_key] = pd.DataFrame(columns=cols)
                    st.rerun()
            with colb[1]:
                if not HAS_AGGRID:
                    st.caption("Dica: instale `streamlit-aggrid` para clique direto na linha.")

        return dfp_filt if 'dfp_filt' in locals() else pd.DataFrame()

# =========================
# ABAS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Projeção & Carteira Sugerida","💼 Portfólio Atual","🎨 Personalizar Carteira","📊 Comparativos","📋 Relatório"])

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
    st.plotly_chart(fig_proj, use_container_width=True, key="chart_proj")

    st.subheader(f"Alocação Sugerida — Perfil {perfil_investimento}")
    styled_sug = style_df_br(df_sugerido, money_cols=["Valor (R$)"], pct100_cols=["Alocação (%)"])
    st.dataframe(maybe_hide_index(styled_sug), use_container_width=True)
    fig_aloc_sugerida = criar_grafico_alocacao(df_sugerido.rename(columns={"Classe de Ativo":"Descrição"}), "Alocação da Carteira Sugerida")
    st.plotly_chart(fig_aloc_sugerida, use_container_width=True, key="chart_aloc_sugerida")

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
# Preparos COMPARATIVOS
# =========================
df_atual_state = st.session_state.get('portfolio_atual', pd.DataFrame())
df_atual_for_rate = _df_normalizar_pesos_cached(df_atual_state)
rent_atual_aa_liq = _taxa_portfolio_aa_cached(df_atual_for_rate, cdi_aa, ipca_aa, apply_tax=True)

df_pers_state = st.session_state.get('portfolio_personalizado', pd.DataFrame())
df_pers_for_rate = _df_normalizar_pesos_cached(df_pers_state)
rent_pers_aa_liq = _taxa_portfolio_aa_cached(df_pers_for_rate, cdi_aa, ipca_aa, apply_tax=True)

# Defina a alíquota de IR padrão para a carteira sugerida (ajuste conforme necessário)
ir_eq_sugerida = 15.0  # exemplo: 15% de IR

rent_sugerida_aa_liq = rent_aa_sugerida * (1 - ir_eq_sugerida/100.0)

# =========================
# ABA 4 — COMPARATIVOS
# =========================
with tab4:
    st.subheader("Comparativo de Projeção (líquido de IR)")
    ir_cdi = ir_eq_sugerida  # Define IR para CDI (ajuste conforme necessário)
    cdi_liq_aa = (cdi_aa/100.0) * (1 - ir_cdi/100.0)
    monthly_rates = {
        "Carteira Sugerida (líquida)":        safe_aa_to_am(rent_sugerida_aa_liq),
        "Portfólio Atual (líquido)":          safe_aa_to_am(rent_atual_aa_liq),
        "Portfólio Personalizado (líquido)":  safe_aa_to_am(rent_pers_aa_liq),
        "CDI líquido de IR":                  safe_aa_to_am(cdi_liq_aa),
    }
    df_comp = pd.DataFrame({'Mês': range(25)})
    for nome, taxa_m in monthly_rates.items():
        df_comp[nome] = calcular_projecao(valor_inicial, aportes_mensais, taxa_m, 24)

    desired_order = ["CDI líquido de IR","Carteira Sugerida (líquida)","Portfólio Personalizado (líquido)","Portfólio Atual (líquido)"]
    df_comp = df_comp[["Mês"] + [c for c in desired_order if c in df_comp.columns]]

    # Se houver sobreposição perfeita, desloca levemente para não "sumir" a linha
    tol_r, tol_a = 1e-10, 1e-6
    if "CDI líquido de IR" in df_comp.columns:
        base = df_comp["CDI líquido de IR"].to_numpy(dtype=float)
        for col in [c for c in df_comp.columns if c not in ("Mês","CDI líquido de IR")]:
            arr = df_comp[col].to_numpy(dtype=float)
            if np.allclose(arr, base, rtol=tol_r, atol=tol_a): df_comp[col] = arr + 0.25

    fig_comp = criar_grafico_projecao(df_comp, "Projeção — Líquido de Impostos (24 meses)")
    for tr in fig_comp.data:
        if tr.name == "CDI líquido de IR": tr.update(line=dict(dash="dot", width=2))
        if tr.name == "Portfólio Atual (líquido)": tr.update(line=dict(width=4))
    st.plotly_chart(fig_comp, use_container_width=True, key="chart_comp")
    st.session_state['fig_comp'] = fig_comp

    linhas = []
    for nome in [c for c in desired_order if c in df_comp.columns and c != "Mês"]:
        taxa_m = monthly_rates[nome]; taxa_aa = (1 + float(taxa_m)) ** 12 - 1; valor_12m = valor_inicial * (1 + taxa_aa)
        linhas.append({"Cenário": nome, "Rent. 12M (a.a.) Líquida": taxa_aa, "Resultado estimado em 12M (R$)": valor_12m})
    df_resumo = pd.DataFrame(linhas)
    styled_resumo = style_df_br(df_resumo, money_cols=["Resultado estimado em 12M (R$)"], pct_cols=["Rent. 12M (a.a.) Líquida"])
    st.dataframe(maybe_hide_index(styled_resumo), use_container_width=True)

# =========================
# ABA 5 — RELATÓRIO
# =========================
def build_html_report(nome: str, perfil: str, prazo_meses: int, valor_inicial: float, aportes: float, meta: float,
                      df_sug_classe: pd.DataFrame, df_produtos: pd.DataFrame, email_text_html: str,
                      fig_comp_placeholder: str, fig_aloc_atual_placeholder: str,
                      fig_aloc_pers_placeholder: str, fig_aloc_sug_placeholder: str) -> str:
    df_sug = df_sug_classe.copy()
    if "Valor (R$)" in df_sug.columns: df_sug["Valor (R$)"] = df_sug["Valor (R$)"].map(fmt_brl)
    if "Alocação (%)" in df_sug.columns: df_sug["Alocação (%)"] = df_sug["Alocação (%)"].map(fmt_pct100_br)

    cols_prod_all = ["Tipo","Descrição","Indexador","Parâmetro Indexação (% a.a.)","IR (%)","Isento",
                     "Rent. 12M (%)","Rent. 6M (%)","Alocação (%)","Alocação Normalizada (%)","Valor (R$)"]
    cols_prod = [c for c in cols_prod_all if c in df_produtos.columns]
    df_prod = (df_produtos[cols_prod].copy() if cols_prod else pd.DataFrame())
    for c in ["IR (%)","Rent. 12M (%)","Rent. 6M (%)","Alocação (%)","Alocação Normalizada (%)"]:
        if c in df_prod.columns: df_prod[c] = df_prod[c].map(fmt_pct100_br)
    if "Parâmetro Indexação (% a.a.)" in df_prod.columns: df_prod["Parâmetro Indexação (% a.a.)"] = df_prod["Parâmetro Indexação (% a.a.)"].map(lambda x: _fmt_num_br(x,2))
    if "Valor (R$)" in df_prod.columns: df_prod["Valor (R$)"] = df_prod["Valor (R$)"].map(fmt_brl)

    tabela_sug_classe = df_sug[["Classe de Ativo","Alocação (%)","Valor (R$)"]].to_html(index=False, border=0)
    tabela_produtos = df_prod.to_html(index=False, border=0)

    style = """
    <style>
      body { font-family:-apple-system, Segoe UI, Roboto, Arial, sans-serif; color:#222; padding:12px; }
      .card { border:1px solid #e5e5e5; border-radius:12px; padding:16px; margin:12px 0; }
      h1,h2,h3 { margin:0.2rem 0 0.6rem; } table { width:100%; border-collapse:collapse; table-layout:auto; }
      th,td { padding:8px 10px; border-bottom:1px solid #eee; text-align:left; white-space:normal; word-break:break-word; overflow-wrap:anywhere; hyphens:auto; font-size:14px; }
      .muted { color:#666; font-size:0.9rem; } .tag { background:#f3f4f6; border:1px solid #e5e7eb; padding:3px 8px; border-radius:999px; font-size:0.85rem; }
      .grid { display:grid; grid-template-columns:1fr; gap:16px; } @media (min-width:1024px){ .grid-2 { grid-template-columns:1fr 1fr; } }
      .highlight { background:#fff7ed; border-color:#fdba74; } .imgwrap { text-align:center; } .imgwrap img { max-width:640px; width:640px; height:auto; }
    </style>"""
    html_report = f"""{style}
    <div id="report-root">
      <h1>Relatório — Análise de Portfólio</h1>
      <div class="muted">Cliente: <span class="tag">{nome}</span> • Perfil: <span class="tag">{perfil}</span> • Prazo: <span class="tag">{prazo_meses} meses</span></div>
      {"<div class='card'><h2>Mensagem do Assessor</h2><div class='muted'>Conteúdo preparado para e-mail</div><div style='margin-top:6px'>" + email_text_html + "</div></div>" if email_text_html else ""}
      <div class="card"><h2>Dados Iniciais</h2><ul>
        <li><b>Valor Inicial:</b> {fmt_brl(valor_inicial)}</li>
        <li><b>Aportes Mensais:</b> {fmt_brl(aportes)}</li>
        <li><b>Meta Financeira:</b> {fmt_brl(meta)}</li></ul></div>
      <div class="card highlight"><h2>Carteira Sugerida — Alocação por Classe (Destaque)</h2>{tabela_sug_classe}</div>
      <div class="card"><h2>Produtos Selecionados (Portfólio Sugerido ao Cliente)</h2>{tabela_produtos}</div>
      <div class="card"><h2>Comparativo de Projeção (líquido de IR)</h2><div class="imgwrap">{fig_comp_placeholder}</div></div>
      <div class="card"><h2>Alocações — Antes e Depois</h2>
        <div class="grid grid-2"><div><h3>Portfólio Atual</h3><div class="imgwrap">{fig_aloc_atual_placeholder}</div></div>
        <div><h3>Portfólio Personalizado</h3><div class="imgwrap">{fig_aloc_pers_placeholder}</div></div></div>
        <div style="margin-top:12px"><h3>Carteira Sugerida</h3><div class="imgwrap">{fig_aloc_sug_placeholder}</div></div>
      </div>
      <div class="card"><h3>Avisos Importantes</h3>
        <p class="muted">Os resultados simulados são ilustrativos, não configuram garantia de rentabilidade futura.
        As projeções foram consideradas líquidas de IR conforme parâmetros informados/estimados no aplicativo.
        Leia os documentos dos produtos antes de investir.</p></div>
    </div>"""
    return html_report

fig_aloc_atual_rep = criar_grafico_alocacao(st.session_state.get('portfolio_atual', pd.DataFrame()).rename(columns={"Tipo":"Classe","Descrição":"Descrição"}), "Alocação — Portfólio Atual")
fig_aloc_pers_rep  = criar_grafico_alocacao(st.session_state.get('portfolio_personalizado', pd.DataFrame()).rename(columns={"Tipo":"Classe","Descrição":"Descrição"}), "Alocação — Portfólio Personalizado")
fig_aloc_sug_rep   = criar_grafico_alocacao(pd.DataFrame(list(aloc_sugerida.items()), columns=["Descrição","Valor (R$)"]), "Alocação — Carteira Sugerida")

comp_img = fig_to_img_html(st.session_state.get('fig_comp', None), "Projeção Líquida")
atual_img = fig_to_img_html(fig_aloc_atual_rep, "Alocação — Portfólio Atual")
pers_img  = fig_to_img_html(fig_aloc_pers_rep, "Alocação — Portfólio Personalizado")
sug_img   = fig_to_img_html(fig_aloc_sug_rep, "Alocação — Carteira Sugerida")

with tab5:
    st.subheader("Relatório (copiar conteúdo formatado)")
    st.caption("Os gráficos são convertidos automaticamente em imagens PNG no momento da cópia.")
    email_msg = st.text_area("Mensagem do e-mail (edite aqui)",
                             value="Olá, tudo bem? Segue abaixo a análise e a sugestão de carteira preparada conforme seu perfil e objetivos.",
                             height=140, help="Este conteúdo vai junto no relatório copiado para colar no e-mail.")
    email_msg_html = "<br>".join(html.escape(l) for l in email_msg.splitlines())

    df_prod_report = st.session_state.get('portfolio_personalizado', pd.DataFrame()).copy()
    if not df_prod_report.empty:
        if "Alocação Normalizada (%)" not in df_prod_report.columns and "Alocação (%)" in df_prod_report.columns:
            soma_p = df_prod_report["Alocação (%)"].sum() or 1.0
            df_prod_report["Alocação Normalizada (%)"] = (df_prod_report["Alocação (%)"]/soma_p*100.0).round(2)
        if "Valor (R$)" not in df_prod_report.columns and "Alocação Normalizada (%)" in df_prod_report.columns:
            df_prod_report["Valor (R$)"] = (valor_inicial * df_prod_report["Alocação Normalizada (%)"]/100.0).round(2)

    html_report = build_html_report(nome_cliente, perfil_investimento, prazo_meses, valor_inicial, aportes_mensais, meta_financeira,
                                    df_sugerido, df_prod_report, email_msg_html,
                                    fig_comp_placeholder=comp_img, fig_aloc_atual_placeholder=atual_img,
                                    fig_aloc_pers_placeholder=pers_img, fig_aloc_sug_placeholder=sug_img)

    copy_block = f"""
    <div>{html_report}
      <div style="margin-top:10px">
        <button id="cpy" style="background:#0b1221;color:#e5e7eb;border:1px solid #1f2937;padding:10px 14px;border-radius:8px;cursor:pointer">
          Copiar conteúdo formatado
        </button>
        <span id="cpyst" style="margin-left:10px;color:#10b981"></span>
      </div>
    </div>
    <script>
      (function(){{
        function ensurePlotly(){{return new Promise(function(resolve,reject){{if(window.Plotly)return resolve();var s=document.createElement('script');s.src='https://cdn.plot.ly/plotly-2.35.3.min.js';s.onload=function(){{resolve();}};s.onerror=function(){{reject(new Error('Falha ao carregar plotly.js'));}};document.head.appendChild(s);}})}}
        let rendered=false;
        async function renderAll(){{
          await ensurePlotly();
          const wraps=Array.from(document.querySelectorAll('.figwrap'));
          for(const w of wraps){{
            const specEl=w.querySelector('.figspec'); if(!specEl) continue;
            let fig; try{{fig=JSON.parse(specEl.textContent);}}catch(e){{w.innerHTML='<div style="padding:8px;border:1px dashed #ccc;border-radius:8px;color:#666">Erro ao ler gráfico.</div>';continue;}}
            const div=document.createElement('div'); div.style.width='100%'; div.style.maxWidth='640px'; div.style.margin='0 auto';
            w.innerHTML=''; w.appendChild(div);
            try{{ await Plotly.newPlot(div, fig.data||[], fig.layout||{{}}, {{staticPlot:true, displayModeBar:false}});
                  const url=await Plotly.toImage(div, {{format:'png', scale:2}});
                  w.innerHTML='<img src=\"'+url+'\" style=\"max-width:100%;height:auto;border:1px solid #eee;border-radius:12px\" />';}}
            catch(e){{ w.innerHTML='<div style="padding:8px;border:1px dashed #ccc;border-radius:8px;color:#666">Falha ao gerar imagem.</div>'; }}
          }} rendered=true;
        }}
        async function copyHtml(){{
          if(!rendered) await renderAll();
          const root=document.getElementById('report-root'); if(!root) return;
          const html=root.outerHTML;
          try{{
            if(navigator.clipboard && window.ClipboardItem){{
              const item=new ClipboardItem({{'text/html': new Blob([html], {{type:'text/html'}}),
                                           'text/plain': new Blob([root.innerText], {{type:'text/plain'}})}});
              await navigator.clipboard.write([item]);
            }} else {{
              const sel=window.getSelection(); const range=document.createRange();
              range.selectNode(root); sel.removeAllRanges(); sel.addRange(range);
              document.execCommand('copy'); sel.removeAllRanges();
            }}
            document.getElementById('cpyst').textContent='Conteúdo copiado para a área de transferência!';
            setTimeout(()=> document.getElementById('cpyst').textContent='', 3000);
          }} catch(e) {{ document.getElementById('cpyst').textContent='Falha ao copiar'; }}
        }}
        renderAll(); document.getElementById('cpy').addEventListener('click', copyHtml);
      }})();
    </script>"""
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
