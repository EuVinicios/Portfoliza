# app.py
from __future__ import annotations
import io, os, re, base64, math, json
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------- CONFIGURAÇÃO DA PÁGINA ----------
st.set_page_config(
    page_title="Gerador de Portfólios de Investimento",
    page_icon="💹",
    layout="wide"
)

# ---------- TENTATIVAS DE IMPORT OPCIONAIS ----------
HAS_PDFPLUMBER = False
HAS_WEASYPRINT = False
HAS_AGGRID = False
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    pass

try:
    from weasyprint import HTML  # precisa do Cairo instalado no SO
    HAS_WEASYPRINT = True
except Exception:
    pass

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    HAS_AGGRID = True
except Exception:
    pass

# ---------- CORES E TEMA DE GRÁFICO ----------
PALETA = px.colors.qualitative.Vivid
TEMPLATE = "plotly_white"

# ---------- MOCK/DEFAULT (USADO QUANDO PDF NÃO FOR COMPATÍVEL) ----------
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

# ---------- HELPERS DE NUMÉRICO VAZIO ----------
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
    """Input que permite apagar (vazio vira 0)."""
    placeholder = f"{default:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    val_str = st.text_input(label, value=placeholder, key=key, help=help)
    return _parse_float(val_str, default=0.0)

# ---------- PDF: CARREGAR E EXIBIR ----------
def load_pdf_bytes(
    uploaded_file, default_path: Optional[str]
) -> Tuple[Optional[bytes], str]:
    if uploaded_file is not None:
        return uploaded_file.read(), "PDF carregado por upload."
    if default_path and os.path.exists(default_path):
        with open(default_path, "rb") as f:
            return f.read(), f"PDF carregado do caminho local: {default_path}"
    return None, "Nenhum PDF carregado (usando configurações padrão)."

def pdf_bytes_to_embed_html(pdf_bytes: bytes, height: int=600) -> str:
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"""
    <iframe
        src="data:application/pdf;base64,{b64}#toolbar=1"
        width="100%" height="{height}" style="border:1px solid #ddd; border-radius:8px;">
    </iframe>
    """

# ---------- PDF: PARSE DE ALOCAÇÕES ----------
_CLASSE_NORMALIZAR = {
    # mapeia variações/aliases para nomes padronizados
    r"renda fixa pós.*fixada": "Renda Fixa Pós-Fixada",
    r"p[oó]s[\s\-]*cdi|cdi": "Renda Fixa Pós-Fixada",
    r"renda fixa infla[cç][aã]o|ipca\+?": "Renda Fixa Inflação",
    r"cr[eé]dito privado|deb[eê]ntures|cra|cri": "Crédito Privado",
    r"fundos imobili[aá]rios|fii": "Fundos Imobiliários",
    r"a[cç][oõ]es.*[íi]ndice|etf|fundos de [íi]ndice": "Ações e Fundos de Índice",
    r"previd[eê]ncia": "Previdência Privada",
}
_PERFIS = ["Conservador", "Moderado", "Arrojado"]

def _normalizar_classe(label: str) -> Optional[str]:
    l = label.lower()
    for pat, out in _CLASSE_NORMALIZAR.items():
        if re.search(pat, l, flags=re.I):
            return out
    return None

def extrair_carteiras_do_pdf(pdf_bytes: bytes) -> Dict[str, Dict]:
    """
    Tenta extrair alocações por perfil do PDF.
    Espera encontrar linhas com '<classe> ... 10%' em seções 'Conservador/Moderado/Arrojado'.
    Fallback: DEFAULT_CARTEIRAS.
    """
    if not (pdf_bytes and HAS_PDFPLUMBER):
        return DEFAULT_CARTEIRAS

    try:
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tx = page.extract_text() or ""
                text += "\n" + tx

        # organiza por perfil
        blocos = {}
        for i, perfil in enumerate(_PERFIS):
            start = re.search(perfil, text, flags=re.I)
            if not start:
                continue
            start_idx = start.start()
            end_idx = len(text)
            # define fim na próxima seção
            for j in range(i+1, len(_PERFIS)):
                nxt = re.search(_PERFIS[j], text, flags=re.I)
                if nxt:
                    end_idx = min(end_idx, nxt.start())
            blocos[perfil] = text[start_idx:end_idx]

        carteiras: Dict[str, Dict] = {}
        for perfil, bloco in blocos.items():
            pairs: Dict[str, float] = {}
            for line in bloco.splitlines():
                # busca números percentuais
                m = re.search(r"([A-Za-zÀ-ÿ \-\+\/]+?)\s+(\d{1,3})\s*%", line.strip())
                if m:
                    rotulo = m.group(1).strip()
                    pct = float(m.group(2)) / 100.0
                    classe = _normalizar_classe(rotulo)
                    if classe:
                        pairs[classe] = pairs.get(classe, 0.0) + pct
            # normaliza e define rentabilidade esperada default por perfil (ajuste fino se seu PDF trouxer)
            if pairs:
                soma = sum(pairs.values())
                if soma > 0:
                    pairs = {k: v/soma for k, v in pairs.items()}
                rent = {"Conservador": 0.08, "Moderado": 0.10, "Arrojado": 0.12}.get(perfil, 0.10)
                carteiras[perfil] = {"rentabilidade_esperada_aa": rent, "alocacao": pairs}

        return carteiras if carteiras else DEFAULT_CARTEIRAS
    except Exception:
        return DEFAULT_CARTEIRAS

# ---------- FINANCE HELPERS ----------
def aa_to_am(taxa_aa: float) -> float:
    return (1 + taxa_aa) ** (1/12) - 1

def calcular_projecao(valor_inicial, aportes_mensais, taxa_mensal, prazo_meses: int):
    vals = [valor_inicial]
    for _ in range(prazo_meses):
        vals.append( (vals[-1] + aportes_mensais) * (1 + taxa_mensal) )
    return vals

def criar_grafico_projecao(df, title="Projeção de Crescimento"):
    fig = px.line(
        df, x='Mês', y=[c for c in df.columns if c != 'Mês'],
        title=title, labels={'value': 'Patrimônio (R$)', 'variable': 'Cenário'},
        markers=True, color_discrete_sequence=PALETA, template=TEMPLATE
    )
    fig.update_layout(legend_title_text='Cenários', yaxis_title='Patrimônio (R$)', xaxis_title='Meses')
    return fig

def criar_grafico_alocacao(df, title: str):
    if df.empty or (('Valor (R$)' in df.columns) and df['Valor (R$)'].sum() == 0):
        return go.Figure()
    nomes = 'Descrição' if 'Descrição' in df.columns else 'Classe de Ativo'
    fig = px.pie(
        df, values='Valor (R$)', names=nomes, title=title,
        hole=.35, color_discrete_sequence=PALETA, template=TEMPLATE
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    return fig

def gross_up(net_rate_aa: float, ir_equivalente: float) -> float:
    """Converte taxa líquida isenta em equivalente bruto comparável a trib. (alíquota ex.: 15 -> 0.15)."""
    t = ir_equivalente/100.0
    if t >= 1.0:
        return net_rate_aa
    return net_rate_aa / (1 - t)

# ---------- SESSION STATE INICIAL ----------
if 'portfolio_atual' not in st.session_state:
    st.session_state.portfolio_atual = pd.DataFrame(columns=[
        "Tipo", "Descrição", "Indexador", "Parâmetro Indexação (% a.a.)",
        "IR (%)", "Isento", "Rent. 12M (%)", "Rent. 6M (%)", "Alocação (%)"
    ])
if 'portfolio_personalizado' not in st.session_state:
    st.session_state.portfolio_personalizado = st.session_state.portfolio_atual.copy()

# ---------- SIDEBAR ----------
with st.sidebar:
    # Ícone com fallback
    try:
        st.image("https://img.icons8.com/plasticine/100/000000/stocks-growth.png", width=88)
    except Exception:
        st.markdown("### 💹")

    st.title("Parâmetros do Cliente")
    st.markdown("---")

    nome_cliente = st.text_input("Nome do Cliente", "Cliente Exemplo")

    # PDF Carteiras Sugeridas
    st.subheader("Carteiras Sugeridas (PDF)")
    pdf_upload = st.file_uploader("Anexar PDF", type=["pdf"], help="Opcional: anexe o PDF de carteiras sugeridas.")
    default_pdf_path = "/Users/macvini/Library/CloudStorage/OneDrive-Pessoal/Repos/Portfoliza/Materiais/CarteiraSugeridaBB.pdf"
    pdf_bytes, pdf_msg = load_pdf_bytes(pdf_upload, default_pdf_path)
    st.caption(pdf_msg)

    # Mercado (para indexadores)
    st.subheader("Parâmetros de Mercado (a.a.)")
    cdi_aa = number_input_allow_blank("CDI esperado (% a.a.)", 12.0, key="cdi_aa", help="Usado para 'Pós CDI'")
    ipca_aa = number_input_allow_blank("IPCA esperado (% a.a.)", 4.0, key="ipca_aa", help="Usado para 'IPCA+'")
    ir_equivalente = number_input_allow_blank("IR equivalente para Gross-up (%)", 15.0, key="ir_eq", help="Usado na comparação entre isentos e tributáveis")

    # Perfil
    carteiras_from_pdf = extrair_carteiras_do_pdf(pdf_bytes) if pdf_bytes else DEFAULT_CARTEIRAS
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

included_classes = set()
if incluir_credito_privado: included_classes.add("Crédito Privado")
if incluir_previdencia: included_classes.add("Previdência Privada")
if incluir_fundos_imobiliarios: included_classes.add("Fundos Imobiliários")
if incluir_acoes_indice: included_classes.add("Ações e Fundos de Índice")

# ---------- HEADER ----------
st.title(f"💹 Análise de Portfólio — {nome_cliente}")
st.caption(f"Perfil selecionado: **{perfil_investimento}** • Prazo: **{prazo_meses} meses**")
st.markdown("---")

# ---------- CARTEIRA SUGERIDA (com filtros da sidebar) ----------
carteira_base = carteiras_from_pdf[perfil_investimento]
aloc_sugerida = carteira_base["alocacao"].copy()

# remove classes de acordo com toggles (somente as classes afetadas pelos toggles)
for classe, flag in {
    "Crédito Privado": incluir_credito_privado,
    "Fundos Imobiliários": incluir_fundos_imobiliarios,
    "Ações e Fundos de Índice": incluir_acoes_indice,
    "Previdência Privada": incluir_previdencia,  # se False, remove; se True e não houver, adiciona
}.items():
    if flag:
        if classe == "Previdência Privada" and classe not in aloc_sugerida:
            aloc_sugerida[classe] = 0.10  # adiciona exemplo
    else:
        aloc_sugerida.pop(classe, None)

# normaliza
tot = sum(aloc_sugerida.values()) or 1.0
aloc_sugerida = {k: v/tot for k, v in aloc_sugerida.items()}

df_sugerido = pd.DataFrame(list(aloc_sugerida.items()), columns=["Classe de Ativo", "Alocação (%)"])
df_sugerido["Alocação (%)"] = (df_sugerido["Alocação (%)"] * 100).round(2)
df_sugerido["Valor (R$)"] = (valor_inicial * df_sugerido["Alocação (%)"] / 100.0).round(2)

rent_aa_sugerida = carteira_base.get("rentabilidade_esperada_aa", 0.10)
rent_am_sugerida = aa_to_am(rent_aa_sugerida)

# ---------- ABAS ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Projeção & Carteira Sugerida",
    "💼 Portfólio Atual",
    "🎨 Personalizar Carteira",
    "📊 Comparativos",
    "📋 Relatório"
])

# ---------- ABA 1 ----------
with tab1:
    st.subheader("Projeção da Carteira Sugerida")
    proj_sugerida = calcular_projecao(valor_inicial, aportes_mensais, rent_am_sugerida, prazo_meses)
    df_proj = pd.DataFrame({"Mês": list(range(prazo_meses + 1)), "Carteira Sugerida": proj_sugerida})
    fig_proj = criar_grafico_projecao(df_proj, "Projeção de Crescimento do Patrimônio")
    fig_proj.add_hline(y=meta_financeira, line_dash="dash", line_color="red",
                       annotation_text="Meta Financeira", annotation_position="top left")
    st.plotly_chart(fig_proj, use_container_width=True)

    st.subheader(f"Alocação Sugerida — Perfil {perfil_investimento}")
    c1, c2 = st.columns([1,1])
    with c1:
        st.dataframe(df_sugerido, use_container_width=True, hide_index=True)
    with c2:
        fig_aloc = criar_grafico_alocacao(df_sugerido.rename(columns={"Classe de Ativo":"Descrição"}), "Alocação da Carteira Sugerida")
        st.plotly_chart(fig_aloc, use_container_width=True)

    st.divider()
    if pdf_bytes:
        st.subheader("Referência: Carteiras Sugeridas (PDF)")
        st.components.v1.html(pdf_bytes_to_embed_html(pdf_bytes, height=480), height=500)
        st.download_button("⬇️ Baixar PDF de Referência", data=pdf_bytes, file_name="CarteiraSugeridaBB.pdf", mime="application/pdf")

# ---------- FORM DE PORTFÓLIO (reuso) ----------
TIPOS_ATIVO = [
    "Debêntures","CRA","CRI","Tesouro Direto","Ações",
    "Fundos de Índice (ETF)","Fundos Imobiliários (FII)",
    "CDB","LCA","LCI","Renda Fixa Pós-Fixada","Renda Fixa Inflação",
    "Crédito Privado","Previdência Privada","Sintético","Outro"
]
INDEXADORES = ["Pós CDI", "Prefixado", "IPCA+"]

def form_portfolio(portfolio_key: str, titulo: str):
    st.subheader(titulo)
    dfp = st.session_state[portfolio_key]

    with st.expander("Adicionar/Remover Ativos", expanded=True if dfp.empty else False):
        with st.form(f"form_{portfolio_key}", clear_on_submit=True):
            c = st.columns((1.6, 2.4, 1.1, 1.2, 1.0, 1.1, 1.1, 1.1, 1.1))
            tipo = c[0].selectbox("Tipo", TIPOS_ATIVO, key=f"tipo_{portfolio_key}")
            desc = c[1].text_input("Descrição", key=f"desc_{portfolio_key}")
            indexador = c[2].selectbox("Indexador", INDEXADORES, key=f"idx_{portfolio_key}")
            # parâmetro de indexação:
            if indexador == "Pós CDI":
                par_idx = c[3].number_input("% do CDI (a.a.)", min_value=0.0, value=110.0, step=1.0, key=f"par_{portfolio_key}")
            elif indexador == "Prefixado":
                par_idx = c[3].number_input("Taxa Prefixada (% a.a.)", min_value=0.0, value=12.0, step=0.1, key=f"par_{portfolio_key}")
            else:  # IPCA+
                par_idx = c[3].number_input("Spread sobre IPCA (% a.a.)", min_value=0.0, value=5.0, step=0.1, key=f"par_{portfolio_key}")

            ir_opt = c[4].selectbox("IR", ["Isento", "15", "17.5", "20", "22.5", "Outro"], key=f"ir_{portfolio_key}")
            if ir_opt == "Outro":
                ir_pct = c[5].number_input("IR personalizado (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.5, key=f"irv_{portfolio_key}")
            else:
                ir_pct = 0.0 if ir_opt == "Isento" else float(ir_opt)

            r12 = c[6].number_input("Rent. 12M (%)", min_value=0.0, value=0.0, step=0.1, key=f"r12_{portfolio_key}")
            r6 = c[7].number_input("Rent. 6M (%)", min_value=0.0, value=0.0, step=0.1, key=f"r6_{portfolio_key}")
            aloc = c[8].number_input("Alocação (%)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, key=f"aloc_{portfolio_key}")

            sub = st.form_submit_button("Adicionar Ativo")
            if sub and desc:
                novo = pd.DataFrame([{
                    "Tipo": tipo, "Descrição": desc, "Indexador": indexador, "Parâmetro Indexação (% a.a.)": par_idx,
                    "IR (%)": ir_pct, "Isento": (ir_opt == "Isento"),
                    "Rent. 12M (%)": r12, "Rent. 6M (%)": r6, "Alocação (%)": aloc
                }])
                st.session_state[portfolio_key] = pd.concat([dfp, novo], ignore_index=True)
                st.rerun()

    dfp = st.session_state[portfolio_key]  # refresh
    if not dfp.empty:
        soma = dfp["Alocação (%)"].sum()
        dfp["Alocação Normalizada (%)"] = (dfp["Alocação (%)"]/soma*100.0).round(2)
        dfp["Valor (R$)"] = (valor_inicial * dfp["Alocação Normalizada (%)"]/100.0).round(2)

        c1, c2 = st.columns([1.6,1])
        with c1:
            if HAS_AGGRID:
                gob = GridOptionsBuilder.from_dataframe(dfp)
                gob.configure_selection('single', use_checkbox=True)
                gob.configure_grid_options(domLayout='autoHeight')
                grid = AgGrid(
                    dfp, gridOptions=gob.build(),
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    theme='streamlit', fit_columns_on_grid_load=True
                )
                sel = grid["selected_rows"]
                if sel:
                    st.info(f"Editar: **{sel[0].get('Descrição','')}**")
                    with st.form(f"edit_{portfolio_key}"):
                        nova_aloc = st.number_input("Nova Alocação (%)", min_value=0.1, max_value=100.0, step=0.1, value=float(sel[0]["Alocação (%)"]))
                        novo_ir = st.number_input("Novo IR (%) (0 p/ Isento)", min_value=0.0, max_value=100.0, step=0.5, value=float(sel[0]["IR (%)"]))
                        submitted = st.form_submit_button("Aplicar")
                        if submitted:
                            idx = int(sel[0]["_selectedRowNodeInfo"]["nodeRowIndex"])
                            st.session_state[portfolio_key].loc[idx, "Alocação (%)"] = nova_aloc
                            st.session_state[portfolio_key].loc[idx, "IR (%)"] = novo_ir
                            st.session_state[portfolio_key].loc[idx, "Isento"] = (novo_ir == 0.0)
                            st.rerun()
            else:
                st.dataframe(
                    dfp[["Tipo","Descrição","Indexador","Parâmetro Indexação (% a.a.)","IR (%)","Isento","Rent. 12M (%)","Alocação Normalizada (%)","Valor (R$)"]],
                    use_container_width=True, hide_index=True
                )
                # fallback para "clicar": seleção por descrição
                if len(dfp) > 0:
                    escolha = st.selectbox("Selecionar ativo para editar (fallback)", dfp["Descrição"].tolist())
                    if escolha:
                        idx = dfp.index[dfp["Descrição"]==escolha][0]
                        with st.form(f"edit_{portfolio_key}"):
                            nova_aloc = st.number_input("Nova Alocação (%)", min_value=0.1, max_value=100.0, step=0.1, value=float(dfp.loc[idx,"Alocação (%)"]))
                            novo_ir = st.number_input("Novo IR (%) (0 p/ Isento)", min_value=0.0, max_value=100.0, step=0.5, value=float(dfp.loc[idx,"IR (%)"]))
                            submitted = st.form_submit_button("Aplicar")
                            if submitted:
                                st.session_state[portfolio_key].loc[idx, "Alocação (%)"] = nova_aloc
                                st.session_state[portfolio_key].loc[idx, "IR (%)"] = novo_ir
                                st.session_state[portfolio_key].loc[idx, "Isento"] = (novo_ir == 0.0)
                                st.rerun()

        with c2:
            fig = criar_grafico_alocacao(dfp.rename(columns={"Tipo":"Classe","Descrição":"Descrição"}), f"Alocação — {titulo}")
            st.plotly_chart(fig, use_container_width=True)

        if soma > 100.1 or soma < 99.9:
            st.warning(f"A soma da alocação é {soma:.2f}%. Os valores foram normalizados para 100%.")

        colb = st.columns(2)
        with colb[0]:
            if st.button(f"Limpar {titulo}", key=f"clear_{portfolio_key}"):
                st.session_state[portfolio_key] = pd.DataFrame(columns=dfp.columns)
                st.rerun()
        with colb[1]:
            st.caption("Dica: instale `streamlit-aggrid` para clique direto na linha.")

    return st.session_state[portfolio_key]

# ---------- ABA 2 ----------
with tab2:
    df_atual = form_portfolio('portfolio_atual', "Portfólio Atual")

# ---------- ABA 3 ----------
with tab3:
    # aplica filtros de inclusão da sidebar: mantemos no personalizado apenas classes permitidas (quando marcadas)
    df_personalizado = st.session_state['portfolio_personalizado']
    df_personalizado = form_portfolio('portfolio_personalizado', "Portfólio Personalizado")

# ---------- FUNÇÕES DE TAXA A PARTIR DE INDEXADOR ----------
def taxa_aa_from_indexer(indexador: str, par_idx: float, cdi_aa: float, ipca_aa: float) -> float:
    if indexador == "Pós CDI":
        return (par_idx/100.0) * (cdi_aa/100.0)
    elif indexador == "Prefixado":
        return par_idx/100.0
    else:  # IPCA+
        return (ipca_aa/100.0) + (par_idx/100.0)

def taxa_portfolio_aa(df: pd.DataFrame, cdi_aa: float, ipca_aa: float, use_grossup: bool=False, ir_eq: float=15.0) -> float:
    if df.empty:
        return 0.0
    pesos = (df["Alocação Normalizada (%)"]/100.0).to_numpy()
    taxas = []
    for _, r in df.iterrows():
        taxa = taxa_aa_from_indexer(str(r["Indexador"]), float(r["Parâmetro Indexação (% a.a.)"]), cdi_aa, ipca_aa)
        # se isento e comparar em equivalente bruto
        if use_grossup and (bool(r["Isento"]) or float(r["IR (%)"]) == 0.0):
            taxa = gross_up(taxa, ir_eq)
        taxas.append(taxa)
    if len(taxas) == 0:
        return 0.0
    return float(np.average(taxas, weights=pesos))

# ---------- ABA 4 ----------
with tab4:
    st.subheader("Comparativos de Projeção (24 meses)")

    rent_atual_aa_net = taxa_portfolio_aa(df_atual, cdi_aa, ipca_aa, use_grossup=False, ir_eq=ir_equivalente)
    rent_pers_aa_net = taxa_portfolio_aa(df_personalizado, cdi_aa, ipca_aa, use_grossup=False, ir_eq=ir_equivalente)

    rent_atual_aa_gross = taxa_portfolio_aa(df_atual, cdi_aa, ipca_aa, use_grossup=True, ir_eq=ir_equivalente)
    rent_pers_aa_gross = taxa_portfolio_aa(df_personalizado, cdi_aa, ipca_aa, use_grossup=True, ir_eq=ir_equivalente)

    cenarios = {
        "Sugerido (net)": aa_to_am(rent_aa_sugerida),
        "Atual (net)": aa_to_am(rent_atual_aa_net),
        "Personalizado (net)": aa_to_am(rent_pers_aa_net),
        "Atual (equiv. bruto)": aa_to_am(rent_atual_aa_gross),
        "Personalizado (equiv. bruto)": aa_to_am(rent_pers_aa_gross),
    }

    df_comp = pd.DataFrame({'Mês': range(25)})
    for nome, taxa_m in cenarios.items():
        if taxa_m > -0.9999:  # evita absurdos
            df_comp[nome] = calcular_projecao(valor_inicial, aportes_mensais, taxa_m, 24)

    fig_comp = criar_grafico_projecao(df_comp, "Projeção Comparativa (Net vs Equivalente Bruta)")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.caption("Obs.: Para **produtos isentos**, a linha 'equiv. bruto' utiliza **Gross-up** com IR equivalente definido na barra lateral para permitir comparação justa com produtos tributáveis.")

    # Tabela de 12M
    st.subheader("Resumo 12 meses")
    df_resumo = pd.DataFrame({
        "Cenário": ["Carteira Sugerida (net)", "Portfólio Atual (net)", "Portfólio Personalizado (net)",
                    "Portfólio Atual (equiv. bruto)", "Portfólio Personalizado (equiv. bruto)"],
        "Rent. 12M (a.a.)": [
            f"{rent_aa_sugerida:.2%}", f"{rent_atual_aa_net:.2%}", f"{rent_pers_aa_net:.2%}",
            f"{rent_atual_aa_gross:.2%}", f"{rent_pers_aa_gross:.2%}"
        ],
        "Resultado final estimado (R$)": [
            f"R$ {(valor_inicial*(1+rent_aa_sugerida)) :,.2f}",
            f"R$ {(valor_inicial*(1+rent_atual_aa_net)) :,.2f}",
            f"R$ {(valor_inicial*(1+rent_pers_aa_net)) :,.2f}",
            f"R$ {(valor_inicial*(1+rent_atual_aa_gross)) :,.2f}",
            f"R$ {(valor_inicial*(1+rent_pers_aa_gross)) :,.2f}",
        ]
    })
    st.dataframe(df_resumo, hide_index=True, use_container_width=True)

# ---------- ABA 5: RELATÓRIO ----------
def build_html_report(nome: str, perfil: str, prazo_meses: int, valor_inicial: float, aportes: float,
                      meta: float, df_sug: pd.DataFrame) -> str:
    tabela = df_sug[["Classe de Ativo","Alocação (%)","Valor (R$)"]].to_html(index=False, border=0)
    style = """
    <style>
        body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; color:#222; }
        .card { border:1px solid #e5e5e5; border-radius:12px; padding:16px; margin:12px 0; }
        h1,h2,h3 { margin: 0.2rem 0 0.6rem; }
        table { width:100%; border-collapse: collapse; }
        th, td { padding: 8px 10px; border-bottom: 1px solid #eee; text-align:left; }
        .muted { color:#666; font-size: 0.9rem; }
        .tag { background:#f3f4f6; border:1px solid #e5e7eb; padding:3px 8px; border-radius:999px; font-size:0.85rem; }
    </style>
    """
    html = f"""
    <!doctype html><html><head><meta charset="utf-8">{style}</head><body>
    <h1>Relatório — Análise de Portfólio</h1>
    <div class="muted">Cliente: <span class="tag">{nome}</span> • Perfil: <span class="tag">{perfil}</span> • Prazo: <span class="tag">{prazo_meses} meses</span></div>
    <div class="card">
        <h2>Dados Iniciais</h2>
        <ul>
            <li><b>Valor Inicial:</b> R$ {valor_inicial:,.2f}</li>
            <li><b>Aportes Mensais:</b> R$ {aportes:,.2f}</li>
            <li><b>Meta Financeira:</b> R$ {meta:,.2f}</li>
        </ul>
    </div>
    <div class="card">
        <h2>Carteira Sugerida</h2>
        {tabela}
    </div>
    <div class="card">
        <h3>Avisos Importantes</h3>
        <p class="muted">Os resultados simulados são ilustrativos, não configuram garantia de rentabilidade futura. Para comparação entre produtos isentos e tributáveis, utilizamos o conceito de <i>Gross-up</i> aplicando uma alíquota equivalente (definida no app). Leia os documentos dos produtos antes de investir.</p>
    </div>
    </body></html>
    """
    return html

with tab5:
    st.subheader("Relatório para Envio")

    html_report = build_html_report(
        nome_cliente, perfil_investimento, prazo_meses, valor_inicial, aportes_mensais, meta_financeira, df_sugerido
    )

    st.markdown("**Prévia HTML:**", help="Formato mais profissional para enviar/printar")
    st.components.v1.html(html_report, height=540, scrolling=True)

    # Exportações
    st.download_button("⬇️ Baixar Relatório (HTML)", data=html_report.encode("utf-8"),
                       file_name=f"relatorio_{nome_cliente.lower().replace(' ', '_')}.html", mime="text/html")

    # TXT (legado)
    relatorio_txt = f"""Relatório — {nome_cliente}
Perfil: {perfil_investimento} | Prazo: {prazo_meses} meses
Valor Inicial: R$ {valor_inicial:,.2f} | Aportes: R$ {aportes_mensais:,.2f} | Meta: R$ {meta_financeira:,.2f}

Carteira Sugerida:
{df_sugerido[["Classe de Ativo","Alocação (%)","Valor (R$)"]].to_string(index=False)}

(Observações: simulações ilustrativas; Gross-up aplicado para comparação entre isentos e tributáveis.)
"""
    st.download_button("⬇️ Baixar Relatório (TXT)", data=relatorio_txt, file_name=f"relatorio_{nome_cliente.lower().replace(' ', '_')}.txt", mime="text/plain")

    # PDF (se possível)
    if HAS_WEASYPRINT:
        pdf_bytes_buf = io.BytesIO()
        HTML(string=html_report).write_pdf(target=pdf_bytes_buf)
        st.download_button("⬇️ Baixar Relatório (PDF)", data=pdf_bytes_buf.getvalue(),
                           file_name=f"relatorio_{nome_cliente.lower().replace(' ', '_')}.pdf",
                           mime="application/pdf")
    else:
        st.info("Para exportar PDF automaticamente, instale o WeasyPrint (e dependências do Cairo). Enquanto isso, use o HTML para imprimir em PDF.")

# ---------- RODAPÉ ----------
st.markdown("---")
with st.expander("Avisos Importantes", expanded=False):
    st.warning("""
Os resultados simulados são meramente ilustrativos, não configurando garantia de rentabilidade futura ou promessa de retorno para os produtos sugeridos.
Para comparação entre produtos isentos e tributáveis, aplicamos o conceito de Gross-up com alíquota equivalente definida na barra lateral.
Fundos de investimento não contam com garantia do FGC. Leia os documentos dos produtos antes de investir.
    """)
