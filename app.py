import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Configurações da Página ---
st.set_page_config(
    page_title="Gerador de Portfólios de Investimento",
    page_icon="💹",
    layout="wide"
)

# --- DADOS MOCK (Simulando as carteiras sugeridas do PDF) ---
# Em um caso real, você poderia carregar isso de um arquivo CSV ou de uma base de dados.
CARTEIRAS_SUGERIDAS = {
    "Conservador": {
        "rentabilidade_esperada_aa": 0.08, # 8% ao ano
        "alocacao": {
            "Renda Fixa Pós-Fixada": 0.70,
            "Renda Fixa Inflação": 0.20,
            "Crédito Privado": 0.10,
        }
    },
    "Moderado": {
        "rentabilidade_esperada_aa": 0.10, # 10% ao ano
        "alocacao": {
            "Renda Fixa Pós-Fixada": 0.40,
            "Renda Fixa Inflação": 0.25,
            "Crédito Privado": 0.15,
            "Fundos Imobiliários": 0.10,
            "Ações e Fundos de Índice": 0.10,
        }
    },
    "Arrojado": {
        "rentabilidade_esperada_aa": 0.12, # 12% ao ano
        "alocacao": {
            "Renda Fixa Pós-Fixada": 0.20,
            "Renda Fixa Inflação": 0.10,
            "Crédito Privado": 0.20,
            "Fundos Imobiliários": 0.20,
            "Ações e Fundos de Índice": 0.30,
        }
    }
}


# --- Funções Auxiliares ---

def calcular_projecao(valor_inicial, aportes_mensais, taxa_mensal, prazo_meses):
    """Calcula a projeção de crescimento de um investimento."""
    projecao = [valor_inicial]
    for _ in range(prazo_meses):
        novo_valor = (projecao[-1] + aportes_mensais) * (1 + taxa_mensal)
        projecao.append(novo_valor)
    return projecao

def criar_grafico_projecao(df_projecao):
    """Cria um gráfico de linha com a projeção dos portfólios."""
    fig = px.line(df_projecao, x='Mês', y=df_projecao.columns[1:],
                    title="Projeção de Crescimento do Patrimônio",
                    labels={'value': 'Patrimônio (R$)', 'variable': 'Portfólio'},
                    markers=True)
    fig.update_layout(legend_title_text='Cenários', yaxis_title='Patrimônio (R$)', xaxis_title='Meses')
    return fig

def criar_grafico_alocacao(df_portfolio, title):
    """Cria um gráfico de pizza com a alocação de ativos."""
    if df_portfolio.empty or df_portfolio['Valor (R$)'].sum() == 0:
        return go.Figure()

    fig = px.pie(df_portfolio, values='Valor (R$)', names='Descrição', title=title,
                 hole=.3, color_discrete_sequence=px.colors.sequential.PuBuGn)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

# --- Inicialização do Session State ---
# Usado para armazenar os dados dos portfólios enquanto o usuário interage com o app.
if 'portfolio_atual' not in st.session_state:
    st.session_state.portfolio_atual = pd.DataFrame(columns=["Tipo", "Descrição", "Rent. 12M (%)", "Rent. 6M (%)", "Alocação (%)"])
if 'portfolio_personalizado' not in st.session_state:
    st.session_state.portfolio_personalizado = pd.DataFrame(columns=["Tipo", "Descrição", "Rent. 12M (%)", "Rent. 6M (%)", "Alocação (%)"])


# --- BARRA LATERAL (INPUTS DO USUÁRIO) ---
with st.sidebar:
    st.image("https://img.icons8.com/plasticine/100/000000/stocks-growth.png", width=100)
    st.title("Parâmetros do Cliente")
    st.markdown("---")

    nome_cliente = st.text_input("Nome do Cliente", "Cliente Exemplo")
    perfil_investimento = st.selectbox("Perfil de Investimento", list(CARTEIRAS_SUGERIDAS.keys()))

    st.subheader("Opções da Carteira Sugerida")
    incluir_credito_privado = st.checkbox("Incluir Crédito Privado", True)
    incluir_previdencia = st.checkbox("Incluir Previdência", False) # Adicionado
    incluir_fundos_imobiliarios = st.checkbox("Incluir Fundos Imobiliários", True)
    incluir_acoes_indice = st.checkbox("Incluir Ações e Fundos de Índice", True)
    st.markdown("---")

    st.subheader("Projeção Financeira")
    valor_inicial = st.number_input("Valor Inicial do Investimento (R$)", min_value=1000.0, value=50000.0, step=1000.0)
    aportes_mensais = st.number_input("Aportes Mensais (R$)", min_value=0.0, value=1000.0, step=100.0)
    prazo_anos = st.slider("Prazo de Permanência (anos)", 1, 30, 10)
    meta_financeira = st.number_input("Meta a Atingir (R$)", min_value=0.0, value=500000.0, step=10000.0)

prazo_meses = prazo_anos * 12

# --- PÁGINA PRINCIPAL ---
st.title(f"💹 Análise de Portfólio para {nome_cliente}")
st.markdown(f"**Perfil de Investimento:** `{perfil_investimento}` | **Prazo Objetivo:** `{prazo_anos} anos`")
st.markdown("---")


# --- Lógica para Carteira Sugerida ---
carteira_base = CARTEIRAS_SUGERIDAS[perfil_investimento]
alocacao_sugerida = carteira_base["alocacao"].copy()

# Filtra a carteira sugerida com base nas opções do usuário
if not incluir_credito_privado:
    alocacao_sugerida.pop("Crédito Privado", None)
if not incluir_fundos_imobiliarios:
    alocacao_sugerida.pop("Fundos Imobiliários", None)
if not incluir_acoes_indice:
    alocacao_sugerida.pop("Ações e Fundos de Índice", None)
if incluir_previdencia: # Se incluir previdência, adiciona uma linha
     alocacao_sugerida["Previdência Privada"] = 0.10 # Exemplo de alocação

# Recalcula os pesos para somarem 100%
total_peso = sum(alocacao_sugerida.values())
alocacao_sugerida = {k: v / total_peso for k, v in alocacao_sugerida.items()}

df_sugerido = pd.DataFrame(list(alocacao_sugerida.items()), columns=['Classe de Ativo', 'Alocação (%)'])
df_sugerido['Alocação (%)'] = (df_sugerido['Alocação (%)'] * 100).round(2)
df_sugerido['Valor (R$)'] = (valor_inicial * df_sugerido['Alocação (%)'] / 100).round(2)
rentabilidade_sugerida_aa = carteira_base['rentabilidade_esperada_aa']
rentabilidade_sugerida_am = (1 + rentabilidade_sugerida_aa)**(1/12) - 1

# --- ABAS DE NAVEGAÇÃO ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Projeção e Carteira Sugerida",
    "💼 Portfólio Atual",
    "🎨 Personalizar Carteira",
    "📊 Comparativo",
    "📋 Gerar Relatório"
])


# --- ABA 1: Projeção e Carteira Sugerida ---
with tab1:
    st.header("Projeção de Investimentos e Carteira Sugerida")
    
    # Gráfico de Projeção
    projecao_sugerida = calcular_projecao(valor_inicial, aportes_mensais, rentabilidade_sugerida_am, prazo_meses)
    df_projecao = pd.DataFrame({
        'Mês': range(prazo_meses + 1),
        'Carteira Sugerida': projecao_sugerida
    })
    
    fig_projecao_sugerida = criar_grafico_projecao(df_projecao)
    
    # Adiciona linha da meta
    fig_projecao_sugerida.add_hline(y=meta_financeira, line_dash="dash", line_color="red", annotation_text="Meta Financeira", annotation_position="top left")
    st.plotly_chart(fig_projecao_sugerida, use_container_width=True)

    # Detalhes da Carteira Sugerida
    st.subheader(f"Alocação Sugerida - Perfil {perfil_investimento}")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(df_sugerido, use_container_width=True, hide_index=True)
    with col2:
        fig_aloc_sugerida = criar_grafico_alocacao(df_sugerido.rename(columns={'Classe de Ativo': 'Descrição'}), "Alocação da Carteira Sugerida")
        st.plotly_chart(fig_aloc_sugerida, use_container_width=True)


# --- Funções para gerenciar portfólios manuais ---
def gerenciar_portfolio(portfolio_key, titulo):
    st.header(titulo)
    df_portfolio = st.session_state[portfolio_key]

    with st.expander("Adicionar/Remover Ativos", expanded=not df_portfolio.empty):
        # Formulário para adicionar novo ativo
        with st.form(f"form_{portfolio_key}", clear_on_submit=True):
            cols = st.columns((2, 3, 1, 1, 1))
            tipo = cols[0].selectbox("Tipo", ["Renda Fixa", "Ação", "Fundo de Investimento", "FII", "Outro"], key=f"tipo_{portfolio_key}")
            descricao = cols[1].text_input("Descrição do Ativo", key=f"desc_{portfolio_key}")
            rent_12m = cols[2].number_input("Rent. 12M (%)", format="%.2f", key=f"r12_{portfolio_key}")
            rent_6m = cols[3].number_input("Rent. 6M (%)", format="%.2f", key=f"r6_{portfolio_key}")
            alocacao = cols[4].number_input("Alocação (%)", min_value=0.1, max_value=100.0, format="%.2f", key=f"aloc_{portfolio_key}")
            
            submitted = st.form_submit_button("Adicionar Ativo")
            if submitted and descricao:
                nova_linha = pd.DataFrame([{
                    "Tipo": tipo, "Descrição": descricao, "Rent. 12M (%)": rent_12m,
                    "Rent. 6M (%)": rent_6m, "Alocação (%)": alocacao
                }])
                st.session_state[portfolio_key] = pd.concat([df_portfolio, nova_linha], ignore_index=True)
                st.rerun()

    if not df_portfolio.empty:
        # Normaliza a alocação para que a soma seja 100%
        soma_alocacao = df_portfolio['Alocação (%)'].sum()
        df_portfolio['Alocação Normalizada (%)'] = (df_portfolio['Alocação (%)'] / soma_alocacao * 100).round(2)
        df_portfolio['Valor (R$)'] = (valor_inicial * df_portfolio['Alocação Normalizada (%)'] / 100).round(2)
        
        st.subheader("Composição do Portfólio")
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.dataframe(df_portfolio[["Tipo", "Descrição", "Rent. 12M (%)", "Alocação Normalizada (%)", "Valor (R$)"]].rename(columns={"Alocação Normalizada (%)": "Alocação (%)"}),
                         use_container_width=True, hide_index=True)
            if soma_alocacao > 100.1 or soma_alocacao < 99.9:
                st.warning(f"A soma da alocação é {soma_alocacao:.2f}%. Os valores foram normalizados para 100%.")

        with col2:
            fig = criar_grafico_alocacao(df_portfolio, f"Alocação do {titulo}")
            st.plotly_chart(fig, use_container_width=True)

        # Botão para limpar a tabela
        if st.button(f"Limpar {titulo}", key=f"clear_{portfolio_key}"):
            st.session_state[portfolio_key] = pd.DataFrame(columns=df_portfolio.columns[:-2]) # Mantém colunas originais
            st.rerun()
            
    return df_portfolio


# --- ABA 2: Portfólio Atual ---
with tab2:
    df_atual = gerenciar_portfolio('portfolio_atual', "Portfólio Atual")


# --- ABA 3: Personalizar Carteira ---
with tab3:
    df_personalizado = gerenciar_portfolio('portfolio_personalizado', "Portfólio Personalizado")


# --- ABA 4: Comparativo ---
with tab4:
    st.header("Comparativo de Desempenho")

    # Calcula rentabilidades médias dos portfólios manuais
    rent_atual_aa = 0.0
    if not df_atual.empty:
        pesos_atuais = df_atual['Alocação Normalizada (%)'] / 100
        rentabilidades_atuais = df_atual['Rent. 12M (%)'] / 100
        rent_atual_aa = np.average(rentabilidades_atuais, weights=pesos_atuais)

    rent_personalizado_aa = 0.0
    if not df_personalizado.empty:
        pesos_personalizados = df_personalizado['Alocação Normalizada (%)'] / 100
        rentabilidades_personalizadas = df_personalizado['Rent. 12M (%)'] / 100
        rent_personalizado_aa = np.average(rentabilidades_personalizadas, weights=pesos_personalizados)

    rent_atual_am = (1 + rent_atual_aa)**(1/12) - 1
    rent_personalizado_am = (1 + rent_personalizado_aa)**(1/12) - 1
    
    # Gráfico de Projeção Comparativa para 24 meses
    st.subheader("Projeção Comparativa (24 meses)")
    df_projecao_comp = pd.DataFrame({'Mês': range(25)})
    
    cenarios = {
        'Sugerido': rentabilidade_sugerida_am,
        'Atual': rent_atual_am,
        'Personalizado': rent_personalizado_am
    }

    for nome, taxa in cenarios.items():
        if taxa > 0: # Apenas projeta se houver rentabilidade
            df_projecao_comp[nome] = calcular_projecao(valor_inicial, aportes_mensais, taxa, 24)

    if len(df_projecao_comp.columns) > 1:
        fig_comp = criar_grafico_projecao(df_projecao_comp)
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Adicione ativos nos portfólios para ver a comparação.")

    # Simulação Retrospectiva (12 meses)
    st.subheader("Simulação Retrospectiva (Resultado em 12 meses)")
    valor_final_sugerido = valor_inicial * (1 + rentabilidade_sugerida_aa)
    valor_final_atual = valor_inicial * (1 + rent_atual_aa) if rent_atual_aa > 0 else 0
    valor_final_personalizado = valor_inicial * (1 + rent_personalizado_aa) if rent_personalizado_aa > 0 else 0
    
    retrospectiva_data = {
        'Cenário': ['Carteira Sugerida', 'Portfólio Atual', 'Portfólio Personalizado'],
        'Rentabilidade (12M)': [f"{rentabilidade_sugerida_aa:.2%}", f"{rent_atual_aa:.2%}", f"{rent_personalizado_aa:.2%}"],
        'Resultado Final (R$)': [f"R$ {valor_final_sugerido:,.2f}", f"R$ {valor_final_atual:,.2f}", f"R$ {valor_final_personalizado:,.2f}"],
    }
    df_retro = pd.DataFrame(retrospectiva_data)
    st.table(df_retro)


# --- ABA 5: Gerar Relatório ---
with tab5:
    st.header("Relatório para Envio")
    st.info("Copie o texto abaixo ou faça o download do arquivo para enviar ao cliente.")
    
    # Monta o texto do relatório
    relatorio_texto = f"""
# Análise de Portfólio para {nome_cliente}

Olá {nome_cliente.split(' ')[0]},

Com base nas informações fornecidas, preparamos uma análise e sugestão de portfólio de investimentos alinhada ao seu perfil **{perfil_investimento}**.

## Dados Iniciais
- **Valor Inicial:** R$ {valor_inicial:,.2f}
- **Aportes Mensais:** R$ {aportes_mensais:,.2f}
- **Prazo Objetivo:** {prazo_anos} anos
- **Meta Financeira:** R$ {meta_financeira:,.2f}

## Carteira Sugerida
A alocação sugerida para o seu perfil é:
{df_sugerido[['Classe de Ativo', 'Alocação (%)']].to_markdown(index=False)}

---
*Avisos importantes: Os resultados simulados são meramente ilustrativos, não configurando garantia de rentabilidade futura ou promessa de retorno para os produtos sugeridos. Os números reportados refletem simulações com parâmetro de intervalo de confiança de 95%, representando a análise de cenário e expectativas em relação ao índice livre de risco (CDI). Os resultados simulados para os produtos LCAs e LCIs consideram a incidência de IR à alíquota de 15% para equalização com a rentabilidade dos demais ativos. Em regra, quanto maior a expectativa de retorno (rentabilidade) do investimento, maior será o risco da aplicação, ou seja, há a possibilidade da aplicação não valorizar o esperado e, em alguns casos, até de perda de parte do principal investido (a quantia aplicada). Fundos de investimento são uma modalidade de investimento que não conta com a garantia do Fundo Garantidor de Crédito (FGC). Leia os documentos dos produtos ofertados antes de investir.*
"""

    st.text_area("Relatório:", relatorio_texto, height=400)
    
    st.download_button(
        label="📥 Baixar Relatório (.txt)",
        data=relatorio_texto,
        file_name=f"relatorio_{nome_cliente.lower().replace(' ', '_')}.txt",
        mime="text/plain"
    )

# --- Rodapé com Avisos ---
st.markdown("---")
with st.expander("Avisos Importantes", expanded=False):
    st.warning("""
    Os resultados simulados são meramente ilustrativos, não configurando garantia de rentabilidade futura ou promessa de retorno para os produtos sugeridos. 
    Os números reportados refletem simulações com parâmetro de intervalo de confiança de 95%, representando a análise de cenário e expectativas em relação ao índice livre de risco (CDI). 
    Os resultados simulados para os produtos LCAs e LCIs consideram a incidência de IR à alíquota de 15% para equalização com a rentabilidade dos demais ativos. 
    Em regra, quanto maior a expectativa de retorno (rentabilidade) do investimento, maior será o risco da aplicação, ou seja, há a possibilidade da aplicação não valorizar o esperado e, em alguns casos, até de perda de parte do principal investido (a quantia aplicada). 
    Fundos de investimento são uma modalidade de investimento que não conta com a garantia do Fundo Garantidor de Crédito (FGC). 
    Leia os documentos dos produtos ofertados antes de investir.
    """)
