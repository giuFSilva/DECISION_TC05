import streamlit as st
import pandas as pd
import plotly.express as px
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

@st.cache_data
def load_data():
    try:
        # Certifique-se de que 'applicants.json' também está na pasta 'data/'
        vagas_path = os.path.join(DATA_DIR, "vagas.json")
        prospects_path = os.path.join(DATA_DIR, "prospects.json")
        applicants_path = os.path.join(DATA_DIR, "applicants.json") # Novo path para applicants.json

        # Verificação adicional para garantir que os arquivos existem antes de tentar ler
        if not os.path.exists(vagas_path):
            st.error(f"❌ Erro: Arquivo 'vagas.json' não encontrado em {vagas_path}")
            st.stop()
        if not os.path.exists(prospects_path):
            st.error(f"❌ Erro: Arquivo 'prospects.json' não encontrado em {prospects_path}")
            st.stop()
        if not os.path.exists(applicants_path):
            st.error(f"❌ Erro: Arquivo 'applicants.json' não encontrado em {applicants_path}")
            st.stop()

        # Carrega os DataFrames
        df_vagas = pd.read_json(vagas_path, orient='records')
        df_prospects = pd.read_json(prospects_path, orient='records')
        df_applicants = pd.read_json(applicants_path, orient='records') # Carrega applicants.json

        return df_vagas, df_prospects, df_applicants
    except FileNotFoundError as e:
        st.error(f"❌ Erro ao carregar arquivos JSON: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Ocorreu um erro inesperado ao carregar os dados: {e}")
        st.stop()

# Chama a função para carregar os dados
vagas, prospects, applicants = load_data()


# --- DASHBOARD ---
def home():
    st.title("💼 Dashboard de Vagas e Candidatos")

    # Mantenha o restante do seu código de dashboard inalterado aqui
    # ... todo o código de transformação e visualização do dashboard ...

    df_vagas = vagas.rename(columns={
        "id_vaga": "vaga_id",
        "info_cliente": "cliente",
        "info_analista_responsavel": "recrutador",
        "perfil_nivel_ingles": "nivel_ingles",
        "perfil_nivel_espanhol": "nivel_espanhol"
    })

    df_vagas['cliente'] = df_vagas['cliente'].fillna('Não Informado')
    df_vagas['recrutador'] = df_vagas['recrutador'].fillna('Não Informado')

    def construir_idioma(row):
        idiomas = []
        if pd.notna(row.get('nivel_ingles')) and row['nivel_ingles'].strip() != '':
            idiomas.append('Inglês')
        if pd.notna(row.get('nivel_espanhol')) and row['nivel_espanhol'].strip() != '':
            idiomas.append('Espanhol')
        return ', '.join(idiomas) if idiomas else ''

    df_vagas['idioma'] = df_vagas.apply(construir_idioma, axis=1)

    candidatos_por_vaga = prospects.groupby('id_vaga').size().reset_index(name='candidatos')
    df_vagas = df_vagas.merge(candidatos_por_vaga, how='left', left_on='vaga_id', right_on='id_vaga')
    df_vagas['candidatos'] = df_vagas['candidatos'].fillna(0).astype(int)

    total_vagas = df_vagas['vaga_id'].nunique()
    # Lembre-se que 'applicants' (candidatos_originais) agora está disponível
    # se você quiser usar os dados de 'applicants' para o total de candidatos,
    # em vez de somar os prospects por vaga.
    total_candidatos_reais = applicants['infos_basicas_codigo_profissional'].nunique() # Se cada linha é um candidato único

    col1, col2 = st.columns(2)
    col1.metric("Total de Vagas", total_vagas)
    col2.metric("Total de Candidatos", total_candidatos_reais) # Usando o df 'applicants'

    st.markdown("---")

    st.subheader("👩‍💼 Vagas por Recrutador")
    vagas_por_recrutador = df_vagas['recrutador'].value_counts().reset_index()
    vagas_por_recrutador.columns = ['Recrutador', 'Quantidade']

    fig1 = px.bar(
        vagas_por_recrutador,
        x='Quantidade',
        y='Recrutador',
        orientation='h',
        color_discrete_sequence=['#1f77b4'],
        title='Vagas por Recrutador'
    )
    fig1.update_layout(template='plotly_dark', height=300, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🏢 Vagas por Cliente")
    vagas_por_cliente = df_vagas['cliente'].value_counts().reset_index()
    vagas_por_cliente.columns = ['Cliente', 'Quantidade']

    fig2 = px.bar(
        vagas_por_cliente,
        x='Quantidade',
        y='Cliente',
        orientation='h',
        color_discrete_sequence=['#3c8dbc'],
        title='Vagas por Cliente'
    )
    fig2.update_layout(template='plotly_dark', height=300, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🌍 Vagas que Exigem Idioma")

    def contar_idiomas(col):
        idiomas_contagem = {'Inglês': 0, 'Espanhol': 0, 'Nenhum': 0}
        for item in col:
            item_lower = str(item).lower()
            if 'inglês' in item_lower and 'espanhol' in item_lower:
                idiomas_contagem['Inglês'] += 1
                idiomas_contagem['Espanhol'] += 1
            elif 'inglês' in item_lower:
                idiomas_contagem['Inglês'] += 1
            elif 'espanhol' in item_lower:
                idiomas_contagem['Espanhol'] += 1
            else:
                idiomas_contagem['Nenhum'] += 1
        return idiomas_contagem

    contagem_idiomas = contar_idiomas(df_vagas['idioma'])
    idiomas_df = pd.DataFrame({
        'Idioma': list(contagem_idiomas.keys()),
        'Quantidade': list(contagem_idiomas.values())
    })

    fig3 = px.pie(
        idiomas_df,
        names='Idioma',
        values='Quantidade',
        color_discrete_sequence=px.colors.sequential.Blues,
        title='Distribuição de Vagas por Idioma'
    )
    fig3.update_traces(textfont_size=12, textinfo='percent+label')
    fig3.update_layout(template='plotly_dark', height=350)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("📄 Dados das Vagas")
    st.dataframe(df_vagas[['vaga_id', 'cliente', 'recrutador', 'idioma', 'candidatos']], use_container_width=True)
