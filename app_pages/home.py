import streamlit as st
import pandas as pd
import plotly.express as px


def home():
    st.title("💼 Dashboard de Vagas e Candidatos")

    vagas = pd.read_json("data/vagas.json", orient='records')
    prospects = pd.read_json("data/prospects.json", orient='records')

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
    total_candidatos = df_vagas['candidatos'].sum()

    col1, col2 = st.columns(2)
    col1.metric("Total de Vagas", total_vagas)
    col2.metric("Total de Candidatos", total_candidatos)

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
    fig1.update_layout(
        template='plotly_dark',
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
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
    fig2.update_layout(
        template='plotly_dark',
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
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
    fig3.update_layout(
        template='plotly_dark',
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("📄 Dados das Vagas")
    st.dataframe(df_vagas[['vaga_id', 'cliente', 'recrutador', 'idioma', 'candidatos']], use_container_width=True)
