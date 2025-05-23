import streamlit as st # Importe st primeiro!

# CHAME set_page_config() AQUI, LOGO APÓS A IMPORTAÇÃO DE 'streamlit'
st.set_page_config(
    page_title="Sistema de Recomendação de Talentos",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


import sys
import os
from app_pages.home import home
from app_pages.cadastro_vagas import cadastro_vagas
from app_pages.cadastro_candidatos import cadastro_candidatos
from app_pages import servicos

st.write(f"Python version: {sys.version}")


# Inicializa o estado da sessão para navegação
if "pagina" not in st.session_state:
    st.session_state["pagina"] = "Home"  

# Criando os menus laterais clicáveis
with st.sidebar:
    st.markdown("## **Menu**", unsafe_allow_html=True)
    if st.button("🏠 Home", use_container_width=True, key="home_btn"):
        st.session_state["pagina"] = "Home"
    if st.button("📝 Cadastro de Vagas", use_container_width=True, key="cadastro_vagas_btn"):
        st.session_state["pagina"] = "Cadastro de Vagas"
    if st.button("👥 Cadastro de Candidatos", use_container_width=True, key="cadastro_candidatos_btn"):
        st.session_state["pagina"] = "Cadastro de Candidatos"
    if st.button("🛠️ Serviços", use_container_width=True, key="servicos_btn"):
        st.session_state["pagina"] = "Serviços"


# Renderizando a página ativa
if st.session_state["pagina"] == "Home":
    home()
elif st.session_state["pagina"] == "Cadastro de Vagas":
    cadastro_vagas()
elif st.session_state["pagina"] == "Cadastro de Candidatos":
    cadastro_candidatos()
elif st.session_state["pagina"] == "Serviços":
    servicos.pagina_servicos()
