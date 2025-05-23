import streamlit as st
import json
import os
from datetime import datetime


# 📂 Caminho do JSON
PASTA_DADOS = r"C:\Users\giuliasilva\Desktop\Estudo\POS\TC - Modulo 05\application_web\data"
ARQUIVO_JSON = os.path.join(PASTA_DADOS, "applicants.json")

# 🔧 Garantir que a pasta existe
os.makedirs(PASTA_DADOS, exist_ok=True)

# 🧠 Função para carregar os dados existentes
def carregar_dados():
    if os.path.exists(ARQUIVO_JSON):
        with open(ARQUIVO_JSON, "r", encoding="utf-8") as arquivo:
            try:
                return json.load(arquivo)
            except json.JSONDecodeError:
                return []  # Se estiver vazio ou corrompido
    return []

# 💾 Função para salvar dados
def salvar_dados(dados):
    with open(ARQUIVO_JSON, "w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, indent=4, ensure_ascii=False)

# 🔢 Função para gerar próximo ID
def gerar_proximo_id(dados):
    codigos = [
        int(item.get("infos_basicas_codigo_profissional", 0))
        for item in dados if str(item.get("infos_basicas_codigo_profissional", "")).isdigit()
    ]
    if not codigos:
        return "10000"
    return str(max(codigos) + 1)

# 📝 Função principal de cadastro
def cadastro_candidatos():
    st.title("📋 Cadastro de Candidatos")

    dados = carregar_dados()

    with st.form("formulario_cadastro"):
        st.subheader("Informações Básicas")
        nome = st.text_input("Nome completo*", max_chars=100)
        email = st.text_input("Email*", max_chars=100)
        telefone = st.text_input("Telefone")
        celular = st.text_input("Telefone celular")
        objetivo = st.text_area("Objetivo profissional")
        area_atuacao = st.text_input("Área de atuação")
        remuneracao = st.text_input("Remuneração desejada")

        sexo = st.selectbox("Sexo", ["", "Masculino", "Feminino", "Outro"])
        estado_civil = st.selectbox("Estado civil", ["", "Solteiro", "Casado", "Divorciado", "Viúvo"])

        cpf = st.text_input("CPF")
        data_nascimento = st.date_input("Data de nascimento")

        local = st.text_input("Local (Cidade, Estado)")

        cv_pt = st.text_area("Resumo/CV em Português *")

        submitted = st.form_submit_button("Salvar Candidato")

        if submitted:
            if not nome.strip() or not email.strip() or not cv_pt.strip():
                st.error("⚠️ Preencha todos os campos obrigatórios (*)")
                return

            codigo = gerar_proximo_id(dados)
            agora = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

            novo_candidato = {
                "infos_basicas_telefone_recado": "",
                "infos_basicas_telefone": telefone,
                "infos_basicas_objetivo_profissional": objetivo,
                "infos_basicas_data_criacao": agora,
                "infos_basicas_inserido_por": "Sistema Streamlit",
                "infos_basicas_email": email,
                "infos_basicas_local": local,
                "infos_basicas_sabendo_de_nos_por": "",
                "infos_basicas_data_atualizacao": agora,
                "infos_basicas_codigo_profissional": codigo,
                "infos_basicas_nome": nome,

                "informacoes_pessoais_data_aceite": "Cadastro via Streamlit",
                "informacoes_pessoais_nome": nome,
                "informacoes_pessoais_cpf": cpf,
                "informacoes_pessoais_fonte_indicacao": "",
                "informacoes_pessoais_email": email,
                "informacoes_pessoais_email_secundario": "",
                "informacoes_pessoais_data_nascimento": data_nascimento.strftime("%d-%m-%Y"),
                "informacoes_pessoais_telefone_celular": celular,
                "informacoes_pessoais_telefone_recado": "",
                "informacoes_pessoais_sexo": sexo,
                "informacoes_pessoais_estado_civil": estado_civil,
                "informacoes_pessoais_pcd": "Não",
                "informacoes_pessoais_endereco": local,
                "informacoes_pessoais_skype": "",
                "informacoes_pessoais_url_linkedin": "",
                "informacoes_pessoais_facebook": "",
                "informacoes_pessoais_download_cv": None,

                "informacoes_profissionais_titulo_profissional": "",
                "informacoes_profissionais_area_atuacao": area_atuacao,
                "informacoes_profissionais_conhecimentos_tecnicos": "",
                "informacoes_profissionais_certificacoes": "",
                "informacoes_profissionais_outras_certificacoes": "",
                "informacoes_profissionais_remuneracao": remuneracao,
                "informacoes_profissionais_nivel_profissional": "",
                "informacoes_profissionais_qualificacoes": None,
                "informacoes_profissionais_experiencias": None,

                "formacao_e_idiomas_nivel_academico": "",
                "formacao_e_idiomas_nivel_ingles": "",
                "formacao_e_idiomas_nivel_espanhol": "",
                "formacao_e_idiomas_outro_idioma": "",
                "formacao_e_idiomas_instituicao_ensino_superior": "",
                "formacao_e_idiomas_cursos": "",
                "formacao_e_idiomas_ano_conclusao": "",
                "formacao_e_idiomas_outro_curso": None,

                "cargo_atual_id_ibrati": None,
                "cargo_atual_email_corporativo": None,
                "cargo_atual_cargo_atual": None,
                "cargo_atual_projeto_atual": None,
                "cargo_atual_cliente": None,
                "cargo_atual_unidade": None,
                "cargo_atual_data_admissao": None,
                "cargo_atual_data_ultima_promocao": None,
                "cargo_atual_nome_superior_imediato": None,
                "cargo_atual_email_superior_imediato": None,

                "cv_pt": cv_pt,
                "cv_en": ""
            }

            dados.append(novo_candidato)
            salvar_dados(dados)

            st.success(f"✅ Candidato cadastrado com sucesso! Código: {codigo}")

if __name__ == "__main__":
    cadastro_candidatos()
