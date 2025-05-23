import streamlit as st
import json
import os
from datetime import datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd

# --- CONFIGURAÇÕES E CAMINHOS ---
# Diretório base onde o arquivo .py está rodando (app_pages/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pastas de dados e modelos (um nível acima de app_pages/)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
# ATENÇÃO: Verifique se sua pasta no repositório é 'models' ou 'models1'
# Mantenha a consistência com servicos.py. Vou usar 'models1' como exemplo.
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models1') # AJUSTADO: Consistência

# Caminhos dos arquivos
ARQUIVO_CANDIDATOS = os.path.join(DATA_DIR, "applicants.json")
INDEX_CANDIDATOS_PATH = os.path.join(MODELS_DIR, 'index_candidatos.faiss')
METADADOS_CANDIDATOS_PATH = os.path.join(MODELS_DIR, 'candidatos_metadados.pkl')

# Nome do modelo de embeddings (consistência com servicos.py)
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2' # AJUSTADO: Definido

# Lista de campos obrigatórios para o formulário
CAMPOS_OBRIGATORIOS = [
    "nome", "email", "cv_pt" # Campos do formulário
]

# --- CARREGAMENTO DE RECURSOS (MODELO DE EMBEDDING) ---
# O @st.cache_resource garante que o modelo seja carregado APENAS UMA VEZ.
@st.cache_resource
def carregar_modelo_embedding():
    """Carrega o modelo de embedding uma única vez."""
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        st.success(f"Modelo de embedding '{EMBEDDING_MODEL_NAME}' carregado com sucesso.")
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo de embedding: {e}. As funcionalidades de indexação podem falhar.")
        return None

# Carrega o modelo de embedding globalmente
MODELO_EMBEDDING_GLOBAL = carregar_modelo_embedding()

# --- FUNÇÕES AUXILIARES ---

# Use @st.cache_data para carregar dados JSON para a sessão
@st.cache_data
def carregar_dados_candidatos():
    """
    Carrega os dados dos candidatos do arquivo JSON como uma lista de dicionários.
    """
    if os.path.exists(ARQUIVO_CANDIDATOS):
        with open(ARQUIVO_CANDIDATOS, "r", encoding="utf-8") as arquivo:
            try:
                data = json.load(arquivo)
                if isinstance(data, list):
                    return data
                else:
                    st.warning("O arquivo JSON de candidatos não está no formato de lista. Iniciando lista vazia.")
                    return []
            except json.JSONDecodeError:
                st.warning("Arquivo JSON de candidatos vazio ou inválido. Iniciando lista vazia.")
                return []
    return []

def salvar_dados_candidatos(dados):
    """
    Salva a lista de candidatos no arquivo JSON.
    """
    os.makedirs(os.path.dirname(ARQUIVO_CANDIDATOS) or ".", exist_ok=True)
    with open(ARQUIVO_CANDIDATOS, "w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, indent=4, ensure_ascii=False)

def gerar_proximo_id(dados):
    """
    Retorna o próximo id sequencial baseado no maior infos_basicas_codigo_profissional na lista.
    """
    codigos = [
        int(item.get("infos_basicas_codigo_profissional", 0))
        for item in dados if str(item.get("infos_basicas_codigo_profissional", "")).isdigit()
    ]
    if not codigos:
        return "10000" # Primeiro ID se a lista estiver vazia ou sem IDs válidos
    return str(max(codigos) + 1)

def adicionar_candidato_ao_indice(texto_candidato, candidato_id):
    """
    Gera o embedding do texto do candidato e o adiciona ao índice FAISS.
    """
    if MODELO_EMBEDDING_GLOBAL is None:
        return "❌ Erro: Modelo de embedding não carregado. Não foi possível adicionar o candidato ao índice."

    try:
        embedding = MODELO_EMBEDDING_GLOBAL.encode([texto_candidato])[0].astype(np.float32)

        # Cria o diretório 'models1' (ou 'models') se não existir
        os.makedirs(os.path.dirname(INDEX_CANDIDATOS_PATH) or ".", exist_ok=True)

        if os.path.exists(INDEX_CANDIDATOS_PATH) and os.path.exists(METADADOS_CANDIDATOS_PATH):
            index = faiss.read_index(INDEX_CANDIDATOS_PATH)
            metadados = pd.read_pickle(METADADOS_CANDIDATOS_PATH)
            # Para evitar duplicatas no metadado, se o candidato_id já existe,
            # você pode remover a entrada antiga e adicionar a nova, ou apenas sobrescrever.
            # No contexto de cadastro, um novo ID significa uma nova entrada, então apenas adicionamos.
        else:
            # Assumimos que a dimensão do embedding é 768 para o modelo 'paraphrase-multilingual-mpnet-base-v2'
            index = faiss.IndexFlatL2(embedding.shape[0])
            metadados = pd.DataFrame(columns=["id_original", "faiss_id"]) # Coluna para o ID interno do FAISS

        faiss_internal_id = index.ntotal # Pega o próximo ID interno do FAISS
        index.add(np.array([embedding]))

        novo_metadado = pd.DataFrame([{"id_original": candidato_id, "faiss_id": faiss_internal_id}])
        metadados = pd.concat([metadados, novo_metadado], ignore_index=True)

        faiss.write_index(index, INDEX_CANDIDATOS_PATH)
        metadados.to_pickle(METADADOS_CANDIDATOS_PATH)
        return "✅ Candidato adicionado ao índice vetorial com sucesso!"
    except Exception as e:
        st.error(f"❌ Erro ao adicionar ao índice vetorial: {e}")
        return f"❌ Erro ao adicionar ao índice vetorial: {e}"

# --- FUNÇÃO PRINCIPAL DE CADASTRO ---
def cadastro_candidatos():
    st.title("📋 Cadastro de Candidatos")

    dados_candidatos = carregar_dados_candidatos()

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
        data_nascimento = st.date_input("Data de nascimento", value=datetime(2000, 1, 1)) # Valor padrão para evitar erro se não preenchido

        local = st.text_input("Local (Cidade, Estado)")

        cv_pt = st.text_area("Resumo/CV em Português *")

        submitted = st.form_submit_button("Salvar Candidato")

        if submitted:
            # Validação de campos obrigatórios
            if not nome.strip() or not email.strip() or not cv_pt.strip():
                st.error("⚠️ Preencha todos os campos obrigatórios (*)")
                return

            codigo_candidato = gerar_proximo_id(dados_candidatos)
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
                "infos_basicas_codigo_profissional": codigo_candidato,
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
                "informacoes_pessoais_pcd": "Não", # Assumindo "Não" para cadastro manual, pode ser um selectbox
                "informacoes_pessoais_endereco": local,
                "informacoes_pessoais_skype": "",
                "informacoes_pessoais_url_linkedin": "",
                "informacoes_pessoais_facebook": "",
                "informacoes_pessoais_download_cv": None,

                "informacoes_profissionais_titulo_profissional": "", # Pode ser adicionado como campo de input
                "informacoes_profissionais_area_atuacao": area_atuacao,
                "informacoes_profissionais_conhecimentos_tecnicos": "",
                "informacoes_profissionais_certificacoes": "",
                "informacoes_profissionais_outras_certificacoes": "",
                "informacoes_profissionais_remuneracao": remuneracao,
                "informacoes_profissionais_nivel_profissional": "", # Pode ser adicionado como campo de input
                "informacoes_profissionais_qualificacoes": None,
                "informacoes_profissionais_experiencias": None,

                "formacao_e_idiomas_nivel_academico": "", # Pode ser adicionado como campo de input
                "formacao_e_idiomas_nivel_ingles": "", # Pode ser adicionado como campo de input
                "formacao_e_idiomas_nivel_espanhol": "", # Pode ser adicionado como campo de input
                "formacao_e_idiomas_outro_idioma": "", # Pode ser adicionado como campo de input
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
                "cv_en": "" # Assumindo que CV em inglês não é cadastrado aqui
            }

            dados_candidatos.append(novo_candidato)
            salvar_dados_candidatos(dados_candidatos)

            # Preparar texto para embedding
            from gerar_tudo import extrair_texto_candidato # Importa aqui para ter certeza que está disponível
            texto_para_embedding = extrair_texto_candidato(novo_candidato)

            resultado_faiss = adicionar_candidato_ao_indice(texto_para_embedding, codigo_candidato)

            st.success(f"✅ Candidato cadastrado com sucesso! Código: {codigo_candidato}")
            st.info(resultado_faiss)

            # Limpa o cache para que a lista de candidatos seja recarregada na próxima execução da página
            carregar_dados_candidatos.clear()

    # --- Seção de importação via JSON ---
    st.divider()
    st.markdown("### 📁 Cadastro via arquivo JSON")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Importar JSON do(s) candidato(s)", type="json")
    with col2:
        # Botão para baixar modelo JSON, adaptado para lista
        modelo_exemplo_cand = [
            {
                "infos_basicas_codigo_profissional": "10000",
                "infos_basicas_nome": "Nome Exemplo",
                "infos_basicas_email": "exemplo@email.com",
                "infos_basicas_telefone": "99999-9999",
                "infos_basicas_celular": "99999-9999",
                "infos_basicas_objetivo_profissional": "Desenvolvedor Backend com foco em Python e APIs.",
                "infos_basicas_area_atuacao": "Tecnologia",
                "infos_basicas_remuneracao": "R$ 8.000,00",
                "informacoes_pessoais_sexo": "Masculino",
                "informacoes_pessoais_estado_civil": "Solteiro",
                "informacoes_pessoais_cpf": "123.456.789-00",
                "informacoes_pessoais_data_nascimento": "01-01-1990",
                "infos_basicas_local": "São Paulo, SP",
                "cv_pt": "Experiência em desenvolvimento de sistemas Python, Flask, Django. Conhecimento em bancos de dados SQL e NoSQL. Habilidades em microsserviços e integração contínua."
            }
        ]
        st.download_button(
            label="Baixar Modelo JSON (lista)",
            data=json.dumps(modelo_exemplo_cand, indent=4, ensure_ascii=False),
            file_name="modelo_candidato.json",
            mime="application/json"
        )

    if uploaded_file:
        try:
            arquivo_json = json.load(uploaded_file)
            if not isinstance(arquivo_json, list):
                st.error("O arquivo deve conter uma lista de candidatos.")
                return
            
            dados_candidatos_atuais = carregar_dados_candidatos() # Pega a lista mais atualizada
            max_id_atual = gerar_proximo_id(dados_candidatos_atuais) # Pega o próximo ID para começar

            novos_candidatos_importados = []
            for cand_dict in arquivo_json:
                # Gera um novo ID e atribui
                cand_dict["infos_basicas_codigo_profissional"] = max_id_atual
                # Define datas de criação/atualização
                agora = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                cand_dict["infos_basicas_data_criacao"] = agora
                cand_dict["infos_basicas_data_atualizacao"] = agora
                cand_dict["infos_basicas_inserido_por"] = "Importação Streamlit"
                
                # Preenche campos ausentes com vazios para manter a estrutura
                for key_template in list(modelo_exemplo_cand[0].keys()) + [
                    "infos_basicas_telefone_recado", "infos_basicas_sabendo_de_nos_por",
                    "informacoes_pessoais_data_aceite", "informacoes_pessoais_nome",
                    "informacoes_pessoais_email_secundario", "informacoes_pessoais_telefone_recado",
                    "informacoes_pessoais_pcd", "informacoes_pessoais_endereco",
                    "informacoes_pessoais_skype", "informacoes_pessoais_url_linkedin",
                    "informacoes_pessoais_facebook", "informacoes_pessoais_download_cv",
                    "informacoes_profissionais_titulo_profissional", "informacoes_profissionais_conhecimentos_tecnicos",
                    "informacoes_profissionais_certificacoes", "informacoes_profissionais_outras_certificacoes",
                    "informacoes_profissionais_nivel_profissional", "informacoes_profissionais_qualificacoes",
                    "informacoes_profissionais_experiencias", "formacao_e_idiomas_nivel_academico",
                    "formacao_e_idiomas_nivel_ingles", "formacao_e_idiomas_nivel_espanhol",
                    "formacao_e_idiomas_outro_idioma", "formacao_e_idiomas_instituicao_ensino_superior",
                    "formacao_e_idiomas_cursos", "formacao_e_idiomas_ano_conclusao",
                    "formacao_e_idiomas_outro_curso", "cargo_atual_id_ibrati",
                    "cargo_atual_email_corporativo", "cargo_atual_cargo_atual",
                    "cargo_atual_projeto_atual", "cargo_atual_cliente",
                    "cargo_atual_unidade", "cargo_atual_data_admissao",
                    "cargo_atual_data_ultima_promocao", "cargo_atual_nome_superior_imediato",
                    "cargo_atual_email_superior_imediato", "cv_en"
                ]:
                    if key_template not in cand_dict:
                        cand_dict[key_template] = "" # Define string vazia para campos de texto
                
                # Campos que podem ser None/null
                if "informacoes_profissionais_qualificacoes" not in cand_dict: cand_dict["informacoes_profissionais_qualificacoes"] = None
                if "informacoes_profissionais_experiencias" not in cand_dict: cand_dict["informacoes_profissionais_experiencias"] = None
                if "formacao_e_idiomas_outro_curso" not in cand_dict: cand_dict["formacao_e_idiomas_outro_curso"] = None
                if "informacoes_pessoais_download_cv" not in cand_dict: cand_dict["informacoes_pessoais_download_cv"] = None
                
                # Trata a data de nascimento, se presente e no formato esperado
                if "informacoes_pessoais_data_nascimento" in cand_dict and isinstance(cand_dict["informacoes_pessoais_data_nascimento"], str):
                    try:
                        # Tenta converter para o formato que você está usando (DD-MM-YYYY)
                        datetime.strptime(cand_dict["informacoes_pessoais_data_nascimento"], "%d-%m-%Y")
                    except ValueError:
                        cand_dict["informacoes_pessoais_data_nascimento"] = "" # Ou o que for mais apropriado
                
                novos_candidatos_importados.append(cand_dict)
                max_id_atual = str(int(max_id_atual) + 1) # Incrementa o ID para o próximo candidato
            
            # Adiciona os novos candidatos à lista existente e salva
            dados_candidatos_atuais.extend(novos_candidatos_importados)
            salvar_dados_candidatos(dados_candidatos_atuais)

            # Processa cada novo candidato para adicionar ao índice FAISS
            for candidato in novos_candidatos_importados:
                from gerar_tudo import extrair_texto_candidato # Importa aqui para ter certeza que está disponível
                texto_para_embedding = extrair_texto_candidato(candidato)
                adicionar_candidato_ao_indice(texto_para_embedding, candidato["infos_basicas_codigo_profissional"])

            st.success(f"✅ {len(novos_candidatos_importados)} candidato(s) importado(s) e adicionado(s) com sucesso!")
            # Limpa o cache para que a lista de candidatos seja recarregada
            carregar_dados_candidatos.clear()
            
        except json.JSONDecodeError:
            st.error("Erro ao decodificar o arquivo JSON. Certifique-se de que é um JSON válido.")
        except Exception as e:
            st.error(f"Erro ao importar arquivo JSON: {e}")
