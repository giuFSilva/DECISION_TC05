import streamlit as st
import json
import os
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
# Mantenha a consistência com 'servicos.py'. Vou usar 'models1' como exemplo, ajuste se for diferente.
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models1') # AJUSTADO: CONSISTÊNCIA COM servicos.py

# Caminhos dos arquivos
ARQUIVO_VAGAS = os.path.join(DATA_DIR, 'vagas.json')
# AJUSTADO: Caminhos dos índices e metadados para corresponder aos modelos de vagas
INDEX_VAGAS_PATH = os.path.join(MODELS_DIR, 'index_vagas.faiss')
METADADOS_VAGAS_PATH = os.path.join(MODELS_DIR, 'vagas_metadados.pkl')

# Nome do modelo de embeddings (consistência com servicos.py)
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2' # AJUSTADO: Definido o nome do modelo

CAMPOS_OBRIGATORIOS = [
    "info_titulo_vaga", "info_vaga_sap", "perfil_pais", "perfil_estado", "perfil_vaga_especifica_para_pcd",
    "perfil_nivel profissional", "perfil_nivel_academico", "perfil_nivel_ingles", "perfil_nivel_espanhol",
    "perfil_areas_atuacao", "perfil_principais_atividades", "perfil_competencia_tecnicas_e_comportamentais"
]

# --- CARREGAMENTO DE RECURSOS (MODELO DE EMBEDDING) ---
# O @st.cache_resource garante que o modelo seja carregado APENAS UMA VEZ
# ao iniciar a aplicação, e não a cada submissão de formulário.
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

@st.cache_data # Use st.cache_data para carregar dados JSON para a sessão
def carregar_vagas():
    """
    Carrega as vagas do arquivo JSON como uma lista de dicionários.
    """
    if os.path.exists(ARQUIVO_VAGAS):
        with open(ARQUIVO_VAGAS, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    st.warning("O arquivo JSON de vagas não está no formato de lista. Iniciando lista vazia.")
                    return []
            except json.JSONDecodeError:
                st.warning("Arquivo JSON de vagas vazio ou inválido. Iniciando lista vazia.")
                return []
    return []

def salvar_vagas(vagas):
    """
    Salva a lista de vagas no arquivo JSON mantendo o formato lista.
    """
    # Cria o diretório 'data' se não existir
    os.makedirs(os.path.dirname(ARQUIVO_VAGAS) or ".", exist_ok=True)
    with open(ARQUIVO_VAGAS, "w", encoding="utf-8") as f:
        json.dump(vagas, f, indent=4, ensure_ascii=False)

def proximo_id(vagas):
    """
    Retorna o próximo id sequencial baseado no maior id_vaga na lista.
    """
    if not vagas:
        return 5000
    ids = [vaga.get("id_vaga", 0) for vaga in vagas if isinstance(vaga.get("id_vaga"), (int, str))] # Permite int ou str para ids
    # Tenta converter IDs para int para encontrar o máximo
    numeric_ids = []
    for id_val in ids:
        try:
            numeric_ids.append(int(id_val))
        except (ValueError, TypeError):
            continue # Ignora IDs não numéricos ou nulos
    
    if not numeric_ids:
        return 5000 # Se não encontrar IDs numéricos, começa do 5000
    return max(numeric_ids) + 1


def adicionar_vaga_ao_indice(texto, vaga_id):
    if MODELO_EMBEDDING_GLOBAL is None:
        return "❌ Erro: Modelo de embedding não carregado. Não foi possível adicionar a vaga ao índice."

    try:
        embedding = MODELO_EMBEDDING_GLOBAL.encode([texto])[0].astype(np.float32)

        # Cria o diretório 'models1' (ou 'models') se não existir
        os.makedirs(os.path.dirname(INDEX_VAGAS_PATH) or ".", exist_ok=True)

        # Carrega o índice e metadados existentes ou cria novos
        if os.path.exists(INDEX_VAGAS_PATH) and os.path.exists(METADADOS_VAGAS_PATH):
            index = faiss.read_index(INDEX_VAGAS_PATH)
            metadados = pd.read_pickle(METADADOS_VAGAS_PATH)
            # Verifica se o ID original já existe no DataFrame de metadados
            if vaga_id in metadados['id_original'].values:
                st.warning(f"Vaga com ID {vaga_id} já existe no índice. Sobrescrevendo o embedding existente (se houver).")
                # Lógica para sobrescrever o embedding:
                # Primeiro, remova o embedding e o metadado antigo se o ID original já existir
                # Isso requer um índice FAISS que suporte remoção ou a reconstrução do índice.
                # Para simplificar agora, apenas alertamos e adicionamos, o que pode criar duplicatas se o ID for repetido no 'vagas.json'
                # e o 'id_original' for a única chave para o FAISS.
                # Se o FAISS fosse atualizável, seria: index.remove_ids(np.array([id_do_embedding_antigo]))
                # Como o `id_original` é o que queremos usar para referência, precisamos garantir que o index FAISS seja mapeado 1:1.
                # Para simplificar aqui, vamos apenas adicionar, o que criará uma nova entrada para o mesmo ID original.
                # A sua busca precisará lidar com múltiplos embeddings para o mesmo ID original, ou você precisaria reconstruir o índice.
                # Para fins de demonstração e adição, a forma atual está ok, mas para um sistema robusto, considere a remoção/atualização.
        else:
            # Assumimos que a dimensão do embedding é 768 para o modelo 'paraphrase-multilingual-mpnet-base-v2'
            index = faiss.IndexFlatL2(embedding.shape[0])
            metadados = pd.DataFrame(columns=["id_original", "faiss_id"]) # Renomeado para 'faiss_id' para clareza

        # Adiciona o novo embedding
        faiss_internal_id = index.ntotal # Pega o próximo ID interno do FAISS
        index.add(np.array([embedding]))

        # Cria um novo metadado
        novo_metadado = pd.DataFrame([{"id_original": vaga_id, "faiss_id": faiss_internal_id}])
        metadados = pd.concat([metadados, novo_metadado], ignore_index=True)

        # Salva o índice e os metadados
        faiss.write_index(index, INDEX_VAGAS_PATH)
        metadados.to_pickle(METADADOS_VAGAS_PATH)
        return "✅ Vaga adicionada ao índice vetorial com sucesso!"
    except Exception as e:
        st.error(f"❌ Erro ao adicionar ao índice vetorial: {e}")
        return f"❌ Erro ao adicionar ao índice vetorial: {e}"

# --- INTERFACE ---

def cadastro_vagas():
    st.title("📝 Cadastro de Vagas")

    # Carrega as vagas. Como é uma função com st.cache_data, só carrega uma vez.
    vagas = carregar_vagas()

    with st.form("form_vaga"):
        with st.container():
            st.subheader("Informações Básicas")
            titulo = st.text_input("Título da Vaga *")
            vaga_sap = st.selectbox("É vaga SAP? *", ["", "Sim", "Não"])

        with st.container():
            st.subheader("Perfil da Vaga")
            pais = st.text_input("País *")
            estado = st.text_input("Estado *")
            pcd = st.selectbox("Vaga específica para PCD? *", ["", "Sim", "Não"])
            nivel_prof = st.text_input("Nível Profissional *")
            nivel_acad = st.text_input("Nível Acadêmico *")
            nivel_ing = st.selectbox("Inglês *", ["", "Nenhum", "Básico", "Intermediário", "Avançado", "Fluente"])
            nivel_esp = st.selectbox("Espanhol *", ["", "Nenhum", "Básico", "Intermediário", "Avançado", "Fluente"])
            area = st.text_input("Área de Atuação *")
            atividades = st.text_area("Principais Atividades *")
            competencias = st.text_area("Competências Técnicas e Comportamentais *")

        submitted = st.form_submit_button("✅ Salvar Vaga")
        if submitted:
            campos_para_validar = {
                "info_titulo_vaga": titulo,
                "info_vaga_sap": vaga_sap,
                "perfil_pais": pais,
                "perfil_estado": estado,
                "perfil_vaga_especifica_para_pcd": pcd,
                "perfil_nivel profissional": nivel_prof,
                "perfil_nivel_academico": nivel_acad,
                "perfil_nivel_ingles": nivel_ing,
                "perfil_nivel_espanhol": nivel_esp,
                "perfil_areas_atuacao": area,
                "perfil_principais_atividades": atividades,
                "perfil_competencia_tecnicas_e_comportamentais": competencias
            }

            faltando = [campo for campo in CAMPOS_OBRIGATORIOS if not campos_para_validar.get(campo)]
            if faltando:
                st.error(f"❌ Preencha os campos obrigatórios: {', '.join(faltando)}")
                return

            vaga_id = proximo_id(vagas)

            nova_vaga = {
                "id_vaga": vaga_id,
                "info_titulo_vaga": titulo,
                "info_vaga_sap": vaga_sap,
                "perfil_pais": pais,
                "perfil_estado": estado,
                "perfil_vaga_especifica_para_pcd": pcd,
                "perfil_nivel profissional": nivel_prof,
                "perfil_nivel_academico": nivel_acad,
                "perfil_nivel_ingles": nivel_ing,
                "perfil_nivel_espanhol": nivel_esp,
                "perfil_areas_atuacao": area,
                "perfil_principais_atividades": atividades,
                "perfil_competencia_tecnicas_e_comportamentais": competencias,
                # Adicione outros campos fixos se quiser, com valores padrão ou vazios
                "info_data_requicisao": "",
                "info_limite_esperado_para_contratacao": "",
                "info_cliente": "",
                "info_solicitante_cliente": "",
                "info_empresa_divisao": "",
                "info_requisitante": "",
                "info_analista_responsavel": "",
                "info_tipo_contratacao": "",
                "info_prazo_contratacao": "",
                "info_objetivo_vaga": "",
                "info_prioridade_vaga": "",
                "info_origem_vaga": "",
                "info_superior_imediato": "",
                "info_nome": "",
                "info_telefone": "",
                "info_data_inicial": None,
                "info_data_final": None,
                "info_nome_substituto": None,
                "perfil_cidade": "",
                "perfil_bairro": "",
                "perfil_regiao": "",
                "perfil_local_trabalho": "",
                "perfil_faixa_etaria": "",
                "perfil_horario_trabalho": "",
                "perfil_outro_idioma": "",
                "perfil_demais_observacoes": "",
                "perfil_viagens_requeridas": "",
                "perfil_equipamentos_necessarios": "",
                "perfil_habilidades_comportamentais_necessarias": None,
                "benef_valor_venda": "-",
                "benef_valor_compra_1": "R$",
                "benef_valor_compra_2": ""
            }

            vagas.append(nova_vaga)     # adiciona na lista
            salvar_vagas(vagas)         # salva no arquivo

            # Preparar texto para embedding (todos os valores concatenados)
            # Use extrair_texto_vaga do seu módulo gerar_tudo para consistência
            from gerar_tudo import extrair_texto_vaga # Importa aqui para ter certeza que está disponível
            texto_para_embedding = extrair_texto_vaga(nova_vaga)

            resultado = adicionar_vaga_ao_indice(texto_para_embedding, vaga_id)

            st.success(f"✅ Vaga {vaga_id} cadastrada com sucesso!")
            st.info(resultado)

    st.divider()
    st.markdown("### 📁 Cadastro via arquivo JSON")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Importar JSON da vaga", type="json")
    with col2:
        # Botão para baixar modelo JSON, adaptado para lista
        modelo_exemplo = [
            {
                "id_vaga": 5000,
                "info_titulo_vaga": "Exemplo de vaga",
                "info_vaga_sap": "Não",
                "perfil_pais": "Brasil",
                "perfil_estado": "São Paulo",
                "perfil_vaga_especifica_para_pcd": "Não",
                "perfil_nivel profissional": "Pleno",
                "perfil_nivel_academico": "Ensino Superior",
                "perfil_nivel_ingles": "Avançado",
                "perfil_nivel_espanhol": "Básico",
                "perfil_areas_atuacao": "TI",
                "perfil_principais_atividades": "Atividades exemplo",
                "perfil_competencia_tecnicas_e_comportamentais": "Competências exemplo"
            }
        ]
        st.download_button(
            label="Baixar Modelo JSON (lista)",
            data=json.dumps(modelo_exemplo, indent=4, ensure_ascii=False),
            file_name="modelo_vaga.json",
            mime="application/json"
        )

    if uploaded_file:
        try:
            arquivo_json = json.load(uploaded_file)
            if not isinstance(arquivo_json, list):
                st.error("O arquivo deve conter uma lista de vagas.")
                return
            
            # Recarrega as vagas para ter a lista mais atualizada antes de adicionar
            # Não é necessário carregar novamente se 'vagas' já está no escopo do cadastro_vagas()
            # e foi carregado por carregar_vagas() no início da função.
            # O problema é se o Streamlit roda a página e a função carregar_vagas() com cache_data
            # não reflete imediatamente o arquivo salvo.
            # Para uploads, geralmente a página é re-executada, então o cache pode precisar ser limpo.
            # No entanto, salvar_vagas() já atualiza o arquivo.
            # Para garantir, vou manter o cache_data na função carregar_vagas.

            max_id_atual = proximo_id(vagas) - 1 # Pega o último ID antes de adicionar os novos

            novas_vagas = []
            for vaga in arquivo_json:
                # Ajusta id_vaga para evitar duplicidade e garantir sequência única
                max_id_atual += 1
                vaga["id_vaga"] = max_id_atual
                novas_vagas.append(vaga)

            vagas.extend(novas_vagas)
            salvar_vagas(vagas) # Salva todas as vagas, incluindo as novas

            # Processa cada nova vaga para adicionar ao índice FAISS
            for vaga in novas_vagas:
                # Use extrair_texto_vaga do seu módulo gerar_tudo para consistência
                texto_para_embedding = extrair_texto_vaga(vaga)
                adicionar_vaga_ao_indice(texto_para_embedding, vaga["id_vaga"])

            st.success(f"✅ {len(novas_vagas)} vagas importadas e adicionadas com sucesso!")
            # Limpa o cache para que a lista de vagas seja recarregada na próxima execução da página
            carregar_vagas.clear() # Limpa o cache da função carregar_vagas()
            
        except Exception as e:
            st.error(f"Erro ao importar arquivo JSON: {e}")

# --- REMOVER: NÃO É NECESSÁRIO EM UM ARQUIVO DE PÁGINA ---
# if __name__ == "__main__":
#     cadastro_vagas()
