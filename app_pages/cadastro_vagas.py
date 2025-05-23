import streamlit as st
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd

# --- CONFIGURAÇÕES ---
ARQUIVO_VAGAS = "C:/Users/giuliasilva/Desktop/Estudo/POS/TC - Modulo 05/application_web/data/vagas.json"
INDEX_PATH = "C:/Users/giuliasilva/Desktop/Estudo/POS/TC - Modulo 05/application_web/models/index_vagas.faiss"
META_PATH = "C:/Users/giuliasilva/Desktop/Estudo/POS/TC - Modulo 05/application_web/models/vagas_metadados.pkl"
EMBEDDING_MODEL = 'paraphrase-multilingual-mpnet-base-v2'

CAMPOS_OBRIGATORIOS = [
    "info_titulo_vaga", "info_vaga_sap", "perfil_pais", "perfil_estado", "perfil_vaga_especifica_para_pcd",
    "perfil_nivel profissional", "perfil_nivel_academico", "perfil_nivel_ingles", "perfil_nivel_espanhol",
    "perfil_areas_atuacao", "perfil_principais_atividades", "perfil_competencia_tecnicas_e_comportamentais"
]

# --- FUNÇÕES AUXILIARES ---

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
                    st.warning("O arquivo JSON não está no formato de lista. Iniciando lista vazia.")
                    return []
            except json.JSONDecodeError:
                st.warning("Arquivo JSON vazio ou inválido. Iniciando lista vazia.")
                return []
    return []

def salvar_vagas(vagas):
    """
    Salva a lista de vagas no arquivo JSON mantendo o formato lista.
    """
    os.makedirs(os.path.dirname(ARQUIVO_VAGAS) or ".", exist_ok=True)
    with open(ARQUIVO_VAGAS, "w", encoding="utf-8") as f:
        json.dump(vagas, f, indent=4, ensure_ascii=False)

def proximo_id(vagas):
    """
    Retorna o próximo id sequencial baseado no maior id_vaga na lista.
    """
    if not vagas:
        return 5000
    ids = [vaga.get("id_vaga", 0) for vaga in vagas if isinstance(vaga.get("id_vaga"), int)]
    if not ids:
        return 5000
    return max(ids) + 1

def adicionar_vaga_ao_indice(texto, vaga_id):
    try:
        modelo = SentenceTransformer(EMBEDDING_MODEL)
        embedding = modelo.encode([texto])[0].astype(np.float32)

        if os.path.exists(INDEX_PATH):
            index = faiss.read_index(INDEX_PATH)
            metadados = pd.read_pickle(META_PATH)
        else:
            index = faiss.IndexFlatL2(768)  # dimensão do embedding do modelo
            metadados = pd.DataFrame(columns=["id_original", "embedding_id"])

        novo_id_embedding = f"vaga_{len(metadados) + 1}"
        index.add(np.array([embedding]))
        metadados = pd.concat([
            metadados,
            pd.DataFrame([{"id_original": vaga_id, "embedding_id": novo_id_embedding}])
        ], ignore_index=True)

        faiss.write_index(index, INDEX_PATH)
        metadados.to_pickle(META_PATH)
        return "✅ Vaga adicionada ao índice vetorial com sucesso!"
    except Exception as e:
        return f"❌ Erro ao adicionar ao índice vetorial: {e}"

# --- INTERFACE ---

def cadastro_vagas():
    st.title("📝 Cadastro de Vagas")

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

            vagas.append(nova_vaga)    # adiciona na lista
            salvar_vagas(vagas)        # salva no arquivo

            # Preparar texto para embedding (todos os valores concatenados)
            texto = " ".join([str(v) for v in nova_vaga.values()])
            resultado = adicionar_vaga_ao_indice(texto, vaga_id)

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
            vagas = carregar_vagas()
            max_id_atual = proximo_id(vagas) - 1

            novas_vagas = []
            for vaga in arquivo_json:
                # Ajusta id_vaga para evitar duplicidade
                max_id_atual += 1
                vaga["id_vaga"] = max_id_atual
                novas_vagas.append(vaga)

            vagas.extend(novas_vagas)
            salvar_vagas(vagas)

            for vaga in novas_vagas:
                texto = " ".join([str(v) for v in vaga.values()])
                adicionar_vaga_ao_indice(texto, vaga["id_vaga"])

            st.success(f"✅ {len(novas_vagas)} vagas importadas e adicionadas com sucesso!")
        except Exception as e:
            st.error(f"Erro ao importar arquivo JSON: {e}")

# --- RODAR APLICATIVO ---
if __name__ == "__main__":
    cadastro_vagas()
