import os
import sys

embeddings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'embeddings'))

# Adiciona o caminho ao sys.path se ainda não estiver lá
if embeddings_path not in sys.path:
    sys.path.insert(0, embeddings_path) # Usar insert(0, ...) para dar prioridade

# -----------------------------------------------------------

# Agora, as importações padrão e as importações do seu módulo 'gerar_tudo'
import faiss
import numpy as np
import pandas as pd
import json
import logging
import streamlit as st
from sentence_transformers import SentenceTransformer

# A importação do 'gerar_tudo' agora deve funcionar
from gerar_tudo import extrair_texto_vaga, extrair_texto_candidato, extrair_texto_prospect

# Caminho base do projeto (onde está rodando este script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Diretórios de modelos e dados (um nível acima da pasta onde está este script)
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models1')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

# Nome do modelo de embeddings
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CARREGAMENTO DE RECURSOS GLOBAIS (CACHEADOS PELO STREAMLIT) ---
# O @st.cache_resource garante que estas funções sejam executadas APENAS UMA VEZ
# mesmo que o Streamlit re-execute o script (o que acontece frequentemente).

@st.cache_resource
def carregar_modelo_embedding():
    """Carrega o modelo de embedding uma única vez."""
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info(f"Modelo de embedding '{EMBEDDING_MODEL_NAME}' carregado com sucesso.")
        return model
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo de embedding: {e}. As buscas não funcionarão.")
        st.warning(f"**Aviso:** O modelo de embedding não pôde ser carregado. As funcionalidades de busca de similaridade estarão limitadas. Erro: {e}")
        return None

@st.cache_resource
def carregar_todos_dados_e_indices():
    """
    Carrega todos os dados originais, índices FAISS e metadados.
    Retorna um dicionário com todos os recursos.
    """
    recursos = {
        "vagas_originais": {},
        "candidatos_originais": {},
        "prospects_data_list": [], # Carrega prospects como uma lista para facilitar a busca
        "index_vagas": None,
        "metadados_vagas": pd.DataFrame(),
        "index_candidatos": None,
        "metadados_candidatos": pd.DataFrame(),
        "index_prospects": None,
        "metadados_prospects": pd.DataFrame(),
    }

    # Funções auxiliares para carregar dados JSON e converter para dicionário com ID
    def _carregar_json_para_dict(caminho_arquivo, id_key, default_prefix):
        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                data_raw = json.load(f)
                if isinstance(data_raw, list):
                    return {str(item.get(id_key, f"{default_prefix}_{i}")): item for i, item in enumerate(data_raw)}
                elif isinstance(data_raw, dict):
                    # Se for um único dicionário no nível raiz, tentar usar o id_key
                    if id_key in data_raw:
                        return {str(data_raw[id_key]): data_raw}
                    else: # Caso não tenha id_key, usar um ID anônimo
                        return {f"{default_prefix}_0": data_raw}
                else:
                    logging.warning(f"Formato inesperado para {caminho_arquivo}. Esperado lista ou dicionário.")
                    return {}
        except FileNotFoundError:
            logging.warning(f"Arquivo não encontrado: {caminho_arquivo}")
            st.warning(f"**Aviso:** Arquivo de dados '{os.path.basename(caminho_arquivo)}' não encontrado em '{DATA_DIR}'.")
            return {}
        except json.JSONDecodeError:
            logging.warning(f"Erro ao decodificar JSON: {caminho_arquivo}")
            st.error(f"**Erro:** Problema ao ler o arquivo JSON '{os.path.basename(caminho_arquivo)}'. Verifique o formato.")
            return {}
        except Exception as e:
            logging.warning(f"Erro inesperado ao carregar {caminho_arquivo}: {e}")
            st.error(f"**Erro inesperado** ao carregar '{os.path.basename(caminho_arquivo)}': {e}")
            return {}

    # Funções auxiliares para carregar FAISS e metadados
    def _carregar_faiss_index(nome_index):
        caminho_index = os.path.join(MODEL_DIR, f"index_{nome_index}.faiss")
        try:
            index = faiss.read_index(caminho_index)
            logging.info(f"Índice '{nome_index}' carregado com sucesso.")
            return index
        except Exception as e:
            logging.error(f"Erro ao carregar o índice {caminho_index}: {e}")
            st.warning(f"**Aviso:** Índice FAISS '{f'index_{nome_index}.faiss'}' não encontrado ou corrompido em '{MODEL_DIR}'. As buscas de similaridade para {nome_index} podem não funcionar.")
            return None

    def _carregar_metadados(nome_metadados):
        caminho_metadados = os.path.join(MODEL_DIR, f"{nome_metadados}_metadados.pkl")
        try:
            df = pd.read_pickle(caminho_metadados)
            logging.info(f"Metadados '{nome_metadados}' carregados com sucesso.")
            return df
        except Exception as e:
            logging.error(f"Erro ao carregar metadados {caminho_metadados}: {e}")
            st.warning(f"**Aviso:** Arquivo de metadados '{f'{nome_metadados}_metadados.pkl'}' não encontrado ou corrompido em '{MODEL_DIR}'. As buscas de similaridade para {nome_metadados} podem não funcionar corretamente.")
            return pd.DataFrame()

    # Carregamento dos dados originais
    recursos["vagas_originais"] = _carregar_json_para_dict(os.path.join(DATA_DIR, "vagas.json"), "id_vaga", "vaga_anon")
    recursos["candidatos_originais"] = _carregar_json_para_dict(os.path.join(DATA_DIR, "applicants.json"), "infos_basicas_codigo_profissional", "anon_cand")
    
    # Carrega prospects como uma lista para facilitar a busca por candidato/vaga
    try:
        with open(os.path.join(DATA_DIR, "prospects.json"), 'r', encoding='utf-8') as f:
            recursos["prospects_data_list"] = json.load(f) # Carrega como lista
        logging.info("Dados de prospects carregados como lista.")
    except FileNotFoundError:
        logging.warning(f"Arquivo não encontrado: {os.path.join(DATA_DIR, 'prospects.json')}")
        st.warning(f"**Aviso:** Arquivo de prospects 'prospects.json' não encontrado em '{DATA_DIR}'. A pontuação de histórico estará indisponível.")
        recursos["prospects_data_list"] = []
    except json.JSONDecodeError:
        logging.warning(f"Erro ao decodificar JSON: {os.path.join(DATA_DIR, 'prospects.json')}")
        st.error(f"**Erro:** Problema ao ler o arquivo JSON 'prospects.json'. Verifique o formato.")
        recursos["prospects_data_list"] = []
    except Exception as e:
        logging.error(f"Erro ao carregar prospects.json como lista: {e}")
        st.error(f"**Erro inesperado** ao carregar 'prospects.json': {e}")
        recursos["prospects_data_list"] = []

    # Carregamento dos índices e metadados
    # Estes serão None ou DataFrame vazios se os arquivos não existirem/estiverem corrompidos
    recursos["index_vagas"] = _carregar_faiss_index("vagas")
    recursos["metadados_vagas"] = _carregar_metadados("vagas")

    recursos["index_candidatos"] = _carregar_faiss_index("candidatos")
    recursos["metadados_candidatos"] = _carregar_metadados("candidatos")

    recursos["index_prospects"] = _carregar_faiss_index("prospects")
    recursos["metadados_prospects"] = _carregar_metadados("prospects")

    return recursos

# Carrega os recursos (modelo, dados, índices) uma única vez ao iniciar a aplicação
embedding_model = carregar_modelo_embedding()
recursos_carregados = carregar_todos_dados_e_indices()

# Atribui os recursos para uso fácil
vagas_originais = recursos_carregados["vagas_originais"]
candidatos_originais = recursos_carregados["candidatos_originais"]
prospects_data_list = recursos_carregados["prospects_data_list"] # Agora é uma lista
index_vagas = recursos_carregados["index_vagas"]
metadados_vagas = recursos_carregados["metadados_vagas"]
index_candidatos = recursos_carregados["index_candidatos"]
metadados_candidatos = recursos_carregados["metadados_candidatos"] 
index_prospects = recursos_carregados["index_prospects"]
metadados_prospects = recursos_carregados["metadados_prospects"]


# --- FUNÇÕES DE BUSCA DE SIMILARIDADE ---

def buscar_similares(query_embedding, faiss_index, metadados_df, k=10):
    """
    Realiza a busca de similaridade no índice FAISS.
    Args:
        query_embedding (np.array): O embedding da query (vaga ou candidato).
        faiss_index (faiss.Index): O índice FAISS onde a busca será feita.
        metadados_df (pd.DataFrame): DataFrame com os metadados correspondentes ao índice.
        k (int): Número de resultados a retornar.
    Returns:
        list: Uma lista de dicionários contendo os resultados da busca (id_original, distância).
    """
    if faiss_index is None or metadados_df.empty or query_embedding is None:
        logging.warning("Índice, metadados ou embedding da query inválido para busca. Retornando lista vazia.")
        st.warning("**Aviso:** Funcionalidade de busca de similaridade não disponível. Verifique se os arquivos de índice e metadados foram carregados corretamente.")
        return []

    # Assegura que query_embedding é um array 2D para faiss.search
    query_embedding = np.array([query_embedding]).astype(np.float32)

    try:
        distances, indices = faiss_index.search(query_embedding, k)
        resultados = []
        for dist, idx_faiss in zip(distances[0], indices[0]):
            if idx_faiss != -1 and idx_faiss < len(metadados_df): # Valida se o índice retornado é válido e dentro dos limites do DataFrame
                original_id = metadados_df.iloc[idx_faiss]['id_original']
                resultados.append({"id_original": original_id, "distancia": float(dist)})
        return resultados
    except Exception as e:
        logging.error(f"Erro durante a busca FAISS: {e}")
        st.error(f"**Erro:** Não foi possível realizar a busca de similaridade. Detalhes: {e}")
        return []

def calcular_pontuacao_historico(candidato_id, prospects_data):
    """
    Calcula uma pontuação de histórico para um candidato com base nas suas situações em vagas.
    Maior pontuação para situações mais positivas.
    """
    if not prospects_data: # Verifica se a lista de prospects está vazia
        return 0 # Nenhuma entrada de histórico para este candidato

    pontuacoes = {
        "Contratado": 10,
        "Encaminhado ao Requisitante": 8,
        "Entrevista com Cliente": 7,
        "Em Negociação": 6,
        "Em Andamento": 3,
        "Aguardando Contato": 2,
        "Em avaliação pelo RH": -1, # Ligeiramente negativo para indicar que está em outro processo
        "Desistiu": -5,
        "Rejeitado": -8,
        "Não Atende aos Requisitos": -10,
        "Outros": 0 # Situações não mapeadas
    }
    
    historico_pontos = []
    
    # Filtrar os prospects relevantes para o candidato
    prospects_do_candidato = [p for p in prospects_data if str(p.get("prospect_codigo")) == str(candidato_id)]
    
    if not prospects_do_candidato:
        return 0 # Nenhuma entrada de histórico para este candidato

    for prospect in prospects_do_candidato:
        situacao = prospect.get("prospect_situacao_candidado", "Outros")
        historico_pontos.append(pontuacoes.get(situacao, pontuacoes["Outros"]))
    
    # Retorna a média das pontuações do histórico do candidato
    return np.mean(historico_pontos) if historico_pontos else 0

def encontrar_candidatos_para_vaga(id_vaga, num_candidatos=5, peso_historico=0.3): # Valor padrão de 0.3 (30%)
    """
    Busca candidatos aderentes a uma vaga específica, calculando a pontuação de aderência
    e ponderando pelo histórico do candidato.
    """
    if embedding_model is None:
        return {"erro": "Modelo de embedding não carregado. Não é possível realizar a busca de similaridade."}
    
    if index_candidatos is None or metadados_candidatos.empty:
        return {"erro": "Índice de candidatos ou metadados não carregados. Não é possível realizar a busca de similaridade."}

    vaga_data = vagas_originais.get(str(id_vaga))
    if not vaga_data:
        logging.warning(f"Vaga com ID '{id_vaga}' não encontrada nos dados originais.")
        return {"erro": f"Vaga com ID '{id_vaga}' não encontrada."}

    texto_vaga = extrair_texto_vaga(vaga_data)
    if not texto_vaga:
        logging.warning(f"Vaga ID '{id_vaga}' não possui texto útil para gerar query embedding.")
        return {"erro": "Informações insuficientes na vaga para realizar a busca."}

    try:
        query_embedding = embedding_model.encode([texto_vaga])[0].astype(np.float32)
    except Exception as e:
        logging.error(f"Erro ao gerar embedding para a vaga: {e}")
        return {"erro": f"Erro ao gerar embedding para a vaga. Detalhes: {e}"}


    # Buscar mais candidatos do que o necessário inicialmente para filtrar e ordenar
    resultados_similares = buscar_similares(query_embedding, index_candidatos, metadados_candidatos, k=num_candidatos * 5) # Buscamos mais para ter histórico

    if not resultados_similares:
        return [] # Retorna lista vazia se nenhuma similaridade for encontrada

    candidatos_encontrados = []
    
    # Encontrar a distância máxima para normalização (se houver resultados)
    # Consideramos apenas distâncias positivas para evitar problemas em caso de 0
    distancias_validas = [res['distancia'] for res in resultados_similares if res['distancia'] >= 0] # Distancia pode ser 0 em match perfeito
    max_distancia = max(distancias_validas) if distancias_validas else 1.0 # Garante que não é zero

    for res in resultados_similares:
        candidato_id = str(res['id_original'])
        candidato_detalhes = candidatos_originais.get(candidato_id)
        if candidato_detalhes:
            # Calcular Pontuação de Aderência (baseada em similaridade textual)
            if max_distancia > 0:
                # Quanto menor a distância, maior a similaridade. Invertemos para pontuação: (1 - dist/max_dist)
                pontuacao_aderencia_similaridade = (1 - (res['distancia'] / max_distancia)) * 100
            else: # Se todas as distâncias forem zero (match perfeito ou apenas um resultado com dist=0)
                pontuacao_aderencia_similaridade = 100 if res['distancia'] == 0 else 0 

            # Calcular Pontuação de Histórico
            pontuacao_historico = calcular_pontuacao_historico(candidato_id, prospects_data_list)
            
            # Normalizar pontuação de histórico para uma escala de 0-100
            # Nossas pontuações vão de -10 a 10. Reescalamos para (x - min) / (max - min) * 100
            min_hist_score = -10 
            max_hist_score = 10  
            
            if (max_hist_score - min_hist_score) > 0:
                pontuacao_historico_normalizada = ((pontuacao_historico - min_hist_score) / (max_hist_score - min_hist_score)) * 100
            else:
                pontuacao_historico_normalizada = 50 # Neutro se não houver variação na pontuação

            # Pontuação Final Ponderada
            pontuacao_final = (pontuacao_aderencia_similaridade * (1 - peso_historico)) + \
                               (pontuacao_historico_normalizada * peso_historico)
            
            # Garante que a pontuação final esteja entre 0 e 100
            pontuacao_final = max(0, min(100, pontuacao_final))

            candidatos_encontrados.append({
                "id_candidato": candidato_id,
                "Pontuação Final de Aderência (0-100)": round(pontuacao_final, 2), 
                # As pontuações abaixo serão usadas apenas internamente para depuração ou análises futuras,
                # não serão exibidas na interface para simplificar.
                "Pontuação de Similaridade (0-100)_debug": round(pontuacao_aderencia_similaridade, 2),
                "Pontuação de Histórico (Média)_debug": round(pontuacao_historico, 2), 
                "Nome do Profissional": candidato_detalhes.get("infos_basicas_nome", "Nome não disponível"),
                "Email": candidato_detalhes.get("infos_basicas_email", "Não informado"),
                "Telefone": candidato_detalhes.get("infos_basicas_telefone", "Não informado"),
                "Título Profissional": candidato_detalhes.get("informacoes_profissionais_titulo_profissional", "Não informado"),
                "Distância Euclidiana (Referência)": res['distancia'], 
                "Dados Completos": json.dumps(candidato_detalhes, ensure_ascii=False, indent=2) 
            })
    
    # Ordenar pela Pontuação Final de Aderência (decrescente = mais aderente) e pegar os top N
    candidatos_encontrados_df = pd.DataFrame(candidatos_encontrados)
    if not candidatos_encontrados_df.empty:
        candidatos_encontrados_df = candidatos_encontrados_df.sort_values(by="Pontuação Final de Aderência (0-100)", ascending=False).head(num_candidatos)
    
    return candidatos_encontrados_df.to_dict(orient='records')


# --- INTERFACE STREAMLIT ---

def pagina_servicos():
    st.title("Sistema de Recomendação de Talentos")
    st.markdown("Bem-vindo(a)! Utilize o sistema para encontrar os **candidatos mais aderentes a uma vaga**, considerando sua similaridade com a descrição da vaga e seu histórico de engajamento em processos seletivos anteriores.")

    # Exibe métricas de quantos itens foram carregados
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total de Vagas", value=len(vagas_originais))
    with col2:
        st.metric(label="Total de Candidatos", value=len(candidatos_originais))
    with col3:
        # Contamos os prospects únicos por prospect_codigo para exibir um número mais realista de "candidatos prospectados"
        unique_prospects = len(set([p.get("prospect_codigo") for p in prospects_data_list if p.get("prospect_codigo")]))
        st.metric(label="Total de Interações de Prospects", value=len(prospects_data_list), help="Número de registros de interação (candidato-vaga) no histórico de prospects.")
        # Opcional: st.metric(label="Prospects Únicos", value=unique_prospects)

    st.markdown("---")

    # --- Seção principal: Encontrar Candidatos para uma Vaga ---
    st.header("🎯 Encontrar Candidatos para uma Vaga")
    st.markdown("Para encontrar os candidatos ideais, insira o **ID da vaga** e o **número de candidatos** que deseja exibir.")

    with st.form("form_vaga_candidato"):
        col_vaga1, col_vaga2 = st.columns([0.7, 0.3]) # Colunas ajustadas
        with col_vaga1:
            vaga_id_input = st.text_input("ID da Vaga", help="Ex: 1055, 11589").strip()
        with col_vaga2:
            num_candidatos_input = st.slider("Número de Candidatos a exibir", min_value=1, max_value=20, value=5)
        
        # O peso do histórico é fixado no backend, não mais na interface
        peso_historico_normalized = 0.3 # <--- PESO DO HISTÓRICO PADRÃO DEFINIDO AQUI (30%)

        submit_vaga_candidato = st.form_submit_button("Buscar Candidatos")

    if submit_vaga_candidato:
        if vaga_id_input:
            with st.spinner("Buscando candidatos e analisando aderência..."):
                vaga_info = vagas_originais.get(vaga_id_input)
                if not vaga_info:
                    st.error(f"Vaga com ID '{vaga_id_input}' não encontrada. Por favor, verifique o ID.")
                else:
                    st.subheader(f"Resultados encontrados para a Vaga ID: {vaga_id_input}")
                    st.markdown(f"**Vaga:** {vaga_info.get('info_titulo_vaga', 'Não disponível')}")
                    st.markdown(f"**Cliente:** {vaga_info.get('info_cliente', 'Não informado')}")
                    
                    # --- TRECHO ATUALIZADO: Extração e exibição da descrição COMPLETA da vaga ---
                    vaga_description_parts = []
                    # Priorizamos 'perfil_principais_atividades' por ser o mais comum para descrição
                    if vaga_info.get('perfil_principais_atividades'):
                        vaga_description_parts.append(vaga_info['perfil_principais_atividades'])
                    
                    # Adicionamos outras partes relevantes se existirem, para garantir a descrição completa
                    if vaga_info.get('perfil_competencia_tecnicas_e_comportamentais'):
                        vaga_description_parts.append("\n\n**Competências Técnicas e Comportamentais:**\n" + vaga_info['perfil_competencia_tecnicas_e_comportamentais'])
                    if vaga_info.get('perfil_demais_observacoes'):
                        vaga_description_parts.append("\n\n**Outras Observações:**\n" + vaga_info['perfil_demais_observacoes'])
                    
                    full_description = "\n".join(filter(None, (p.strip() for p in vaga_description_parts))).strip()

                    if full_description:
                        st.markdown(f"**Descrição:**\n{full_description}") # Exibe a descrição completa
                    else:
                        st.markdown("**Descrição:** Não disponível.")
                    # --- FIM TRECHO ATUALIZADO ---

                    resultados_df_raw = encontrar_candidatos_para_vaga(vaga_id_input, num_candidatos_input, peso_historico_normalized)
                    
                    if isinstance(resultados_df_raw, dict) and "erro" in resultados_df_raw:
                        st.error(resultados_df_raw["erro"])
                    elif not resultados_df_raw:
                        st.info("Nenhum candidato similar encontrado para esta vaga com os critérios atuais ou funcionalidades de busca não disponíveis.")
                    else:
                        # Converter para DataFrame para melhor visualização e manipulação
                        resultados_df = pd.DataFrame(resultados_df_raw)
                        
                        # Reordenar colunas para a visualização, exibindo apenas a pontuação final
                        cols_display = [
                            "Nome do Profissional", 
                            "Email", 
                            "Telefone", 
                            "Pontuação Final de Aderência (0-100)", 
                            "id_candidato"
                        ]
                        resultados_df_display = resultados_df[cols_display].copy()
                        
                        # Formatar coluna de pontuação para 2 casas decimais
                        resultados_df_display["Pontuação Final de Aderência (0-100)"] = resultados_df_display["Pontuação Final de Aderência (0-100)"].apply(lambda x: f"{x:.2f}")

                        st.markdown("---")
                        st.write("Abaixo estão os candidatos mais aderentes à vaga, ordenados pela **maior Pontuação Final de Aderência**:")
                        st.dataframe(resultados_df_display.set_index('id_candidato'), use_container_width=True)

                        # Botão de download para todos os resultados
                        # NOTA: O CSV de download ainda terá as colunas de debug para análise se necessário,
                        # mas não serão visíveis na tela.
                        csv_data = resultados_df.drop(columns=["Dados Completos", "Pontuação de Similaridade (0-100)_debug", "Pontuação de Histórico (Média)_debug"]).to_csv(index=False).encode('utf-8') 
                        st.download_button(
                            label="Download de Todos os Dados dos Candidatos (CSV)",
                            data=csv_data,
                            file_name=f"candidatos_vaga_{vaga_id_input}.csv",
                            mime="text/csv",
                            key=f"download_csv_vaga_{vaga_id_input}"
                        )

                        # Adicionar opção de download individual (JSON) para cada candidato
                        st.markdown("---")
                        st.subheader("Download individual dos dados completos dos candidatos:")
                        for idx, row in resultados_df.iterrows():
                            col_name, col_download = st.columns([0.7, 0.3])
                            with col_name:
                                st.write(f"**{row['Nome do Profissional']}** (ID: {row['id_candidato']})")
                            with col_download:
                                st.download_button(
                                    label="Download JSON",
                                    data=row['Dados Completos'].encode('utf-8'),
                                    file_name=f"candidato_{row['id_candidato']}.json",
                                    mime="application/json",
                                    key=f"download_json_cand_{row['id_candidato']}"
                                )
        else:
            st.warning("Por favor, insira um ID de vaga para buscar.")

    st.markdown("---")
