import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import logging
from datetime import datetime

# --- CONFIGURAÇÃO ---
DATA_DIR = 'C:/Users/giuliasilva/Desktop/Estudo/POS/TC - Modulo 05/application_web/data' # Ajuste conforme seu caminho
MODEL_DIR = 'C:/Users/giuliasilva/Desktop/Estudo/POS/TC - Modulo 05/application_web/models' # Ajuste conforme seu caminho

# Modelos que serão utilizados (adicione ou remova conforme necessidade)
EMBEDDING_MODELS = {
    "original": 'paraphrase-multilingual-mpnet-base-v2',  # Seu modelo atual
    "e5_large": 'intfloat/multilingual-e5-large',           # Modelo mais novo e robusto
    "all_mpnet_base": 'sentence-transformers/all-mpnet-base-v2' # Muito forte em inglês, bom geral
}

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FUNÇÕES AUXILIARES ---

def carregar_json(caminho):
    """
    Carrega um arquivo JSON e retorna seu conteúdo.
    """
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logging.error(f"Arquivo não encontrado: {caminho}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Erro ao decodificar JSON: {caminho}")
        return None
    except Exception as e:
        logging.error(f"Erro inesperado ao carregar JSON {caminho}: {e}")
        return None

def gerar_embedding(texto, modelo):
    """Gera o embedding de um texto usando o modelo."""
    if not texto or not isinstance(texto, str):
        return None
    try:
        clean_text = ' '.join(texto.split()).strip()
        if not clean_text:
            return None
        return modelo.encode([clean_text])[0].astype(np.float32)
    except Exception as e:
        logging.error(f"Erro ao gerar embedding para o texto: '{texto[:50]}...' - {e}")
        return None

def salvar_index(index, caminho):
    """Salva um índice Faiss."""
    try:
        faiss.write_index(index, caminho)
        logging.info(f"Índice salvo em: {caminho}")
    except Exception as e:
        logging.error(f"Erro ao salvar índice {caminho}: {e}")

def salvar_metadados(df, caminho):
    """Salva metadados em formato pickle."""
    try:
        df.to_pickle(caminho)
        logging.info(f"Metadados salvos em: {caminho}")
    except Exception as e:
        logging.error(f"Erro ao salvar metadados {caminho}: {e}")

# --- FUNÇÕES DE EXTRAÇÃO DE TEXTO ---

def extrair_texto_vaga(vaga_data):
    """
    Extrai e concatena informações relevantes de uma vaga para gerar o embedding.
    Prioriza campos que descrevem as necessidades e o perfil da vaga.
    """
    partes = []
    # Títulos e Níveis
    partes.append(f"Vaga: {vaga_data.get('info_titulo_vaga', '')}")
    partes.append(f"Nível Profissional: {vaga_data.get('perfil_nivel_profissional', vaga_data.get('perfil_nivel profissional', ''))}") # Compatibilidade
    partes.append(f"Nível Acadêmico: {vaga_data.get('perfil_nivel_academico', '')}")
    
    # Atividades e Competências
    partes.append(f"Atividades Principais: {vaga_data.get('perfil_principais_atividades', '')}")
    partes.append(f"Competências Técnicas e Comportamentais: {vaga_data.get('perfil_competencia_tecnicas_e_comportamentais', '')}")
    
    # Área e Tipo de Contratação
    partes.append(f"Área de Atuação: {vaga_data.get('perfil_areas_atuacao', '')}")
    partes.append(f"Tipo de Contratação: {vaga_data.get('info_tipo_contratacao', '')}")
    
    # Idiomas (apenas se especificado e não vazio/nenhum)
    if vaga_data.get("perfil_nivel_ingles") and vaga_data['perfil_nivel_ingles'].lower() not in ["", "nenhum"]:
        partes.append(f"Inglês Requerido: {vaga_data['perfil_nivel_ingles']}")
    if vaga_data.get("perfil_nivel_espanhol") and vaga_data['perfil_nivel_espanhol'].lower() not in ["", "nenhum"]:
        partes.append(f"Espanhol Requerido: {vaga_data['perfil_nivel_espanhol']}")
    if vaga_data.get("perfil_outro_idioma") and vaga_data['perfil_outro_idioma'].lower() not in ["", "nenhum", "-"]:
        partes.append(f"Outro Idioma Requerido: {vaga_data['perfil_outro_idioma']}")

    # Localização e Observações
    local = []
    if vaga_data.get("perfil_cidade"):
        local.append(vaga_data["perfil_cidade"])
    if vaga_data.get("perfil_estado"):
        local.append(vaga_data["perfil_estado"])
    if local:
        partes.append(f"Localização da Vaga: {', '.join(local)}")
    
    partes.append(f"Local de Trabalho: {vaga_data.get('perfil_local_trabalho', '')}")
    partes.append(f"Observações da Vaga: {vaga_data.get('perfil_demais_observacoes', '')}")
    partes.append(f"Cliente: {vaga_data.get('info_cliente', '')}")

    texto_final = " ".join(p.strip() for p in partes if p and isinstance(p, str)).strip()
    return texto_final if texto_final else None

def extrair_texto_candidato(candidato_data):
    """
    Extrai e concatena informações relevantes de um candidato para gerar o embedding.
    Prioriza o currículo, habilidades e experiências.
    """
    partes = []
    # CV
    if candidato_data.get("cv_pt"):
        partes.append(candidato_data["cv_pt"])
    
    # Informações Básicas e Profissionais
    infos_basicas = candidato_data.get("infos_basicas", {})
    if infos_basicas.get("objetivo_profissional"):
        partes.append(f"Objetivo Profissional: {infos_basicas['objetivo_profissional']}")

    info_prof = candidato_data.get("informacoes_profissionais", {})
    if info_prof.get("titulo_profissional"):
        partes.append(f"Título Profissional: {info_prof['titulo_profissional']}")
    if info_prof.get("area_atuacao"):
        partes.append(f"Área de Atuação: {info_prof['area_atuacao']}")
    if info_prof.get("conhecimentos_tecnicos"):
        partes.append(f"Conhecimentos Técnicos: {info_prof['conhecimentos_tecnicos']}")
    if info_prof.get("certificacoes"):
        partes.append(f"Certificações: {info_prof['certificacoes']}")
    if info_prof.get("qualificacoes"):
        partes.append(f"Qualificações: {info_prof['qualificacoes']}")
    
    # Experiências: Verifica se é lista e extrai
    experiencias_list = info_prof.get("experiencias", [])
    if isinstance(experiencias_list, list):
        for exp in experiencias_list:
            if isinstance(exp, dict):
                exp_text = []
                if exp.get("cargo"): exp_text.append(f"Cargo: {exp['cargo']}")
                if exp.get("empresa"): exp_text.append(f"Empresa: {exp['empresa']}")
                if exp.get("descricao"): exp_text.append(f"Descrição: {exp['descricao']}")
                if exp_text:
                    partes.append("Experiência: " + ", ".join(exp_text))
            elif isinstance(exp, str): # Caso seja uma string simples na lista
                    partes.append(f"Experiência: {exp}")
    elif isinstance(experiencias_list, str): # Caso 'experiencias' seja um texto diretamente
        partes.append(f"Experiências: {experiencias_list}")

    # Formação e Idiomas
    formacao_idiomas = candidato_data.get("formacao_e_idiomas", {})
    if formacao_idiomas.get("nivel_academico"):
        partes.append(f"Nível Acadêmico: {formacao_idiomas['nivel_academico']}")
    if formacao_idiomas.get("instituicao_ensino_superior"):
        partes.append(f"Instituição de Ensino Superior: {formacao_idiomas['instituicao_ensino_superior']}")
    if formacao_idiomas.get("cursos"):
        partes.append(f"Cursos: {formacao_idiomas['cursos']}")
    
    if formacao_idiomas.get("nivel_ingles") and formacao_idiomas['nivel_ingles'].lower() not in ["", "nenhum"]:
        partes.append(f"Nível de Inglês: {formacao_idiomas['nivel_ingles']}")
    if formacao_idiomas.get("nivel_espanhol") and formacao_idiomas['nivel_espanhol'].lower() not in ["", "nenhum"]:
        partes.append(f"Nível de Espanhol: {formacao_idiomas['nivel_espanhol']}")
    if formacao_idiomas.get("outro_idioma") and formacao_idiomas['outro_idioma'].lower() not in ["", "nenhum", "-"]:
        partes.append(f"Outro Idioma: {formacao_idiomas['outro_idioma']}")

    # Outros dados
    informacoes_pessoais = candidato_data.get("informacoes_pessoais", {})
    if informacoes_pessoais.get("url_linkedin"):
        partes.append(f"LinkedIn: {informacoes_pessoais['url_linkedin']}")

    texto_final = " ".join(p.strip() for p in partes if p and isinstance(p, str)).strip()
    return texto_final if texto_final else None

def extrair_texto_prospect(prospect_data):
    """
    Extrai e concatena informações relevantes de um prospect.
    """
    partes = []
    partes.append(prospect_data.get("titulo", ""))
    partes.append(prospect_data.get("prospect_nome", ""))
    partes.append(prospect_data.get("prospect_situacao_candidado", ""))
    partes.append(prospect_data.get("prospect_comentario", ""))

    texto_final = " ".join(p.strip() for p in partes if p and isinstance(p, str)).strip()
    return texto_final if texto_final else None

# --- FUNÇÃO PRINCIPAL DE GERAÇÃO DE ÍNDICES ---
def gerar_indices_para_todos_os_modelos():
    logging.info("Iniciando geração de índices para múltiplos modelos...")

    # Carrega os dados brutos
    vagas_raw = carregar_json(os.path.join(DATA_DIR, "vagas.json"))
    candidatos_raw = carregar_json(os.path.join(DATA_DIR, "applicants.json"))
    prospects_raw = carregar_json(os.path.join(DATA_DIR, "prospects.json"))

    if not any([vagas_raw, candidatos_raw, prospects_raw]):
        logging.error("Nenhum dado carregado. Abortando.")
        return

    # Prepara os dados (normaliza para um dicionário {id: data})
    def preparar_dados_para_processamento(dados_brutos, chave_id, tipo_dado):
        out = {}
        if isinstance(dados_brutos, list):
            for i, item in enumerate(dados_brutos):
                item_id = str(item.get(chave_id, f"{tipo_dado}_{i}"))
                if item_id in out:
                    logging.warning(f"ID duplicado {item_id} em {tipo_dado}. Ignorado.")
                    continue
                out[item_id] = item
        elif isinstance(dados_brutos, dict):
            # Se for um único dicionário no nível raiz, use a chave_id ou um fallback
            item_id = str(dados_brutos.get(chave_id, f"{tipo_dado}_0"))
            out[item_id] = dados_brutos
            if chave_id not in dados_brutos:
                logging.warning(f"Arquivo {tipo_dado}.json é um único dicionário sem '{chave_id}'. Usando '{item_id}' como ID.")
        else:
            logging.warning(f"Arquivo {tipo_dado}.json não está no formato esperado (lista ou dicionário). Ignorando.")
        return out

    vagas = preparar_dados_para_processamento(vagas_raw, "id_vaga", "vaga")
    candidatos = preparar_dados_para_processamento(candidatos_raw, "infos_basicas_codigo_profissional", "candidato")
    prospects = preparar_dados_para_processamento(prospects_raw, "prospect_codigo", "prospect")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Loop sobre modelos
    for apelido_modelo, nome_modelo in EMBEDDING_MODELS.items():
        logging.info(f"Iniciando processamento para o modelo: {apelido_modelo} ({nome_modelo})")

        try:
            modelo = SentenceTransformer(nome_modelo)
            dim = modelo.get_sentence_embedding_dimension()
            logging.info(f"Modelo '{nome_modelo}' carregado. Dimensão: {dim}")
        except Exception as e:
            logging.error(f"Falha ao carregar modelo {nome_modelo}: {e}. Pulando para o próximo.")
            continue

        # Inicializa FAISS e listas de metadados para o modelo atual
        index_vagas = faiss.IndexFlatL2(dim)
        index_candidatos = faiss.IndexFlatL2(dim)
        index_prospects = faiss.IndexFlatL2(dim)

        metadados_vagas = []
        metadados_candidatos = []
        metadados_prospects = []

        # Função interna para processar e adicionar embeddings
        def processar_e_adicionar(dados_dict, extrator_func, faiss_index, metadados_list, tipo_dado_nome):
            total_processed = 0
            total_indexed = 0
            for item_id, item_data in dados_dict.items():
                texto = extrator_func(item_data)
                if texto:
                    emb = gerar_embedding(texto, modelo) # Usa o modelo carregado no loop externo
                    if emb is not None:
                        faiss_index.add(np.array([emb]))
                        metadados_list.append({"id_original": item_id, "texto_original": texto})
                        total_indexed += 1
                    else:
                        logging.warning(f"Não foi possível gerar embedding para {tipo_dado_nome} ID: {item_id} (texto vazio após limpeza ou erro no modelo).")
                else:
                    logging.warning(f"{tipo_dado_nome} ID: {item_id} sem texto útil para embedding (campos importantes vazios).")
                total_processed += 1
            logging.info(f"{tipo_dado_nome}: {total_processed} processados, {total_indexed} indexados.")
            return total_processed, total_indexed

        # Processa os dados para o modelo atual
        logging.info(f"Processando vagas para {apelido_modelo}...")
        processar_e_adicionar(vagas, extrair_texto_vaga, index_vagas, metadados_vagas, "vaga")

        logging.info(f"Processando candidatos para {apelido_modelo}...")
        processar_e_adicionar(candidatos, extrair_texto_candidato, index_candidatos, metadados_candidatos, "candidato")

        logging.info(f"Processando prospects para {apelido_modelo}...")
        processar_e_adicionar(prospects, extrair_texto_prospect, index_prospects, metadados_prospects, "prospect")

        # Salva os índices e metadados para o modelo atual
        if index_vagas.ntotal > 0:
            salvar_index(index_vagas, os.path.join(MODEL_DIR, f'faiss_index_vagas_{apelido_modelo}.index'))
            salvar_metadados(pd.DataFrame(metadados_vagas), os.path.join(MODEL_DIR, f'metadados_vagas_{apelido_modelo}.pkl'))
        else:
            logging.warning(f"Nenhuma vaga indexada para o modelo {apelido_modelo}. Arquivos não serão criados.")

        if index_candidatos.ntotal > 0:
            salvar_index(index_candidatos, os.path.join(MODEL_DIR, f'faiss_index_candidatos_{apelido_modelo}.index'))
            salvar_metadados(pd.DataFrame(metadados_candidatos), os.path.join(MODEL_DIR, f'metadados_candidatos_{apelido_modelo}.pkl'))
        else:
            logging.warning(f"Nenhum candidato indexado para o modelo {apelido_modelo}. Arquivos não serão criados.")

        if index_prospects.ntotal > 0:
            salvar_index(index_prospects, os.path.join(MODEL_DIR, f'faiss_index_prospects_{apelido_modelo}.index'))
            salvar_metadados(pd.DataFrame(metadados_prospects), os.path.join(MODEL_DIR, f'metadados_prospects_{apelido_modelo}.pkl'))
        else:
            logging.warning(f"Nenhum prospect indexado para o modelo {apelido_modelo}. Arquivos não serão criados.")

        logging.info(f"Finalizado para o modelo: {apelido_modelo}")

    logging.info("Geração de todos os índices concluída com sucesso.")

if __name__ == "__main__":
    gerar_indices_para_todos_os_modelos()