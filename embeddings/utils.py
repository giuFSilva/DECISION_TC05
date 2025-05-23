import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging  # Importe a biblioteca logging
import pandas as pd # Importe a biblioteca pandas

# --- Configuração ---
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"  # Modelo para embeddings
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Diretório base do script
DATA_DIR = os.path.join(BASE_DIR, "../data") # Diretório para dados
MODEL_DIR = os.path.join(BASE_DIR, "../models") # Diretório para modelos

# --- Configuração do Logging (Opcional, mas Recomendado) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Funções ---
def carregar_json(path):
    """
    Carrega dados de um arquivo JSON.

    Args:
        path (str): Caminho para o arquivo JSON.

    Returns:
        dict: Dados carregados do JSON ou um dicionário vazio em caso de erro.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Arquivo JSON não encontrado: {path}")
        print(f"Erro: Arquivo JSON não encontrado: {path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Erro ao decodificar JSON: {path}")
        print(f"Erro ao decodificar JSON: {path}")
        return {}
    except Exception as e:
        logging.exception(f"Erro inesperado ao carregar JSON de {path}: {e}") # Usando logging.exception
        print(f"Erro inesperado ao carregar JSON de {path}: {e}")
        return {}


def salvar_json(path, data):
    """
    Salva dados em um arquivo JSON.

    Args:
        path (str): Caminho para salvar o arquivo JSON.
        data (dict): Dados a serem salvos.
    """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Dados salvos em {path}")
        print(f"Dados salvos em {path}")
    except Exception as e:
        logging.exception(f"Erro ao salvar JSON em {path}: {e}") # Usando logging.exception
        print(f"Erro ao salvar JSON em {path}: {e}")


def gerar_embedding(texto, model_name=EMBEDDING_MODEL_NAME):
    """
    Gera um embedding para o texto usando o modelo Sentence Transformer.

    Args:
        texto (str): Texto para gerar o embedding.
        model_name (str, optional): Nome do modelo Sentence Transformer a ser usado.
                                     Padrão é EMBEDDING_MODEL_NAME.

    Returns:
        numpy.ndarray: Embedding do texto.
    """
    try:
        model = SentenceTransformer(model_name)
        return model.encode([texto])[0].astype(np.float32)
    except Exception as e:
        logging.error(f"Erro ao gerar embedding para '{texto}': {e}")
        print(f"Erro ao gerar embedding para '{texto}': {e}")
        return None  # Retorna None em caso de erro


def carregar_index(faiss_path):
    """
    Carrega um índice Faiss do disco.

    Args:
        faiss_path (str): Caminho para o arquivo do índice Faiss.

    Returns:
        faiss.Index: O índice Faiss carregado, ou None se o arquivo não existir.
    """
    if os.path.exists(faiss_path):
        try:
            return faiss.read_index(faiss_path)
        except Exception as e:
            logging.error(f"Erro ao carregar índice Faiss de {faiss_path}: {e}")
            print(f"Erro ao carregar índice Faiss de {faiss_path}: {e}")
            return None
    else:
        logging.warning(f"Índice Faiss não encontrado: {faiss_path}")
        print(f"Aviso: Índice Faiss não encontrado: {faiss_path}")
        return None


def salvar_index(index, path):
    """
    Salva um índice Faiss no disco.

    Args:
        index (faiss.Index): Índice Faiss a ser salvo.
        path (str): Caminho para salvar o índice.
    """
    try:
        faiss.write_index(index, path)
        logging.info(f"Índice Faiss salvo em {path}")
        print(f"Índice Faiss salvo em {path}")
    except Exception as e:
        logging.exception(f"Erro ao salvar índice Faiss em {path}: {e}") # Usando logging.exception
        print(f"Erro ao salvar índice Faiss em {path}: {e}")


def carregar_metadados(pickle_path):
    """
    Carrega metadados de um arquivo pickle.

    Args:
        pickle_path (str): Caminho para o arquivo pickle.

    Returns:
        pandas.DataFrame: DataFrame com os metadados, ou um DataFrame vazio em caso de erro ou arquivo inexistente.
    """
    try:
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        else:
            logging.warning(f"Arquivo de metadados não encontrado: {pickle_path}. Criando DataFrame vazio.")
            print(f"Aviso: Arquivo de metadados não encontrado: {pickle_path}. Criando DataFrame vazio.")
            return pd.DataFrame(columns=['id_original', 'embedding_id'])  # Colunas consistentes
    except Exception as e:
        logging.exception(f"Erro ao carregar metadados de {pickle_path}: {e}") # Usando logging.exception
        print(f"Erro ao carregar metadados de {pickle_path}: {e}")
        return pd.DataFrame(columns=['id_original', 'embedding_id'])


def salvar_metadados(df, path):
    """
    Salva metadados em um arquivo pickle.

    Args:
        df (pandas.DataFrame): DataFrame contendo os metadados.
        path (str): Caminho para salvar o arquivo pickle.
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump(df, f)
        logging.info(f"Metadados salvos em {path}")
        print(f"Metadados salvos em {path}")
    except Exception as e:
        logging.exception(f"Erro ao salvar metadados em {path}: {e}") # Usando logging.exception
        print(f"Erro ao salvar metadados em {path}: {e}")