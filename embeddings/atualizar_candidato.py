a
import pandas as pd
from embeddings.utils import *

def adicionar_candidato(cand_id, descricao):
    cand_embedding = gerar_embedding(descricao)

    faiss_path = os.path.join(MODEL_DIR, "index_candidatos.faiss")
    meta_path = os.path.join(MODEL_DIR, "candidatos_metadados.pkl")

    index = carregar_index(faiss_path)
    index.add(np.array([cand_embedding]))
    salvar_index(index, faiss_path)

    metadados = carregar_metadados(meta_path)
    novo_id = f"cand_{len(metadados)}"
    metadados.loc[len(metadados)] = [cand_id, novo_id]
    salvar_metadados(metadados, meta_path)
