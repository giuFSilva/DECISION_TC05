
import pandas as pd
from embeddings.utils import *

def adicionar_vaga(vaga_id, descricao):
    vaga_embedding = gerar_embedding(descricao)

    faiss_path = os.path.join(MODEL_DIR, "index_vagas.faiss")
    meta_path = os.path.join(MODEL_DIR, "vagas_metadados.pkl")

    index = carregar_index(faiss_path)
    index.add(np.array([vaga_embedding]))
    salvar_index(index, faiss_path)

    metadados = carregar_metadados(meta_path)
    novo_id = f"vaga_{len(metadados)}"
    metadados.loc[len(metadados)] = [vaga_id, novo_id]
    salvar_metadados(metadados, meta_path)