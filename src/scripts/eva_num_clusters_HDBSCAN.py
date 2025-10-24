import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import torch
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# LOG CONFIG
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(file_folder_path: str) -> pd.DataFrame:
    """Carrega um CSV (mesma interface do seu utils.load_dataset.load_data)."""
    if not os.path.exists(file_folder_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_folder_path}")

    df = pd.read_csv(file_folder_path)
    if "sentence" not in df.columns:
        raise ValueError(f"O arquivo {file_folder_path} não contém coluna 'sentence'.")

    return df


def generate_embeddings(sentences, model_name, device):
    """Gera embeddings usando SentenceTransformer."""
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True, device=device)
    logging.info(f"Embeddings gerados: {embeddings.shape}")
    return embeddings


def reduce_dimensions(embeddings, method="umap", n_components=2):
    """Reduz dimensionalidade para visualização."""
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    logging.info(f"Dimensões reduzidas para {n_components} componentes ({method}).")
    return reduced


def cluster_embeddings(embeddings, min_cluster_size=5):
    """Aplica HDBSCAN e retorna labels."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.sum(labels == -1) / len(labels)
    logging.info(f"Clusters encontrados: {n_clusters} | Ruído: {noise_ratio:.2%}")
    return labels


def plot_clusters(reduced, labels, output_path, title):
    """Gera gráfico de clusters 2D."""
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=reduced[:, 0],
        y=reduced[:, 1],
        hue=labels,
        palette="tab10",
        s=60,
        alpha=0.8
    )
    plt.title(title)
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend(title="Cluster", loc="best", bbox_to_anchor=(1.02, 1))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close()
    logging.info(f"📈 Gráfico salvo em: {output_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Usando dispositivo: {device}\n")

    # Diretório base (igual ao original)
    base_dir = "./data/ICs- Vitor e Ana/DATASETS - LLMs e VLMs/IMAGE_DESCRIPTION/sentences-of-image-description"
    model_name = "all-MiniLM-L6-v2"
    reducer_method = "umap"
    min_cluster_size = 5

    output_root = "./results"
    os.makedirs(output_root, exist_ok=True)

    csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]

    if not csv_files:
        logging.warning(f"Nenhum arquivo CSV encontrado em {base_dir}")
        return

    for csv_file in csv_files:
        file_path = os.path.join(base_dir, csv_file)
        file_name = os.path.splitext(csv_file)[0]
        logging.info(f"\n--- Processando arquivo: {file_name} ---")

        try:
            # === 1. Carregar dataset ===
            dataset = load_data(file_path)
            sentences = dataset["sentence"].astype(str).tolist()

            # === 2. Embeddings ===
            embeddings = generate_embeddings(sentences, model_name, device)

            # === 3. Clusterização ===
            labels = cluster_embeddings(embeddings, min_cluster_size)

            # === 4. Redução dimensional para visualização ===
            reduced = reduce_dimensions(embeddings, method=reducer_method)

            # === 5. Salvar resultados ===
            output_dir = os.path.join(output_root, file_name)
            os.makedirs(output_dir, exist_ok=True)

            df_out = pd.DataFrame({
                "sentence": sentences,
                "cluster": labels,
                "x": reduced[:, 0],
                "y": reduced[:, 1]
            })
            csv_out = os.path.join(output_dir, f"clusters_{file_name}.csv")
            df_out.to_csv(csv_out, index=False)

            plot_out = os.path.join(output_dir, f"clusters_{file_name}.png")
            plot_clusters(reduced, labels, plot_out, f"Clusters: {file_name}")

            logging.info(f"✅ Arquivo {file_name} processado com sucesso!\n")

        except Exception as e:
            logging.error(f"❌ Erro ao processar {csv_file}: {e}\n")

    logging.info("\n--- Processamento de todos os arquivos CSV concluído! ---")


if __name__ == "__main__":
    main()
