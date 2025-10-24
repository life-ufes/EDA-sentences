import os
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import utils.load_dataset as load_dataset_script
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main(csv_file_folder_path: str, csv_file_name: str, device: str = "cuda"):
    """
    Processa um único arquivo CSV para gerar embeddings, aplicar t-SNE e salvar o gráfico.

    Args:
        csv_file_folder_path (str): O caminho para o diretório contendo os arquivos CSV.
        csv_file_name (str): O nome do arquivo CSV (sem extensão) a ser processado.
        device (str, optional): O dispositivo para executar o modelo ('cuda' ou 'cpu'). Padrão é 'cuda'.
    """
    full_csv_path = os.path.join(csv_file_folder_path, csv_file_name + ".csv")
    print(f"\n--- Processando o arquivo: {full_csv_path} ---")

    try:
        dataset = load_dataset_script.load_data(file_folder_path=full_csv_path)

        # Carrega o modelo e o tokenizer pré-treinados apenas uma vez no início da função main
        # Isso evita recarregar o modelo para cada arquivo, otimizando o desempenho
        # Se você tiver MUITOS arquivos e memória limitada, pode ser necessário carregar e descarregar.
        # No entanto, para a maioria dos casos, carregar uma vez é mais eficiente.
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.to(device) # Move o modelo para o dispositivo (GPU/CPU)
        model.eval() # Coloca o modelo em modo de avaliação (importante para BERT)

        all_embeddings = []
        all_diagnostics = []

        print("Iniciando a obtenção dos embeddings...")
        for index, sentence_sample in dataset.iterrows():
            sentence_sample_text = sentence_sample["sentence"]
            sentence_diagnostic = sentence_sample["diagnostic"]

            inputs = tokenizer(sentence_sample_text, return_tensors='pt', truncation=True, padding=True).to(device)

            with torch.no_grad(): # Desativa o cálculo de gradientes para inferência, economizando memória
                outputs = model(**inputs)

            # Extrai o embedding do token [CLS]
            sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            all_embeddings.append(sentence_embedding)
            all_diagnostics.append(sentence_diagnostic)

        embeddings_array = np.array(all_embeddings)
        print(f"Total de embeddings obtidos. Shape: {embeddings_array.shape}\n")

        print("Aplicando t-SNE para redução de dimensionalidade...")
        # Parâmetros comuns para t-SNE: perplexity=30 é um bom ponto de partida, n_iter para convergência.
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        reduced_embeddings = tsne.fit_transform(embeddings_array)
        print(f"Embeddings reduzidos com t-SNE. Shape: {reduced_embeddings.shape}\n")

        # --- Plotagem e Salvamento ---
        print("Gerando o gráfico t-SNE...")

        tsne_df = pd.DataFrame(data=reduced_embeddings, columns=['Componente 1', 'Componente 2'])
        tsne_df['Diagnóstico'] = all_diagnostics

        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            x='Componente 1',
            y='Componente 2',
            hue='Diagnóstico',
            palette='viridis',
            data=tsne_df,
            legend='full',
            alpha=0.7
        )
        plt.title(f'Visualização t-SNE: {csv_file_name.replace("_", " ").title()} Sentenças')
        plt.xlabel('Componente t-SNE 1')
        plt.ylabel('Componente t-SNE 2')
        plt.grid(True, linestyle='--', alpha=0.6)

        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True) # Cria o diretório se não existir

        # Nome do arquivo de saída específico para cada CSV
        plot_path = os.path.join(output_dir, f"tsne_sentence_embeddings_{csv_file_name}.png")

        plt.savefig(plot_path, dpi=400, bbox_inches='tight') # Aumentado DPI para melhor qualidade
        print(f"Gráfico t-SNE salvo em: {plot_path}")
        plt.close() # Fecha a figura para liberar memória, crucial ao processar múltiplos arquivos
        # plt.show() # Removido plt.show() para evitar que cada gráfico apareça na tela individualmente

    except Exception as e:
        print(f"Erro ao processar o arquivo {full_csv_path}! Erro: {e}\n")
        # Não retorna None para continuar com os outros arquivos, apenas imprime o erro.


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}\n")

    # Define o diretório onde os arquivos CSV estão localizados
    file_directory_path = "./data/sentences-of-image-description" # sentences-of-patient-data-description" # "vllms-and-llms-sentences"

    # Lista todos os arquivos no diretório e filtra os CSVs
    csv_files_in_dir = [f for f in os.listdir(file_directory_path) if f.endswith('.csv')]

    if not csv_files_in_dir:
        print(f"Nenhum arquivo CSV encontrado no diretório: {file_directory_path}")
    else:
        for file_full_name in csv_files_in_dir:
            # Extrai o nome do arquivo sem a extensão .csv
            file_name_without_extension = os.path.splitext(file_full_name)[0]
            
            # Chama a função main para cada arquivo CSV
            main(csv_file_folder_path=file_directory_path, 
                 csv_file_name=file_name_without_extension, 
                 device=device)

    print("\n--- Processamento de todos os arquivos CSV concluído! ---")