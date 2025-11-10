#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
import torch


# ==========================
# CONFIGURAÇÕES GERAIS
# ==========================

BASE_DIR = "./data/ICs- Vitor e Ana/DATASETS - LLMs e VLMs/IMAGE_DESCRIPTION/vlmsentences/PAD20/gemma3"
OUTPUT_DIR = "./results/results_similarity_models"
EMBEDDING_MODEL = "neuralmind/bert-base-portuguese-cased"
SAMPLE_SIZE = 1000  # nº máx. de sentenças por modelo para evitar OOM


# ==========================
# FUNÇÕES AUXILIARES
# ==========================

def detect_text_column(df: pd.DataFrame):
    """
    Detecta automaticamente a coluna de texto se 'sentence' não existir.
    Retorna o nome da coluna ou None se não encontrar.
    """
    if "sentence" in df.columns:
        return "sentence"

    # candidatos comuns
    candidates = ["text", "texto", "descricao", "description", "sentence",
                  "sentenca", "sentença", "desc"]
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]

    return None


def load_sentences_from_dir(dir_path: str):
    """
    Lê todos os CSVs de um diretório e retorna um dicionário:
        { nome_modelo (sem .csv): [lista de sentenças] }
    Ignora arquivos que não tenham coluna de texto detectável.
    """
    models_data = {}

    if not os.path.isdir(dir_path):
        print(f"⚠️ Diretório não encontrado: {dir_path}")
        return models_data

    files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
    if not files:
        print(f"⚠️ Nenhum arquivo .csv encontrado em: {dir_path}")
        return models_data

    print(f"📁 CSVs encontrados em {dir_path}: {files}")

    for file in files:
        full_path = os.path.join(dir_path, file)
        try:
            df = pd.read_csv(full_path)
        except Exception as e:
            print(f"⚠️ Erro ao ler {full_path}: {e}")
            continue

        col = detect_text_column(df)
        if col is None:
            print(f"⚠️ {file} ignorado: nenhuma coluna de texto detectada.")
            continue

        df_text = df[col].astype(str).dropna()
        sentences = df_text.tolist()
        if len(sentences) < 2:
            print(f"⚠️ {file} ignorado: menos de 2 sentenças.")
            continue

        model_name = os.path.splitext(file)[0]
        models_data[model_name] = sentences
        print(f"   ✅ {model_name}: {len(sentences)} sentenças (coluna: {col})")

    return models_data


def compute_intra_similarity(embeddings):
    """
    Calcula similaridade de cosseno intra-modelo (todos os pares) e
    retorna um vetor com as similaridades (sem diagonal).
    embeddings: torch.Tensor [N, D]
    """
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    triu = np.triu_indices_from(sim_matrix, k=1)
    return sim_matrix[triu]


def analyze_models_similarity(models_data: dict,
                              base_model_name: str,
                              sample_size: int = 1000):
    """
    Gera embeddings com SentenceTransformer e calcula:
    - similaridade intra-modelo para cada modelo
    - similaridade inter-modelos para cada par de modelos

    Retorna:
        df_intra: DataFrame com colunas [model, mean_intra, std_intra, n_sentences]
        df_inter: DataFrame com colunas [pair, mean_inter, std_inter]
    """
    if not models_data:
        print("⚠️ Nenhum modelo/dataset carregado. Encerrando análise.")
        return pd.DataFrame(), pd.DataFrame()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔹 Usando modelo de embeddings: {base_model_name} ({device})")
    model = SentenceTransformer(base_model_name, device=device)

    intra_rows = []
    embeddings_by_model = {}
    sentences_by_model = {}

    # ===== Embeddings + Similaridade Intra =====
    for model_name, sentences in models_data.items():
        if len(sentences) > sample_size:
            sentences = list(np.random.choice(sentences, size=sample_size, replace=False))
            print(f"   🔄 {model_name}: amostrando {sample_size} sentenças.")

        print(f"\n📘 {model_name}: gerando embeddings ({len(sentences)} sentenças)...")
        embs = model.encode(
            sentences,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        embeddings_by_model[model_name] = embs
        sentences_by_model[model_name] = sentences

        if len(sentences) < 2:
            print(f"⚠️ {model_name}: menos de 2 sentenças após amostragem, pulando intra-similaridade.")
            continue

        sims = compute_intra_similarity(embs)
        intra_rows.append({
            "model": model_name,
            "mean_intra": float(np.mean(sims)),
            "std_intra": float(np.std(sims)),
            "n_sentences": int(len(sentences))
        })

    df_intra = pd.DataFrame(intra_rows)

    # ===== Similaridade Inter-Modelos =====
    inter_rows = []
    model_names = list(embeddings_by_model.keys())

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            embs1 = embeddings_by_model[m1]
            embs2 = embeddings_by_model[m2]

            if embs1.size(0) == 0 or embs2.size(0) == 0:
                print(f"⚠️ Pulando par {m1} vs {m2}: embeddings vazios.")
                continue

            print(f"   🔍 Calculando similaridade inter-modelos: {m1} vs {m2}")
            sim_inter = util.cos_sim(embs1, embs2).cpu().numpy().flatten()
            inter_rows.append({
                "pair": f"{m1} vs {m2}",
                "mean_inter": float(np.mean(sim_inter)),
                "std_inter": float(np.std(sim_inter))
            })

    df_inter = pd.DataFrame(inter_rows)

    return df_intra, df_inter


def plot_comparison(df_intra: pd.DataFrame,
                    df_inter: pd.DataFrame,
                    out_dir: str):
    """
    Gera gráficos de:
    - Similaridade média intra-modelo por modelo
    - Similaridade média inter-modelos por par
    Função robusta (não quebra com DF vazio ou colunas ausentes).
    """
    os.makedirs(out_dir, exist_ok=True)

    if df_intra is None:
        df_intra = pd.DataFrame()
    if df_inter is None:
        df_inter = pd.DataFrame()

    if df_intra.empty:
        print("⚠️ df_intra está vazio. Nenhum gráfico intra-modelo será gerado.")
    else:
        print("📋 df_intra columns:", df_intra.columns.tolist())
        print("df_intra head:\n", df_intra.head())

    if df_inter.empty:
        print("⚠️ df_inter está vazio. Nenhum gráfico inter-modelo será gerado.")
    else:
        print("📋 df_inter columns:", df_inter.columns.tolist())
        print("df_inter head:\n", df_inter.head())

    # --- Gráfico 1: Intra-modelo ---
    if not df_intra.empty and all(c in df_intra.columns for c in ["model", "mean_intra"]):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_intra, x="model", y="mean_intra", palette="viridis")
        plt.title("Similaridade Média Intra-Modelo")
        plt.ylabel("Similaridade de Cosseno")
        plt.xlabel("Modelo Gerador")
        plt.tight_layout()
        out_path = os.path.join(out_dir, "intra_similarity.png")
        plt.savefig(out_path, dpi=400)
        plt.close()
        print(f"✅ Gráfico intra-modelo salvo em: {out_path}")
    else:
        print("⚠️ Não foi possível gerar gráfico intra-modelo (faltam colunas esperadas ou DF vazio).")

    # --- Gráfico 2: Inter-modelos ---
    if not df_inter.empty and all(c in df_inter.columns for c in ["pair", "mean_inter"]):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_inter, x="pair", y="mean_inter", palette="magma")
        plt.title("Similaridade Média Inter-Modelos")
        plt.ylabel("Similaridade de Cosseno")
        plt.xlabel("Pares de Modelos")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_path = os.path.join(out_dir, "inter_similarity.png")
        plt.savefig(out_path, dpi=400)
        plt.close()
        print(f"✅ Gráfico inter-modelo salvo em: {out_path}")
    else:
        print("⚠️ Não foi possível gerar gráfico inter-modelo (faltam colunas esperadas ou DF vazio).")


# ==========================
# MAIN
# ==========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🔎 Lendo sentenças de: {BASE_DIR}")
    models_data = load_sentences_from_dir(BASE_DIR)

    print("\n📁 Modelos carregados:")
    for name, sents in models_data.items():
        print(f"   - {name}: {len(sents)} sentenças")

    df_intra, df_inter = analyze_models_similarity(
        models_data=models_data,
        base_model_name=EMBEDDING_MODEL,
        sample_size=SAMPLE_SIZE
    )

    # Salvar CSVs de resumo
    intra_path = os.path.join(OUTPUT_DIR, "intra_similarity_summary.csv")
    inter_path = os.path.join(OUTPUT_DIR, "inter_similarity_summary.csv")

    if not df_intra.empty:
        df_intra.to_csv(intra_path, index=False)
        print(f"💾 Resumo intra-modelo salvo em: {intra_path}")
    else:
        print("⚠️ Nenhum dado intra-modelo para salvar.")

    if not df_inter.empty:
        df_inter.to_csv(inter_path, index=False)
        print(f"💾 Resumo inter-modelos salvo em: {inter_path}")
    else:
        print("⚠️ Nenhum dado inter-modelos para salvar.")

    # Gráficos
    plot_comparison(df_intra, df_inter, OUTPUT_DIR)

    print(f"\n✅ Resultados salvos em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
