# TSNE USAGE TO COMPARE THE IMPACT OF IMAGE DESCRIPTION ON SKIN LESION RECOGNITION
## An Analysis with the PAD-UFES-20 Dataset

We've used the dataset PAD-UFES-20 to explore the impact of image description on the skin lesion recognition. To transform the sentences into vectors we'd used BERT to get the embbedings values referents to thegenerated sentences.

---
# Create enviroment:
```bash
    conda create -n eda-sentences
```
# To activate this environment, use
```bash
    conda activate eda-sentences
```
# Install libraries
To install the necessary libraries just write:

```bash
    pip3 install -r requirements.txt
```

# Run TSNE script
Add your csv file on the data folder and then run the script below:

```bash
    python3 src/scripts/tsne_of_sentences.py
```

It'll generate an image which contain 1 plot about: llm, vlm, or llm and vlm together.
![Plot of the TSNE projections](./images/tsne_sentence_embeddings_metadata_with_sentences_new-prompt-deepseek-r1:70b.png)

# Run KMeans analysis
 
You can use the KMeans analysis to evaluate the sentences.

When using the PAD-UFES-20 dataset, with the sentences text on the folder 'dataimages/umap_metadata_with_sentences_of_patient_description_and_image-description_llm-qwen2.5:72b_vllm-gemma3:27b.png', just run code:

```bash
    python3 src/scripts/evaluate_num_clusters.py
```
![Plot of the KMeans](./images/umap_metadata_with_sentences_of_patient_description_and_image-description_llm-qwen2.5:72b_vllm-gemma3:27b.png)

*Obs.: If you want to do a binary analysis, just change the flag 'is_binary' from False to True.

# Join Different Images for Comparison

This script automatically combines the t-SNE projections of different sentence combinations (LLM, VLM, and LLM+VLM) into a single image.

```bash
    python3 src/scripts/join_images.py
```

# HDBSCAN 

Clustering of sentences using HDBSCAN

Just adjust the variables content:
```
    # Diretório base (igual ao original)
    base_dir = "./data/ICs- Vitor e Ana/DATASETS - LLMs e VLMs/IMAGE_DESCRIPTION/sentences-of-image-description"
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    reducer_method = "umap"
    min_cluster_size = 2

    output_root = "./results"
```

obs.: It's not necessary to define a k_min (number of clusters).
```bash
    python3 src/scripts/eva_num_clusters_HDBSCAN.py
```

It'll generate a jointed image which contains 3 subplots about: llm, vlm,and llm and vlm together, respectivelly.
![Subplot of the TSNE projections](./images/jointed_tsne_results/jointed_tsne_results.png)

# To deactivate an active environment, use
```bash
    conda deactivate
```