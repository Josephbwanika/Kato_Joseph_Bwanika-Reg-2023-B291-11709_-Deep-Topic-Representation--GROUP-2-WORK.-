# Deep Topic Representation for Swahili News

**Student Name:** Kato Joseph Bwanika  
**Registration Number:** Reg:2023-B291-11709               
**Course:** Bachelors In Computer Science             
**Institution:** Uganda Martyrs University                
**Project Title:** Deep Topic Representation for Swahili News Articles  
---

## Abstract
This project explores deep topic representation of Swahili-language news using neural network-based embedding and compression models. The main objective is to develop an effective method for representing and clustering Swahili news articles by leveraging **Sentence-BERT embeddings**, **Deep Autoencoders**, and **Latent Dirichlet Allocation (LDA)**.  
The study demonstrates that deep models can capture semantic meaning in Swahili text, reduce dimensional complexity, and reveal meaningful topic clusters. Results show excellent performance in terms of reconstruction accuracy, topic separation, and cross-language generalization.
---

## Introduction
Swahili is one of the most widely spoken languages in East Africa but remains underexplored in computational linguistics.  
This project addresses that gap by developing a **deep topic representation model** for Swahili news.  
By combining **semantic embeddings** and **autoencoder-based dimensionality reduction**, the model learns to represent Swahili text in a compact form while maintaining semantic relationships.  
The research also evaluates multilingual generalization to assess how well Swahili semantics align with English equivalents.
---

## Dataset Description
**Dataset Source:** [Hugging Face – mteb/swahili_news](https://huggingface.co/datasets/mteb/swahili_news)
**Composition:**
- **Training Samples:** 17,789 articles  
- **Testing Samples:** 2,048 articles  
- **Categories:**  
  - Entertainment (*burudani*)  
  - Economy (*uchumi*)  
  - International (*kimataifa*)  
  - National (*kitaifa*)  
  - Health (*afya*)  
  - Sports (*michezo*)

**Language:** Swahili  
**Average Article Length:** ~2,461 characters (~369 words)  
**Quality:** High — well-structured, balanced distribution, and minimal missing entries.

##  Methodology
### 1. Semantic Embedding
Swahili articles were transformed into **384-dimensional embeddings** using the multilingual **Sentence-BERT MiniLM model** (`paraphrase-multilingual-MiniLM-L12-v2`).  
These embeddings capture semantic meaning beyond word-level similarity.

### 2. Deep Autoencoder Architecture
A **symmetric autoencoder** was designed to compress and reconstruct embeddings efficiently.

**Architecture Design:**
- **Encoder:** `384 → 128 → 64 → 32 → 16 → 2`  
- **Decoder:** `2 → 16 → 32 → 64 → 128 → 384`

**Training Parameters:**
- Activation: ReLU  
- Optimizer: Adam  
- Loss Function: Mean Squared Error (MSE)  
- Epochs: 50  
This setup achieved a **192:1 compression ratio** while retaining meaningful semantics.

### 3. Topic Modeling
An **LDA (Latent Dirichlet Allocation)** model was used as a comparative baseline to identify latent topics from 8,896 unique Swahili words, ensuring interpretability of thematic structures.
---

## Experimental Setup
**Tools and Libraries:**
- Python (Jupyter Notebook)  
- `torch`, `sentence-transformers`, `numpy`, `scikit-learn`, `gensim`, and `matplotlib`
  
**Environment:**  
Experiments were run on CPU/GPU environments for 50 epochs, with loss monitoring and topic visualization.
---

## Results and Analysis
### 1. Autoencoder Performance
| Metric | Value |
|--------|-------|
| Final Validation Loss | **0.02097** |
| Reconstruction Error (MSE) | **0.02126** |
| Best Configuration | **Shallow 2-Layer ReLU Autoencoder** |
| Compression Ratio | **192:1** |

- The model achieved stable and consistent loss curves.  
- Minimal overfitting was observed between training and validation losses.

### 2. Clustering Visualization
- **Distinct Groups:** Sports and Health articles formed well-separated clusters.  
- **Partial Overlap:** Economy, National, and International categories exhibited natural overlap, reflecting real-world topic relationships.

### 3. Multilingual Evaluation
Comparison of Swahili and English translations showed strong cross-language semantic alignment.
| Category | Semantic Distance (Lower = Better) |
|-----------|------------------------------------|
| International | **1.74** |
| Entertainment | **2.38** |
| Sports | **2.67** |
| Health | **3.19** |
| National | **3.49** |
| Economy | **4.79** |

This confirms that the model captures conceptual meaning across languages, with international and entertainment topics showing the strongest alignment.
---

## Discussion
The **simple 2-layer ReLU autoencoder** performed better than deeper or more complex architectures.  
This indicates that Swahili embeddings produced by Sentence-BERT are already semantically rich, requiring minimal non-linear transformation.  
Clusters of sports and health articles showed high separation due to domain-specific vocabulary, while overlap among economy and politics-related categories reflects real-world thematic interdependence.
---

## Conclusion
The study successfully demonstrates that **deep autoencoders** can compress Swahili news embeddings while preserving semantic integrity.  
The models effectively identify topic clusters, generalize across data variations, and capture cross-language relationships.  
This provides a foundation for future research in **low-resource language processing** and **semantic topic modeling**.
---

## Limitations
- Economic and political topics exhibit greater contextual variation, reducing cross-language consistency.  
- The model assumes formal Swahili, which may limit performance on dialects or informal expressions.  
- The dataset is limited to news content; results may differ for other text genres.
---

## Future Work
- Extend the approach to **other Bantu languages** for regional semantic comparisons.  
- Integrate **interactive dashboards** for topic exploration and clustering visualization.  
- Investigate **Variational Autoencoders (VAE)** or **Transformers** for enhanced representation learning.  
- Evaluate performance on informal text and user-generated content.

## References
- Swahili News Dataset — Hugging Face  
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*  
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.  


*Submitted by:*  
**Kato Joseph Bwanika**  
Reg:2023-B291-11709
Uganda Martyrs University

