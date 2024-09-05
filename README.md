# Natural Language Inference with Semantic Role Labeling in Graph-Encoded Transformers

## Description
This project explores advanced methods for **Natural Language Inference (NLI)** by leveraging **Graph-Encoded Transformers** enhanced with **Semantic Role Labeling (SRL)**. NLI, also known as Recognizing Textual Entailment (RTE), is a key task in **Natural Language Processing (NLP)** where the system determines whether a premise sentence supports, contradicts, or is neutral to a hypothesis sentence.

### Key Highlights:
- **Graph-Encoded Transformers**: This model enhances traditional transformers by incorporating graph structures of sentences. Instead of relying purely on token positions, the graph-transformer uses **semantic relations** between words to enable more context-aware attention mechanisms.
  
- **Semantic Role Labeling (SRL)**: SRL is applied to identify the roles that words play in a sentence, such as agent, action, and target. These structures are integrated into the model to improve the system’s understanding of sentence meaning, leading to better performance on NLI tasks.

- **Multi-Genre Natural Language Inference (MNLI) Dataset**: The project uses the MNLI dataset, which contains a diverse set of sentence pairs annotated for entailment, contradiction, and neutrality, across multiple genres.

- **Performance Comparison**: The graph-encoded transformer model is compared against a vanilla transformer baseline. Notably, the graph-encoded model shows improved performance on mismatched datasets, which demonstrates its ability to handle out-of-domain sentences better due to the additional semantic information.

### Use Cases:
- **Question Answering (QA)**: NLI can help determine whether candidate responses in a document logically answer a given question.
- **Semantic Search**: Improves search engines by identifying semantic equivalence between search queries and documents, surpassing simple keyword matching.
- **Text Summarization and Paraphrasing**: NLI can help reduce redundancy in summaries and validate the semantic equivalence of paraphrased texts.

### Project Structure:
- **Preprocessing Scripts**: Includes scripts for parsing sentence pairs and extracting semantic structures using SRL.
- **Model Training**: Contains scripts to train and evaluate both vanilla and graph-encoded transformer models on the MNLI dataset.
- **Results and Comparisons**: The project demonstrates how graph-based semantic encoding impacts NLI tasks, with detailed performance comparisons on matched and mismatched datasets.
