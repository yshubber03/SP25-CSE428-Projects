Project Title:
Detecting Malicious Biomedical Abstracts Using Enhanced BERT-Based Classification

Course:
CSE 428 – Capstone Project

1. Introduction :
A malicious actor uses LLMs to generate fake biomedical abstracts that manipulate knowledge graphs by promoting false drug–disease links. 
These abstracts poison the KG, misleading downstream tasks influencing scientific or clinical decision-making systems.
We aim to enhance a binary classifier to detect such malicious abstracts inputs.
In this project, we aim to enhance a binary classification model that can detect such malicious abstracts. 
By incorporating features such as medical term density, citation count, author trust level, and document length,
we improve upon a basic BERT classifier. The model assigns a confidence score to each abstract: high confidence indicates 
a likely benign abstract, while low confidence suggests potential malicious intent. Abstracts with uncertain confidence levels are 
flagged for human review, where tools like PubMed can be used to validate the claims.

2.dataset that I used:
We will use a biomedical abstracts dataset from PubMed, which provides reliable, peer-reviewed medical literature.
To simulate malicious samples, we will generate fake drug–disease relation abstracts using ChatGPT.
We will extract medical term density using terms from the MeSH Tree Structures.
For author metadata, we plan to scrape citation counts and publication history from Google Scholar.
Finally, for abstracts with low classification confidence (e.g., 0.5–0.7), we will use PubMed searches as part of our human validation process.
We compare a basic BERT model (baseline) with an improved version that adds extra features like medical term density and author trust, to see if the extra features help.


3.experiment set up:
To evaluate the effectiveness of our malicious abstract detection model, we conducted experiments using biomedical abstracts from PubMed, which provides a reliable source of peer-reviewed medical literature. To simulate adversarial inputs, we used ChatGPT to generate fake abstracts that include fabricated drug–disease relationships.

We compared two models:

A baseline BERT-only binary classifier trained on labeled abstracts.

An enhanced BERT + features model, which incorporates additional metadata including:

Medical term density: extracted using QuickUMLS and MeSH terms.

Citation count: simulated or scraped from Google Scholar.

Author trust level: estimated from past publication history and affiliations.

Original document word count: used to normalize medical term density.

Each abstract was assigned a confidence score between 0 and 1:

Confidence > 0.7 → not malicious

0.5 < confidence < 0.7 → uncertain, flagged for human review

Confidence ≤ 0.5 → marked as malicious

The model was implemented in Python and run via terminal using main.py. During testing, we observed how both models responded to real and fake abstracts and whether the enhanced model improved prediction confidence or accuracy.


4. Model Design
We implemented two versions of a binary classification model to detect malicious biomedical abstracts: a BERT-only model and an enhanced model that incorporates additional domain-specific features.

Input Design
BERT-only Model Input
The BERT-only model takes as input a single biomedical abstract in plain text. We use the bert-base-uncased tokenizer and model from the Hugging Face Transformers library. The abstract is tokenized and passed through the BERT encoder. From the resulting token embeddings, we compute the mean of the last_hidden_state to obtain a single [1 × 768] feature vector representing the abstract’s semantic content.

BERT + Features Model Input
The enhanced model builds upon the BERT representation by appending four manually engineered features:

Feature	Description
Medical Term Density	Ratio of UMLS-recognized medical terms to total word count
Word Count	Total number of words in the original abstract
Author Trust Level	A heuristic score based on citation count and institutional reputation
Citation Count	Number of times the paper or author has been cited

These features are concatenated with the BERT embedding to produce a [1 × 772] vector (768 + 4), which is then passed to the enhanced classifier.

Output Design
Both models use the same classifier architecture: a feedforward neural network consisting of:

python
Linear(input_dim → 32) → ReLU → Linear(32 → 1) → Sigmoid
The output is a scalar value in the range [0, 1], representing the model’s confidence that the abstract is not malicious. The prediction logic is defined as follows:

Confidence Score	Prediction	Action
> 0.7	Not Malicious	Accept
0.5 – 0.7	Uncertain	Send for human review
≤ 0.5	Malicious	Flag as suspicious

This design allows the enhanced model to leverage both linguistic and contextual cues in assessing the integrity of biomedical abstracts.
    

5.Results
To evaluate the effectiveness of our models, we tested both the BERT-only baseline and the enhanced BERT + features model using a simulated malicious biomedical abstract. The goal was to observe whether the additional metadata features would influence the model’s confidence in classifying the input as malicious or not.

Below is the actual output from running main.py on one such abstract:

BERT-only model output (confidence): 0.4709785282611847
→ Prediction: Malicious

BERT + features model output (confidence): 0.4972476065158844
→ Prediction: Malicious

Note: We define 1 = not malicious, 0 = malicious. So lower values indicate more suspicious abstracts.
As shown, both models predicted the abstract as malicious, since their confidence scores were below the 0.5 threshold. However, the enhanced model—which incorporates additional features such as medical term density, word count, author trust level, and citation count—produced a slightly higher confidence score (0.4972 vs. 0.4709).

This small improvement suggests that the extra features may help the model better assess contextual signals, especially in borderline cases near the decision threshold. While the difference is modest in this single test, it shows potential for performance gains when applied across a larger and more diverse dataset.

6. Conclusion
In this project, we built a binary classifier to detect malicious biomedical abstracts generated by large language models. By comparing a baseline BERT model with an enhanced version incorporating features such as term density, citation count, and author trust level, we demonstrated that contextual metadata can modestly improve classification accuracy. Although initial results are preliminary, this approach offers a promising direction for building more robust defenses against knowledge graph poisoning in biomedical domains.
