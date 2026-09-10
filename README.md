# Intelligent Wikipedia Question Answering Assistant

An NLP-based question answering system built using the Stanford Question Answering Dataset (SQuAD v1.1) derived from Wikipedia articles.

## Project Overview

The system provides two approaches to question answering:

- **QA-Based Question Answering** – retrieves relevant answers and supporting Wikipedia passages from the SQuAD knowledge base.
- **Knowledge-Based Question Answering** – represents extracted information as Subject → Relation → Object knowledge triples.

The project also includes a **Wikipedia Explorer** for searching relevant passages.

## Features

- SQuAD v1.1 dataset integration
- Question answering
- Knowledge-based question answering
- Subject–Relation–Object representation
- Wikipedia passage search
- Relevance scoring
- Supporting context display
- Interactive Streamlit interface

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Natural Language Processing
- SQuAD v1.1
- Wikipedia-derived text

## Project Structure

```text
Intelligent_QA_Wikipedia_SQuAD_COMPLETE/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── train-v1.1.json
│   └── dev-v1.1.json
│
├── modules/
│   ├── __init__.py
│   └── qa_system.py
│
└── evaluation/
    └── README.md