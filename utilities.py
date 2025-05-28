import fitz
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer    #### BERT model

import pymupdf
fitz = pymupdf
import io
import datetime,numbers


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(stream=pdf_path ,filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Load pre-trained Sentence-BERT model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def get_embedding(text):
    """Convert text to BERT embeddings."""
    return model.encode([text])[0]

### cosine similarity

def compute_similiarity(resume_text, jd_text):
  "this function checks cosine similiarty by using cosine function"
  resume_embedding = get_embedding(resume_text)
  jb_embedding = get_embedding(jd_text)
  similarity_score = cosine_similarity(
      [resume_embedding], [jb_embedding]
  ) [0][0]
  return similarity_score
