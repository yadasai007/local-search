import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from extract import *

# Load the SBERT model
model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

def split_text(text, max_tokens=512):
    """Splits text into chunks of max_tokens."""
    tokens = model.tokenizer.encode(text, add_special_tokens=True)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [model.tokenizer.decode(chunk) for chunk in chunks]

def convert(text):
    """Convert text to SBERT embeddings with chunking if necessary."""
    tokens = model.tokenizer.encode(text, add_special_tokens=True)
    token_count = len(tokens)
    
    if token_count <= 512:
        return {None: model.encode(text, convert_to_numpy=True)}, token_count  # Store as single vector
    else:
        chunks = split_text(text)
        embeddings = {f"{i+1}": model.encode(chunk, convert_to_numpy=True) for i, chunk in enumerate(chunks)}
        return embeddings, token_count  # Store multiple vectors

def search_value(search_key):
    encoded_vector = {}
    file_path = "data.csv"
    pdf_folder_path = "data"

    if os.path.exists(file_path):
        print("CSV file found. Loading data...")
        df = pd.read_csv(file_path)
        pdf_files_text = extract_text_from_all_pdfs(pdf_folder_path)
        existing_files = df['File Name'].tolist()
        new_entries = []
        for file in pdf_files_text.keys():
            if not any(file in f for f in existing_files):  # Check for partial matches (chunked files)
                print("Added new:", file)
                file_embeddings, token_count = convert(pdf_files_text[file])
                
                for chunk_id, embedding in file_embeddings.items():
                    chunk_name = f"{file}-{chunk_id}" if chunk_id else file  # Add chunk ID if applicable
                    encoded_vector[chunk_name] = embedding
                    new_entries.append((chunk_name, token_count, ",".join(map(str, embedding))))
            else:
                filtered_df = df[df["File Name"].str.contains(file, na=False)]
                file_embedding_dict = {
                    file: np.array(embedding.split(","), dtype=np.float64) 
                    for file, embedding in zip(filtered_df["File Name"], filtered_df["Embeddings"])
                }
                encoded_vector.update(file_embedding_dict)
        if new_entries:
            new_df = pd.DataFrame(new_entries, columns=['File Name', 'Token Count', 'Embeddings'])
            df = pd.concat([df, new_df], ignore_index=True)
            
            df.to_csv(file_path, index=False)
            print("CSV updated with new files.")
    else:
        print("CSV file not found. Creating a new one...")
        pdf_files_text = extract_text_from_all_pdfs(pdf_folder_path)

        for file in pdf_files_text.keys():
            file_embeddings, token_count = convert(pdf_files_text[file])
            
            for chunk_id, embedding in file_embeddings.items():
                chunk_name = f"{file}-{chunk_id}" if chunk_id else file
                encoded_vector[chunk_name] = embedding

        df = pd.DataFrame({
            'File Name': encoded_vector.keys(),
            'Embeddings': [",".join(map(str, v)) for v in encoded_vector.values()]
        })
        df.to_csv(file_path, index=False)

    # Convert search query to vector
    search_vector = model.encode(search_key, convert_to_numpy=True)
    scores = {}
    for file, vector in encoded_vector.items():
        # Compute cosine similarity
        similarity = np.dot(vector, search_vector) / (np.linalg.norm(vector) * np.linalg.norm(search_vector))
        scores[file] = similarity

    # Sort and get top 5 results
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
    results = {file: [score, pdf_files_text[file.split("-")[0]]] for file, score in sorted_scores}  # Use base filename
    
    return results
