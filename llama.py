import os
import sys
from transformers import LlamaForCausalLM, LlamaTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import glob

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

def load_documents(directory):
    documents = []
    file_paths = glob.glob(f"{directory}/*.txt")
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())
    return documents, file_paths


def vectorize_documents(documents):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    return vectorizer, doc_vectors

def find_similar_document(query, vectorizer, doc_vectors):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    most_similar_doc_index = similarities.argmax()
    return most_similar_doc_index

documents, file_paths = load_documents("/home/ubuntu/Project_Files/Finetune/Data")
print(f"Loaded {len(documents)} documents") 
vectorizer, doc_vectors = vectorize_documents(documents)

# Chat history for context
chat_history = []

query = None if len(sys.argv) <= 1 else sys.argv[1]

while True:
    # Prompt for input if query is None
    if query is None:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()

    # Find the most similar document
    similar_doc_index = find_similar_document(query, vectorizer, doc_vectors)
    similar_document = documents[similar_doc_index]
    print(f"Retrieved Document: {file_paths[similar_doc_index]}")

    # Generate a response using LLaMA 2
    input_ids = tokenizer.encode(query + tokenizer.eos_token + similar_document, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=512, num_return_sequences=1)[0]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    print(f"Response: {response}")

    chat_history.append((query, response))
    query = None  


# import os
# import sys
# import glob
# import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# model_name = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = LlamaTokenizer.from_pretrained(model_name)

# # Ensure CUDA is available and then parallelize the model across available GPUs
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     model = LlamaForCausalLM.from_pretrained(model_name)
#     model = torch.nn.DataParallel(model)  # This will parallelize the model across all available GPUs
#     model.to(device)
# else:
#     raise EnvironmentError("CUDA is not available or PyTorch is not configured with CUDA support.")

# def load_documents(directory):
#     documents = []
#     file_paths = glob.glob(f"{directory}/*.txt")
#     for file_path in file_paths:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             documents.append(file.read())
#     return documents, file_paths

# def vectorize_documents(documents):
#     vectorizer = TfidfVectorizer()
#     doc_vectors = vectorizer.fit_transform(documents)
#     return vectorizer, doc_vectors

# def find_similar_document(query, vectorizer, doc_vectors):
#     query_vector = vectorizer.transform([query])
#     similarities = cosine_similarity(query_vector, doc_vectors).flatten()
#     most_similar_doc_index = similarities.argmax()
#     return most_similar_doc_index

# documents, file_paths = load_documents("/home/ubuntu/Project_Files/Finetune/Data")
# print(f"Loaded {len(documents)} documents") 
# vectorizer, doc_vectors = vectorize_documents(documents)

# chat_history = []

# query = None if len(sys.argv) <= 1 else sys.argv[1]

# while True:
#     if query is None:
#         query = input("Prompt: ")
#     if query in ['quit', 'q', 'exit']:
#         sys.exit()

#     similar_doc_index = find_similar_document(query, vectorizer, doc_vectors)
#     similar_document = documents[similar_doc_index]
#     print(f"Retrieved Document: {file_paths[similar_doc_index]}")

#     input_ids = tokenizer.encode(query + tokenizer.eos_token + similar_document, return_tensors="pt").to(device)
#     with torch.no_grad():
#         output_ids = model.module.generate(input_ids, max_length=512, num_return_sequences=1)[0]
#     response = tokenizer.decode(output_ids, skip_special_tokens=True)

#     print(f"Response: {response}")

#     chat_history.append((query, response))
#     query = None











# import os
# import sys
# import glob
# import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'

# model_name = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = LlamaTokenizer.from_pretrained(model_name)

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     model = LlamaForCausalLM.from_pretrained(model_name)
#     model = torch.nn.DataParallel(model).to(device)
# else:
#     raise EnvironmentError("CUDA is not available, or PyTorch is not configured with CUDA support.")

# def load_documents(directory):
#     documents = []
#     file_paths = glob.glob(f"{directory}/*.txt")
#     for file_path in file_paths:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             documents.append(file.read())
#     return documents, file_paths

# def vectorize_documents(documents):
#     vectorizer = TfidfVectorizer()
#     doc_vectors = vectorizer.fit_transform(documents)
#     return vectorizer, doc_vectors

# def find_similar_document(query, vectorizer, doc_vectors):
#     query_vector = vectorizer.transform([query])
#     similarities = cosine_similarity(query_vector, doc_vectors).flatten()
#     most_similar_doc_index = similarities.argmax()
#     return most_similar_doc_index

# documents, file_paths = load_documents("/path/to/your/documents")
# vectorizer, doc_vectors = vectorize_documents(documents)
# chat_history = []
# query = None if len(sys.argv) <= 1 else sys.argv[1]

# while True:
#     if query is None:
#         query = input("Prompt: ")
#     if query in ['quit', 'q', 'exit']:
#         sys.exit()

#     similar_doc_index = find_similar_document(query, vectorizer, doc_vectors)
#     similar_document = documents[similar_doc_index]
#     print(f"Retrieved Document: {file_paths[similar_doc_index]}")
#     input_ids = tokenizer.encode(query + tokenizer.eos_token + similar_document, return_tensors="pt")
#     input_ids = input_ids.to(device)
    
#     with torch.no_grad():
#         output_ids = model.module.generate(input_ids, max_length=512, num_return_sequences=1)[0]

#     response = tokenizer.decode(output_ids, skip_special_tokens=True)
#     print(f"Response: {response}")

#     chat_history.append((query, response))
#     query = None
#     torch.cuda.empty_cache()
