from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords') 

app = Flask(__name__)

# Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Initialize the TF-IDF vectorizer
stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = vectorizer.fit_transform(documents)

# Apply Truncated SVD for dimensionality reduction (LSA)
n_components = 100
lsa = TruncatedSVD(n_components=n_components, random_state=42)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# Normalize the LSA matrix
lsa_matrix = lsa_matrix / np.sqrt(np.sum(lsa_matrix**2, axis=1, keepdims=True))

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # Transform the query using the same vectorizer and LSA model
    query_tfidf = vectorizer.transform([query])
    query_lsa = lsa.transform(query_tfidf)
    
    # Normalize the query vector
    query_lsa = query_lsa / np.sqrt(np.sum(query_lsa**2))
    
    # Compute cosine similarity between the query and all documents
    similarities = cosine_similarity(query_lsa, lsa_matrix)[0]
    
    # Get the top 5 most similar documents
    top_indices = np.argsort(similarities)[::-1][:5]
    top_similarities = similarities[top_indices]
    top_documents = [documents[i] for i in top_indices]
    
    # Round similarities for better display
    top_similarities = [round(float(sim), 4) for sim in top_similarities]
    top_indices = [int(i) for i in top_indices]
    
    return top_documents, top_similarities, top_indices


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
