#second try to get the answer
import pickle
from flask import Flask, render_template, request
import numpy as np
import heapq


app = Flask(__name__)

# Load Data
popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
book_similarity_cache = pickle.load(open('book_similarity_cache.pkl', 'rb'))

# Trie for Book Search
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return self._collect_words(node, prefix)

    def _collect_words(self, node, prefix):
        words = []
        if node.is_end_of_word:
            words.append(prefix)
        for char, child in node.children.items():
            words.extend(self._collect_words(child, prefix + char))
        return words

recommendation_cache = {}

def get_cached_recommendations(user_input):
    if user_input in recommendation_cache:
        return recommendation_cache[user_input]  # Return from cache if exists
    
    if user_input not in pt.index:
        return []
    
    recommendations = []
    for similar_book in book_graph.get(user_input, []):
        book_info = books[books["Book-Title"] == similar_book]
        if not book_info.empty:
            author = book_info.iloc[0]["Book-Author"]
            image_url = book_info.iloc[0]["Image-URL-M"]
            recommendations.append({
                "book_title": similar_book,
                "author": author,
                "image_url": image_url
            })
    
    recommendation_cache[user_input] = recommendations  # Store in cache
    return recommendations

# Cosine Similarity Function
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

# Build Book Graph
book_graph = {}


def build_book_graph():
    global book_graph

    # Ensure all books are initialized in the graph
    for book_title in pt.index:
        book_graph[book_title] = set()

    for book_title in pt.index:
        input_vector = pt.loc[book_title].values
        heap = []
        
        for other_book in pt.index:
            if other_book == book_title:
                continue
            other_vector = pt.loc[other_book].values
            similarity = cosine_similarity(input_vector, other_vector)
            heapq.heappush(heap, (-similarity, other_book))

        # Get top 10 similar books
        for _ in range(min(10, len(heap))):
            _, similar_book = heapq.heappop(heap)
            
            # Ensure similar_book exists in book_graph before adding relationships
            if similar_book not in book_graph:
                book_graph[similar_book] = set()
                
            book_graph[book_title].add(similar_book)
            book_graph[similar_book].add(book_title)  # Bidirectional link


# Insert Books into Trie for Prefix Search
book_trie = Trie()
for book_title in pt.index:
    book_trie.insert(book_title)

# Build Graph at Startup
build_book_graph()

@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_rating'].values),
                           rating=list(popular_df['avg_ratings'].values))

@app.route('/recommend', methods=['GET', 'POST'])
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend_books():
    user_input = request.form.get('user_input').strip()

    if user_input not in pt.index:
        return render_template('search.html')

    recommendations = []

    # Use cached similarity scores if available
    if user_input in book_similarity_cache:
        distances, indices = book_similarity_cache[user_input]
        sorted_indices = np.argsort(distances)[::-1]  # Sort indices in descending order of similarity

        for idx in sorted_indices[:20]:  # Get top 20 books
            book_title = pt.index[indices[idx]]
            book_info = books[books["Book-Title"] == str(book_title)]

            if not book_info.empty:
                author = book_info.iloc[0]["Book-Author"]
                image_url = book_info.iloc[0]["Image-URL-M"]

                recommendations.append({
                    "book_title": book_title,
                    "similarity": round(distances[idx], 2),
                    "author": author,
                    "image_url": image_url
                })

    else:
        # Calculate cosine similarity manually and update cache
        input_vector = pt.loc[user_input].values
        heap = []

        for book_title in pt.index:
            if book_title == user_input:
                continue  # Skip self
            book_vector = pt.loc[book_title].values
            similarity = cosine_similarity(input_vector, book_vector)

            heapq.heappush(heap, (-similarity, book_title))  # Store negative similarity to make max-heap

        # Extract top 20 books with highest similarity
        for _ in range(min(20, len(heap))):
            similarity, book_title = heapq.heappop(heap)  # Pop most similar books
            similarity = -similarity  # Convert back to positive similarity

            book_info = books[books["Book-Title"] == book_title]
            if not book_info.empty:
                author = book_info.iloc[0]["Book-Author"]
                image_url = book_info.iloc[0]["Image-URL-M"]

                recommendations.append({
                    "book_title": book_title,
                    "similarity": round(similarity, 2),
                    "author": author,
                    "image_url": image_url
                })

        # Update book_similarity_cache properly
        distances = [rec["similarity"] for rec in recommendations]
        indices = [pt.index.get_loc(rec["book_title"]) for rec in recommendations if rec["book_title"] in pt.index]

        book_similarity_cache[user_input] = (distances, indices)  # Store results in cache
    print(recommendations)
    return render_template('recommend.html', recommendations=recommendations)



@app.route('/search_books', methods=['GET', 'POST'])
def search_books():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        matches = book_trie.search_prefix(query)
        return render_template('search.html', matches=matches)
    return render_template('search.html', matches=None)

if __name__ == '__main__':
    app.run(debug=True)
