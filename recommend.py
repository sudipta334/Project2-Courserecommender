import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple

# Load the dataset
df = pd.read_csv("assignment2dataset.csv")

# Create a combined text field for better semantic search
df['text'] = df['title'] + ". " + df['description']

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate course embeddings
course_embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
faiss.normalize_L2(course_embeddings)

# Map course index to course_id
course_id_map = df['course_id'].tolist()

def recommend_courses(profile: str, completed_ids: List[str]) -> List[Tuple[str, float]]:
    # Filter out completed courses
    filtered_df = df[~df['course_id'].isin(completed_ids)].copy()
    filtered_texts = filtered_df['text'].tolist()

    # Compute and normalize embeddings for filtered courses
    filtered_embeddings = model.encode(filtered_texts, show_progress_bar=False)
    faiss.normalize_L2(filtered_embeddings)

    # Build FAISS index
    dim = filtered_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(filtered_embeddings)

    # Embed and normalize the user profile
    user_embedding = model.encode([profile])
    faiss.normalize_L2(user_embedding)

    # Search top-5 matches
    scores, indices = index.search(user_embedding, k=5)

    results = []
    for i in range(5):
        idx = indices[0][i]
        score = scores[0][i]
        course_id = filtered_df.iloc[idx]['course_id']
        results.append((course_id, round(float(score), 4)))

    return results

#  test cases
if __name__ == "__main__":
    test_cases = [
        ("I’ve completed the ‘Foundations of Machine Learning’ course and enjoy data visualization.", ['C001']),
        ("I know Azure basics and want to manage containers and build CI/CD pipelines.", ['C006']),
        ("My background is in ML fundamentals; I’d like to specialize in neural networks and production workflows.", ['C001']),
        ("I want to learn to build and deploy microservices with Kubernetes—what courses fit best?", []),
        ("I’m interested in blockchain and smart contracts but have no prior experience.", [])
    ]

    for profile, completed in test_cases:
        print(f"\nUser Query: {profile}")
        recommendations = recommend_courses(profile, completed)
        for course_id, score in recommendations:
            title = df[df['course_id'] == course_id]['title'].values[0]
            print(f"  - {course_id} ({score}): {title}")
