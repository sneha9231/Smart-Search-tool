import requests
from bs4 import BeautifulSoup
import pandas as pd
import gradio as gr
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Step 1: Scrape the free courses from Analytics Vidhya
url = "https://courses.analyticsvidhya.com/pages/all-free-courses"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

courses = []

# Extracting course title, image, and course link
for course_card in soup.find_all('header', class_='course-card__img-container'):
    img_tag = course_card.find('img', class_='course-card__img')
    
    if img_tag:
        title = img_tag.get('alt')
        image_url = img_tag.get('src')
        
        link_tag = course_card.find_previous('a')
        if link_tag:
            course_link = link_tag.get('href')
            if not course_link.startswith('http'):
                course_link = 'https://courses.analyticsvidhya.com' + course_link

            courses.append({
                'title': title,
                'image_url': image_url,
                'course_link': course_link
            })

# Step 2: Create DataFrame
df = pd.DataFrame(courses)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate embeddings using BERT
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Create embeddings for course titles
df['embedding'] = df['title'].apply(lambda x: get_bert_embedding(x))

# Function to perform search using BERT-based similarity
def search_courses(query):
    query_embedding = get_bert_embedding(query)
    course_embeddings = np.vstack(df['embedding'].values)
    
    # Compute cosine similarity between query embedding and course embeddings
    similarities = cosine_similarity(query_embedding, course_embeddings).flatten()
    
    # Add the similarity scores to the DataFrame
    df['score'] = similarities
    
    # Sort by similarity score in descending order and return top results
    top_results = df.sort_values(by='score', ascending=False).head(10)
    return top_results[['title', 'image_url', 'course_link', 'score']].to_dict(orient='records')

# Function to simulate autocomplete by updating search results live
def autocomplete(query):
    matching_courses = df[df['title'].str.contains(query, case=False, na=False)]
    return matching_courses['title'].tolist()[:3]  # Show top 3 matching course titles

def gradio_search(query):
    result_list = search_courses(query)
    
    if result_list:
        html_output = '<div class="results-container">'
        for item in result_list:
            course_title = item['title']
            course_image = item['image_url']
            course_link = item['course_link']
            relevance_score = round(item['score'] * 100, 2)
            
            html_output += f'''
            <div class="course-card">
                <img src="{course_image}" alt="{course_title}" class="course-image"/>
                <div class="course-info">
                    <h3>{course_title}</h3>
                    <p>Relevance: {relevance_score}%</p>
                    <a href="{course_link}" target="_blank" class="course-link">View Course</a>
                </div>
            </div>'''
        html_output += '</div>'
        return html_output
    else:
        return '<p class="no-results">No results found.</p>'

# Custom CSS for the Gradio interface
custom_css = """
body {
    font-family: Arial, sans-serif;
    background-color: #f0f2f5;
}
.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
.results-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
}
.course-card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    overflow: hidden;
    width: 48%;
    transition: transform 0.2s;
}
.course-card:hover {
    transform: translateY(-5px);
}
.course-image {
    width: 100%;
    height: 150px;
    object-fit: cover;
}
.course-info {
    padding: 15px;
}
.course-info h3 {
    margin-top: 0;
    font-size: 18px;
    color: #333;
}
.course-info p {
    color: #666;
    font-size: 14px;
    margin-bottom: 10px;
}
.course-link {
    display: inline-block;
    background-color: #007bff;
    color: white;
    padding: 8px 12px;
    text-decoration: none;
    border-radius: 4px;
    font-size: 14px;
    transition: background-color 0.2s;
}
.course-link:hover {
    background-color: #0056b3;
}
.no-results {
    text-align: center;
    color: #666;
    font-style: italic;
}
"""

# Gradio interface
iface = gr.Interface(
    fn=gradio_search,
    inputs=gr.Textbox(label="Enter your search query", placeholder="e.g., machine learning, data science, python"),
    outputs=gr.HTML(label="Search Results"),
    title="Analytics Vidhya Smart Course Search",
    description="Find the most relevant courses from Analytics Vidhya based on your query.",
    theme="huggingface",
    css=custom_css,
    examples=[
        ["machine learning for beginners"],
        ["advanced data visualization techniques"],
        ["python programming basics"], 
        ["Business Analytics"]
    ]
)

if __name__ == "__main__":
    iface.launch()
