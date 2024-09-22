import requests
from bs4 import BeautifulSoup
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import gradio as gr

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

# Step 2: Create DataFrame and load model
df = pd.DataFrame(courses)

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for course titles
course_titles = df['title'].tolist()
course_embeddings = model.encode(course_titles, convert_to_tensor=True)

# Step 3: Search function to return relevant courses
def search_courses(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarity between the query and all course titles
    similarities = util.pytorch_cos_sim(query_embedding, course_embeddings)[0]
    
    # Get top relevant courses based on similarity scores
    top_results = similarities.topk(k=5)  # Get top 5 results

    # Use a set to track unique course titles
    seen_courses = set()
    results = []

    for idx in top_results.indices:
        course = df.iloc[idx.item()]  # Ensure index is converted properly
        
        if course['title'] not in seen_courses:  # Check if the course has already been added
            results.append({
                'title': course['title'],
                'image_url': course['image_url'],
                'course_link': course['course_link'],
                'score': similarities[idx].item()
            })
            seen_courses.add(course['title'])  # Mark this course as seen

    return sorted(results, key=lambda x: x['score'], reverse=True)


# Step 4: Gradio interface to display search results with clickable course links
def gradio_search(query):
    result_list = search_courses(query)
    
    if result_list:
        html_output = ""
        for item in result_list:
            course_title = item['title']
            course_image = item['image_url']
            course_link = item['course_link']
            
            # Create HTML for each result with image, title, and clickable hyperlink to the course
            html_output += f'''
            <div style="margin-bottom: 20px;">
                <img src="{course_image}" alt="{course_title}" style="width:200px;"/><br>
                <a href="{course_link}" target="_blank">{course_title}</a>
            </div>'''
        return html_output
    else:
        return "<p>No results found.</p>"

# Step 5: Create Gradio interface
gr.Interface(
    fn=gradio_search,
    inputs=gr.Textbox(label="Enter your query"),
    outputs=gr.HTML(label="Search Results"),  # Add label for better clarity
    title="Analytics Vidhya Smart Search"
).launch()
