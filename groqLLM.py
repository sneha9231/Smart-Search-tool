import requests
from bs4 import BeautifulSoup
import pandas as pd
import gradio as gr
from groq import Groq

#Scrape the free courses from Analytics Vidhya
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

#Create DataFrame
df = pd.DataFrame(courses)

client.api_key = "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"

def search_courses(query):
    try:
        print(f"Searching for: {query}")
        print(f"Number of courses in database: {len(df)}")

        # Prepare the prompt for Groq
        prompt = f"""Given the following query: "{query}"
        Please analyze the query and rank the following courses based on their relevance to the query. 
        Prioritize courses from Analytics Vidhya. Provide a relevance score from 0 to 1 for each course.
        Only return courses with a relevance score of 0.5 or higher.
        Return the results in the following format:
        Title: [Course Title]
        Relevance: [Score]
        
        Courses:
        {df['title'].to_string(index=False)}
        """

        print("Sending request to Groq...")
        # Get response from Groq
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in course recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        print("Received response from Groq")

        # Parse Groq's response
        results = []
        print("Groq response content:")
        print(response.choices[0].message.content)
        
        for line in response.choices[0].message.content.split('\n'):
            if line.startswith('Title:'):
                title = line.split('Title:')[1].strip()
                print(f"Found title: {title}")
            elif line.startswith('Relevance:'):
                relevance = float(line.split('Relevance:')[1].strip())
                print(f"Relevance for {title}: {relevance}")
                if relevance >= 0.5:
                    matching_courses = df[df['title'] == title]
                    if not matching_courses.empty:
                        course = matching_courses.iloc[0]
                        results.append({
                            'title': title,
                            'image_url': course['image_url'],
                            'course_link': course['course_link'],
                            'score': relevance
                        })
                        print(f"Added course: {title}")
                    else:
                        print(f"Warning: Course not found in database: {title}")

        print(f"Number of results found: {len(results)}")
        return sorted(results, key=lambda x: x['score'], reverse=True)[:5]  # Return top 5 results

    except Exception as e:
        print(f"An error occurred in search_courses: {str(e)}")
        return []

def gradio_search(query):
    result_list = search_courses(query)
    
    if result_list:
        html_output = ""
        for item in result_list:
            course_title = item['title']
            course_image = item['image_url']
            course_link = item['course_link']
            
            html_output += f'''
            <div style="margin-bottom: 20px;">
                <img src="{course_image}" alt="{course_title}" style="width:200px;"/><br>
                <a href="{course_link}" target="_blank">{course_title}</a>
            </div>'''
        return html_output
    else:
        return "<p>No results found. Please try a different query.</p>"

# Gradio interface
iface = gr.Interface(
    fn=gradio_search,
    inputs=gr.Textbox(label="Enter your query"),
    outputs=gr.HTML(label="Search Results"),
    title="Analytics Vidhya Smart Search"
)

if __name__ == "__main__":
    print("Starting Gradio interface...")
    iface.launch()
