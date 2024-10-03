# Smart-Search-tool
# Goal- To create a smart search tool that allows users to find relevant free courses on Analytics Vidhya’s platform.
Here's how I approached the assignment:
Data Collection: I began by scraping the free courses' titles and relevant metadata such as course links and images from Analytics Vidhya’s platform. using beautifulsoup

Model Selection: Initially, I experimented with the Groq API for generating embeddings and conducting searches. However, I found that the results were not as appropriate, prompting me to switch to a more refined solution using BERT (Bidirectional Encoder Representations from Transformers).

I used the pre-trained BERT model (bert-base-uncased) to generate embeddings for both the user query and the course titles.

Relevance: To match user queries with the most relevant courses, I computed cosine similarity between the user’s query embedding and the course title embeddings. This similarity score helped rank the courses based on relevance.

Real-Time Autocomplete Feature: I implemented a real-time autocomplete feature that suggests the top 3 course titles based on the user's input. It uses string matching to enhance the user experience.

Gradio Interface: I designed and deployed the user interface using Gradio and implemented a clean, user-friendly layout for presenting the course search results. The interface dynamically displays relevant course details, such as the title, image, link, and a relevance score.

Deployment on Huggingface Spaces: I deployed the search tool on Huggingface Spaces, making it publicly accessible for review. The interface was customized using CSS for better visual appeal and responsiveness.


Groq model: https://huggingface.co/spaces/snhpndy/search

Bert model: https://huggingface.co/spaces/snhpndy/smartSearch
