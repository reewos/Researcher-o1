import os
import io
import requests
import streamlit as st
from llama_index.llms.openai import OpenAI
import arxiv
import PyPDF2

from dotenv import load_dotenv
load_dotenv()

# Model configuration
gpt_4o_mini = OpenAI(model="gpt-4o-mini", api_key=os.environ['AIML_API_KEY'], api_base="https://api.aimlapi.com")
gpt_o1_mini = OpenAI(model="o1-mini", api_key=os.environ['AIML_API_KEY'], api_base="https://api.aimlapi.com")

if "results_arxiv" not in st.session_state:
    st.session_state.results_arxiv = []
if "pdf_analysis" not in st.session_state:
    st.session_state.pdf_analysis = ""
if "experiment" not in st.session_state:
    st.session_state.experiment = ""

#############
# Functions #
#############

def search_arxiv(query, max_results=5):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for result in search.results():
        results.append({
            'title': result.title,
            'summary': result.summary,
            'authors': ', '.join(author.name for author in result.authors),
            'url': result.pdf_url
        })
    return results

def analyze_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Use gpt-4o-mini for simple analysis
    analysis = gpt_4o_mini.complete(f"Analyze the following text and provide a summary of key findings: {text[:4000]}")  # Limit to 4000 characters for simplicity
    st.session_state.pdf_analysis = analysis.text
    return analysis.text

def create_hypothetical_experiment(topic):
    # Use gpt-o1-mini for complex tasks
    experiment = gpt_o1_mini.complete(f"Create a hypothetical experiment on the following topic: {topic}. Include hypothesis, methodology, and possible outcomes.", max_tokens=6000)
    st.session_state.experiment = experiment.text
    return experiment.text

def main():
    st.set_page_config(page_title="Reasoning-lab")

    st.title("Reasoning-lab: Explore, experiment and find scientific solutions")

    menu = ["Search Articles", "PDF Analysis", "Hypothetical Experiments"]
    choice = st.sidebar.selectbox("Select a function", menu)

    if choice == "Search Articles":
        st.subheader("Search Scientific Articles")
        query = st.text_input("Enter your search query", key='query_search')
        if st.button("Search"):
            results = search_arxiv(query)
            st.session_state.results_arxiv = results
        if st.session_state.results_arxiv:
            results = st.session_state.results_arxiv
            for result in results:
                st.write(f"**{result['title']}**")
                st.write(f"Authors: {result['authors']}")
                st.write(f"Summary: {result['summary'][:200]}...")
                st.write(f"[PDF Link]({result['url']})")
                st.write("---")

    elif choice == "PDF Analysis":
        st.subheader("PDF Analysis")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file is not None:
            if st.session_state.pdf_analysis == "":
                analysis = analyze_pdf(uploaded_file)
            else:
                analysis = st.session_state.pdf_analysis
            st.write("PDF Analysis:")
            st.write(analysis)

    elif choice == "Hypothetical Experiments":
        st.subheader("Create Hypothetical Experiments")
        topic = st.text_input("Enter a topic for the hypothetical experiment")
        if st.button("Generate Experiment"):
            if st.session_state.experiment == "":
                experiment = create_hypothetical_experiment(topic)
            else:
                experiment = st.session_state.experiment
            st.write("Hypothetical Experiment:")
            st.write(experiment)

if __name__ == "__main__":
    main()
