import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import os
import uuid
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import requests
from openai import OpenAI
import logging
from PIL import Image
from io import BytesIO
import base64
import xml.etree.ElementTree as ET
import requests
import time
from urllib.parse import quote_plus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DeepSeek API setup
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-cb3e2fbf705d4c1899a7cf53c49fbaa6")  # Replace with your valid key
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# Set page configuration
st.set_page_config(
    page_title="Drug Discovery Research Playground",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .section-header {
        color: #1E6091;
        font-weight: 600;
        padding-bottom: 10px;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f3ff;
        border-left: 5px solid #2196F3;
    }
    .chat-message.assistant {
        background-color: #f0f0f0;
        border-left: 5px solid #757575;
    }
    .chat-message .message-content {
        margin-top: 0.5rem;
    }
    .chat-message .message-header {
        font-weight: bold;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .timestamp {
        color: #888;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }
    .research-card {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .nav-item {
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.3rem;
    }
    .nav-item:hover {
        background-color: #f0f0f0;
    }
    .nav-item.active {
        background-color: #e6f3ff;
        font-weight: bold;
    }
    .project-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E6091;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------ Data Models ------------------------------------

class Message:
    def __init__(self, role: str, content: str):
        self.id = str(uuid.uuid4())
        self.role = role
        self.content = content
        self.timestamp = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        message = cls(data['role'], data['content'])
        message.id = data['id']
        message.timestamp = datetime.datetime.fromisoformat(data['timestamp'])
        return message

class SearchQuery:
    def __init__(self, query: str, results: Optional[List[Dict[str, Any]]] = None):
        self.id = str(uuid.uuid4())
        self.query = query
        self.results = results or []
        self.timestamp = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'query': self.query,
            'results': self.results,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchQuery':
        search_query = cls(data['query'], data['results'])
        search_query.id = data['id']
        search_query.timestamp = datetime.datetime.fromisoformat(data['timestamp'])
        return search_query

class Project:
    def __init__(self, name: str, description: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description or ""
        self.messages: List[Message] = []
        self.searches: List[SearchQuery] = []
        self.created_at = datetime.datetime.now()
        self.updated_at = datetime.datetime.now()
    
    def add_message(self, role: str, content: str) -> Message:
        message = Message(role, content)
        self.messages.append(message)
        self.updated_at = datetime.datetime.now()
        return message
    
    def add_search(self, query: str, results: Optional[List[Dict[str, Any]]] = None) -> SearchQuery:
        search = SearchQuery(query, results)
        self.searches.append(search)
        self.updated_at = datetime.datetime.now()
        return search
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'messages': [msg.to_dict() for msg in self.messages],
            'searches': [search.to_dict() for search in self.searches],
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        project = cls(data['name'], data['description'])
        project.id = data['id']
        project.messages = [Message.from_dict(msg) for msg in data['messages']]
        project.searches = [SearchQuery.from_dict(search) for search in data['searches']]
        project.created_at = datetime.datetime.fromisoformat(data['created_at'])
        project.updated_at = datetime.datetime.fromisoformat(data['updated_at'])
        return project

class ProjectManager:
    def __init__(self, storage_path: str = "projects"):
        self.storage_path = storage_path
        self.projects: Dict[str, Project] = {}
        self._ensure_storage_exists()
        self._load_projects()
    
    def _ensure_storage_exists(self):
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
    
    def _load_projects(self):
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.storage_path, filename), 'r') as f:
                        project_data = json.load(f)
                        project = Project.from_dict(project_data)
                        self.projects[project.id] = project
                except Exception as e:
                    logger.error(f"Error loading project {filename}: {str(e)}")
    
    def save_project(self, project: Project):
        self.projects[project.id] = project
        try:
            with open(os.path.join(self.storage_path, f"{project.id}.json"), 'w') as f:
                json.dump(project.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving project {project.id}: {str(e)}")
    
    def create_project(self, name: str, description: Optional[str] = None) -> Project:
        project = Project(name, description)
        self.save_project(project)
        return project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        return self.projects.get(project_id)
    
    def delete_project(self, project_id: str) -> bool:
        if project_id in self.projects:
            del self.projects[project_id]
            try:
                os.remove(os.path.join(self.storage_path, f"{project_id}.json"))
                return True
            except Exception as e:
                logger.error(f"Error deleting project {project_id}: {str(e)}")
        return False
    
    def get_all_projects(self) -> List[Project]:
        return list(self.projects.values())

# ------------------------------------ AI Functions ------------------------------------

def analyze_with_deepseek(messages: List[Dict[str, str]]) -> str:
    """Call DeepSeek API using OpenAI client."""
    try:
        logger.info(f"Sending messages to DeepSeek API: {messages}")
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "system", "content": "You are a pharmaceutical research assistant with expertise in medicinal chemistry, drug discovery, and molecular property analysis. Provide detailed, scientifically-grounded analyses focusing on structure-property relationships and actionable insights for drug development."}] + messages,
            max_tokens=5000,
            temperature=0.3
        )
        logger.info(f"Received response from DeepSeek API: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"Error contacting DeepSeek API: {str(e)}"
        logger.error(error_msg)
        return f"Failed to generate analysis due to API error: {str(e)}. Please try again later or check your API key."

def get_deepseek_response(prompt: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """Get response from DeepSeek with conversation history, ensuring interleaved roles."""
    messages = conversation_history or []
    
    # Filter to ensure no consecutive roles
    filtered_messages = []
    last_role = None
    for msg in messages:
        if msg["role"] != last_role:
            filtered_messages.append(msg)
            last_role = msg["role"]
    
    # Ensure the last message before the new prompt is "assistant" (if not empty)
    if filtered_messages and filtered_messages[-1]["role"] == "user":
        filtered_messages.append({"role": "assistant", "content": "Understood, please continue."})
    
    # Append the new user prompt
    filtered_messages.append({"role": "user", "content": prompt})
    
    return analyze_with_deepseek(filtered_messages)


def search_pubmed(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search for publications on PubMed using NCBI E-utilities.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of publication dictionaries with metadata
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # Step 1: Search for IDs matching the query
    search_url = f"{base_url}esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json"
    }
    
    try:
        search_response = requests.get(search_url, params=search_params)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        # Extract the IDs
        pmids = search_data["esearchresult"]["idlist"]
        
        if not pmids:
            return []
        
        # Step 2: Fetch details for each ID
        pubmed_results = []
        
        # Fetch details in batches of 5 to avoid rate limits
        batch_size = 5
        for i in range(0, len(pmids), batch_size):
            batch_ids = pmids[i:i+batch_size]
            
            fetch_url = f"{base_url}efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(batch_ids),
                "retmode": "xml"
            }
            
            fetch_response = requests.get(fetch_url, params=fetch_params)
            fetch_response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(fetch_response.content)
            
            # Process each article
            for article in root.findall(".//PubmedArticle"):
                try:
                    # Extract article metadata
                    article_data = extract_article_metadata(article)
                    if article_data:
                        pubmed_results.append(article_data)
                except Exception as e:
                    logger.error(f"Error parsing article: {str(e)}")
            
            # Respect API rate limits
            time.sleep(0.35)
        
        return pubmed_results
        
    except Exception as e:
        logger.error(f"Error searching PubMed: {str(e)}")
        return []

def extract_article_metadata(article_element) -> Dict[str, Any]:
    """Extract metadata from a PubMed article XML element."""
    try:
        # Extract PMID
        pmid = article_element.find(".//PMID").text
        
        # Extract article title
        title_element = article_element.find(".//ArticleTitle")
        title = title_element.text if title_element is not None else "No title available"
        
        # Extract journal information
        journal_element = article_element.find(".//Journal/Title")
        journal = journal_element.text if journal_element is not None else "Unknown Journal"
        
        # Extract publication year
        year_element = article_element.find(".//PubDate/Year")
        if year_element is None:
            year_element = article_element.find(".//PubDate/MedlineDate")
        year = year_element.text[:4] if year_element is not None else "Unknown Year"
        
        # Extract authors
        author_elements = article_element.findall(".//Author")
        authors = []
        for author in author_elements:
            last_name = author.find("LastName")
            initials = author.find("Initials")
            if last_name is not None and initials is not None:
                authors.append(f"{last_name.text} {initials.text}")
            elif last_name is not None:
                authors.append(last_name.text)
        
        author_string = ", ".join(authors) if authors else "Unknown Authors"
        
        # Extract abstract
        abstract_elements = article_element.findall(".//AbstractText")
        abstract_texts = []
        for abstract_element in abstract_elements:
            if abstract_element.text:
                abstract_texts.append(abstract_element.text)
        
        abstract = " ".join(abstract_texts) if abstract_texts else "Abstract not available"
        
        # Extract DOI if available
        doi_element = article_element.find(".//ArticleId[@IdType='doi']")
        doi = doi_element.text if doi_element is not None else f"PMID: {pmid}"
        
        return {
            "title": title,
            "authors": author_string,
            "journal": journal,
            "year": year,
            "doi": doi,
            "pmid": pmid,
            "abstract": abstract,
            "source": "PubMed"
        }
    except Exception as e:
        logger.error(f"Error extracting article metadata: {str(e)}")
        return None

def search_europe_pmc(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search for publications on Europe PMC.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of publication dictionaries with metadata
    """
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    
    params = {
        "query": query,
        "format": "json",
        "pageSize": max_results,
        "resultType": "core"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for result in data.get("resultList", {}).get("result", []):
            authors = result.get("authorString", "Unknown Authors")
            
            # Extract year from publication date
            pub_date = result.get("firstPublicationDate", "")
            year = pub_date[:4] if pub_date else "Unknown Year"
            
            publication = {
                "title": result.get("title", "No title available"),
                "authors": authors,
                "journal": result.get("journalTitle", "Unknown Journal"),
                "year": year,
                "doi": result.get("doi", f"PMID: {result.get('pmid', 'Unknown')}"),
                "pmid": result.get("pmid", ""),
                "abstract": result.get("abstractText", "Abstract not available"),
                "source": "Europe PMC"
            }
            results.append(publication)
        
        return results
    
    except Exception as e:
        logger.error(f"Error searching Europe PMC: {str(e)}")
        return []

def search_publications(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search for publications using multiple free APIs.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return per source
        
    Returns:
        List of publication dictionaries with metadata
    """
    results = []
    
    # Search PubMed
    try:
        pubmed_results = search_pubmed(query, max_results=max_results)
        results.extend(pubmed_results)
    except Exception as e:
        logger.error(f"PubMed search error: {str(e)}")
    
    # If PubMed fails or returns few results, try Europe PMC
    if len(results) < max_results:
        try:
            remaining = max_results - len(results)
            europe_pmc_results = search_europe_pmc(query, max_results=remaining)
            
            # Filter out duplicates based on DOI or PMID
            seen_ids = {r.get("doi", ""): True for r in results}
            seen_ids.update({r.get("pmid", ""): True for r in results if r.get("pmid")})
            
            for result in europe_pmc_results:
                if result.get("doi") not in seen_ids and result.get("pmid", "") not in seen_ids:
                    results.append(result)
        except Exception as e:
            logger.error(f"Europe PMC search error: {str(e)}")
    
    return results[:max_results]

# ------------------------------------ Streamlit App ------------------------------------

def init_session_state():
    """Initialize session state variables."""
    if "project_manager" not in st.session_state:
        st.session_state.project_manager = ProjectManager()
    
    if "current_project_id" not in st.session_state:
        st.session_state.current_project_id = None
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    
    if "page" not in st.session_state:
        st.session_state.page = "chat"

def render_sidebar():
    """Render the sidebar with project navigation and management."""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/logo/master/streamlit-mark-color.png", width=100)
        st.title("Research Playground")
        
        st.markdown("---")
        
        st.subheader("Projects")
        
        if st.button("➕ New Project"):
            st.session_state.page = "new_project"
        
        projects = st.session_state.project_manager.get_all_projects()
        projects.sort(key=lambda p: p.updated_at, reverse=True)
        
        for project in projects:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(f"{project.name}", key=f"project_{project.id}", help=project.description):
                    st.session_state.current_project_id = project.id
                    st.session_state.page = "chat"
                    current_project = st.session_state.project_manager.get_project(project.id)
                    if current_project:
                        st.session_state.chat_messages = current_project.messages
                        st.session_state.search_history = current_project.searches
            
            with col2:
                if st.button("🗑️", key=f"delete_{project.id}", help=f"Delete {project.name}"):
                    if st.session_state.project_manager.delete_project(project.id):
                        if st.session_state.current_project_id == project.id:
                            st.session_state.current_project_id = None
                            st.session_state.chat_messages = []
                            st.session_state.search_history = []
                        st.rerun()
        
        st.markdown("---")
        
        st.subheader("Navigation")
        
        nav_options = {
            "chat": "💬 Chat",
            "search": "🔍 Search",
            "history": "📜 History",
            "share": "🔗 Share"
        }
        
        for key, label in nav_options.items():
            if st.button(label, key=f"nav_{key}"):
                st.session_state.page = key

def render_new_project_page():
    """Render the new project creation page."""
    st.markdown("<h1 class='section-header'>Create New Project</h1>", unsafe_allow_html=True)
    
    with st.form("new_project_form"):
        project_name = st.text_input("Project Name", "")
        project_description = st.text_area("Project Description", "")
        
        submit_button = st.form_submit_button("Create Project")
        
        if submit_button:
            if project_name:
                new_project = st.session_state.project_manager.create_project(project_name, project_description)
                st.session_state.current_project_id = new_project.id
                st.session_state.chat_messages = []
                st.session_state.search_history = []
                st.session_state.page = "chat"
                st.rerun()
            else:
                st.error("Project name is required")

def render_chat_page():
    """Render the chat interface."""
    current_project = None
    if st.session_state.current_project_id:
        current_project = st.session_state.project_manager.get_project(st.session_state.current_project_id)
    
    if current_project:
        st.markdown(f"<h1 class='section-header'>{current_project.name}</h1>", unsafe_allow_html=True)
        if current_project.description:
            st.markdown(f"<p>{current_project.description}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 class='section-header'>Research Playground</h1>", unsafe_allow_html=True)
        st.warning("No project selected. Please create or select a project from the sidebar.")
        return
    
    # Display existing messages
    for msg in st.session_state.chat_messages:
        render_message(msg)
    
    # Chat input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask anything about drug discovery, targets, pathways, diseases...", height=100)
        cols = st.columns([0.85, 0.15])
        with cols[1]:
            submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Add user message
            user_message = current_project.add_message("user", user_input)
            st.session_state.chat_messages.append(user_message)
            
            # Get AI response
            conversation_history = [{"role": msg.role, "content": msg.content} 
                                   for msg in st.session_state.chat_messages[:-1]]
            ai_response = get_deepseek_response(user_input, conversation_history)
            
            # Check if AI response is an error message
            if "Failed to generate analysis" in ai_response:
                st.error(ai_response)
            else:
                ai_message = current_project.add_message("assistant", ai_response)
                st.session_state.chat_messages.append(ai_message)
            
            # Save project
            st.session_state.project_manager.save_project(current_project)
            
            # Rerun to update UI
            st.rerun()

def render_search_page():
    """Render the publication search interface."""
    # Get current project
    current_project = None
    if st.session_state.current_project_id:
        current_project = st.session_state.project_manager.get_project(st.session_state.current_project_id)
    
    # Display project header
    if current_project:
        st.markdown(f"<h1 class='section-header'>Publication Search - {current_project.name}</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 class='section-header'>Publication Search</h1>", unsafe_allow_html=True)
        st.warning("No project selected. Please create or select a project from the sidebar.")
        return
    
    # Search interface with guidance
    st.markdown("""
    <div class="research-card">
        <h3>Search for Scientific Literature</h3>
        <p>Enter keywords related to your drug discovery research to find relevant publications. 
        You can search for:</p>
        <ul>
            <li>Target proteins (e.g., "EGFR inhibitors")</li>
            <li>Diseases (e.g., "breast cancer targeted therapy")</li>
            <li>Drug classes (e.g., "kinase inhibitor selectivity")</li>
            <li>Pathways (e.g., "PI3K/AKT/mTOR pathway")</li>
            <li>Techniques (e.g., "fragment-based drug discovery")</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Search interface
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        search_query = st.text_input("Search publications:", placeholder="Enter search terms...")
    
    with col2:
        st.write("")
        st.write("")
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Advanced search options
    with st.expander("Advanced Search Options"):
        col1, col2 = st.columns(2)
        with col1:
            max_results = st.slider("Maximum results", 5, 50, 10)
            sort_by = st.radio("Sort by", ["Relevance", "Date (newest first)"])
        with col2:
            search_sources = st.multiselect(
                "Sources", 
                ["PubMed", "Europe PMC"],
                default=["PubMed", "Europe PMC"]
            )
            year_range = st.slider("Publication Year", 1970, 2023, (2010, 2023))
    
    if search_button and search_query:
        with st.spinner(f"Searching publications for '{search_query}'..."):
            # Perform search
            results = search_publications(search_query, max_results=max_results)
            
            # Add search to history
            search_entry = current_project.add_search(search_query, results)
            st.session_state.search_history.append(search_entry)
            
            # Save project
            st.session_state.project_manager.save_project(current_project)
        
        # Display results
        if results:
            st.success(f"Found {len(results)} publications related to '{search_query}'")
            
            # Display results in a tabular format first
            result_data = []
            for i, result in enumerate(results):
                result_data.append({
                    "Title": result["title"][:80] + "..." if len(result["title"]) > 80 else result["title"],
                    "Authors": result["authors"][:50] + "..." if len(result["authors"]) > 50 else result["authors"],
                    "Journal": result["journal"],
                    "Year": result["year"],
                    "Source": result["source"]
                })
            
            df = pd.DataFrame(result_data)
            st.dataframe(df, use_container_width=True)
            
            # Then show detailed results in expandable sections
            st.subheader("Detailed Results")
            for i, result in enumerate(results):
                with st.expander(f"{i+1}. {result['title']}"):
                    st.markdown(f"**Authors:** {result['authors']}")
                    st.markdown(f"**Journal:** {result['journal']}, {result['year']}")
                    st.markdown(f"**DOI/ID:** {result['doi']}")
                    
                    if result.get('abstract'):
                        # Use a checkbox instead of a nested expander
                        show_abstract = st.checkbox("Show Abstract", key=f"abstract_{i}")
                        if show_abstract:
                            st.markdown(result['abstract'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Add to Chat", key=f"chat_{i}"):
                            prompt = f"Let's discuss this paper: {result['title']} by {result['authors']} ({result['year']}). The abstract mentions: {result.get('abstract', 'No abstract available')}"
                            user_message = current_project.add_message("user", prompt)
                            st.session_state.chat_messages.append(user_message)
                            st.session_state.project_manager.save_project(current_project)
                            st.session_state.page = "chat"
                            st.rerun()
                    
                    with col2:
                        if result.get('doi') and result['doi'].startswith('10.'):
                            link_url = f"https://doi.org/{result['doi']}"
                            st.markdown(f"[View Article]({link_url})")
                        elif result.get('pmid'):
                            link_url = f"https://pubmed.ncbi.nlm.nih.gov/{result['pmid']}/"
                            st.markdown(f"[View on PubMed]({link_url})")
        else:
            st.warning(f"No publications found for '{search_query}'. Try modifying your search terms.")
            st.markdown("""
            <div class="research-card">
                <h4>Search Tips</h4>
                <ul>
                    <li>Use more specific keywords</li>
                    <li>Try synonyms or alternative terms</li>
                    <li>Include protein or gene names</li>
                    <li>Combine disease names with drug classes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def render_history_page():
    """Render the conversation and search history."""
    current_project = None
    if st.session_state.current_project_id:
        current_project = st.session_state.project_manager.get_project(st.session_state.current_project_id)
    
    if current_project:
        st.markdown(f"<h1 class='section-header'>History - {current_project.name}</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 class='section-header'>History</h1>", unsafe_allow_html=True)
        st.warning("No project selected. Please create or select a project from the sidebar.")
        return
    
    tab1, tab2 = st.tabs(["Chat History", "Search History"])
    
    with tab1:
        messages_by_date = {}
        for msg in current_project.messages:
            date_str = msg.timestamp.strftime("%Y-%m-%d")
            if date_str not in messages_by_date:
                messages_by_date[date_str] = []
            messages_by_date[date_str].append(msg)
        
        for date_str, messages in sorted(messages_by_date.items(), reverse=True):
            with st.expander(f"📅 {date_str}", expanded=date_str == datetime.datetime.now().strftime("%Y-%m-%d")):
                for msg in messages:
                    render_message(msg, compact=True)
    
    with tab2:
        searches_by_date = {}
        for search in current_project.searches:
            date_str = search.timestamp.strftime("%Y-%m-%d")
            if date_str not in searches_by_date:
                searches_by_date[date_str] = []
            searches_by_date[date_str].append(search)
        
        for date_str, searches in sorted(searches_by_date.items(), reverse=True):
            with st.expander(f"📅 {date_str}", expanded=date_str == datetime.datetime.now().strftime("%Y-%m-%d")):
                for search in searches:
                    st.markdown(f"🔍 **Query:** {search.query}")
                    st.markdown(f"⏱️ {search.timestamp.strftime('%H:%M:%S')}")
                    if search.results:
                        st.markdown(f"Results: {len(search.results)} publications found")
                        if st.button("Show Results", key=f"show_{search.id}"):
                            st.session_state.page = "search"
                    st.markdown("---")

def render_share_page():
    """Render the project sharing interface."""
    current_project = None
    if st.session_state.current_project_id:
        current_project = st.session_state.project_manager.get_project(st.session_state.current_project_id)
    
    if current_project:
        st.markdown(f"<h1 class='section-header'>Share - {current_project.name}</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 class='section-header'>Share Research</h1>", unsafe_allow_html=True)
        st.warning("No project selected. Please create or select a project from the sidebar.")
        return
    
    st.subheader("Share Your Research")
    
    share_option = st.radio("Share as:", ["Link", "PDF Report", "Email", "Team Collaboration"])
    
    if share_option == "Link":
        st.text_input("Shareable Link (Copy this)", f"https://research-playground.example.com/share/{current_project.id}")
        st.markdown("This link will provide read-only access to your project.")
    
    elif share_option == "PDF Report":
        st.markdown("Generate a PDF report with your research findings.")
        report_sections = st.multiselect(
            "Select sections to include:",
            ["Project Summary", "Key Findings", "Conversation History", "Literature References"],
            ["Project Summary", "Key Findings"]
        )
        if st.button("Generate Report"):
            st.success("Report generated successfully!")
            st.download_button(
                "Download Report",
                data=b"Sample PDF content",
                file_name=f"{current_project.name}_report.pdf",
                mime="application/pdf"
            )
    
    elif share_option == "Email":
        with st.form("email_form"):
            recipients = st.text_input("Recipients (comma separated)")
            subject = st.text_input("Subject", f"Research Project: {current_project.name}")
            message = st.text_area("Message", "I'd like to share my research findings with you.")
            submit = st.form_submit_button("Send Email")
            if submit:
                st.success("Email sent successfully!")
    
    elif share_option == "Team Collaboration":
        st.markdown("Invite team members to collaborate on this project.")
        with st.form("collaboration_form"):
            team_members = st.text_input("Team Members (email addresses, comma separated)")
            permission = st.selectbox("Permission Level", ["View Only", "Comment", "Edit", "Admin"])
            submit = st.form_submit_button("Send Invitations")
            if submit:
                st.success("Invitations sent successfully!")

def render_message(message: Message, compact: bool = False):
    """Render a chat message."""
    css_class = "chat-message user" if message.role == "user" else "chat-message assistant"
    role_name = "You" if message.role == "user" else "Research Assistant"
    
    if compact:
        st.markdown(f"""
        <div class="{css_class}" style="padding: 0.8rem;">
            <div class="message-header">{role_name}</div>
            <div class="message-content">{message.content[:100]}{'...' if len(message.content) > 100 else ''}</div>
            <div class="timestamp">{message.timestamp.strftime('%H:%M:%S')}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="{css_class}">
            <div class="message-header">{role_name}</div>
            <div class="message-content">{message.content}</div>
            <div class="timestamp">{message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    
    if st.session_state.page == "new_project":
        render_new_project_page()
    elif st.session_state.page == "chat":
        render_chat_page()
    elif st.session_state.page == "search":
        render_search_page()
    elif st.session_state.page == "history":
        render_history_page()
    elif st.session_state.page == "share":
        render_share_page()

if __name__ == "__main__":
    main()