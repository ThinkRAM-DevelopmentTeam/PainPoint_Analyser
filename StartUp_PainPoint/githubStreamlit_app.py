import io
import zipfile
from datetime import datetime
import pandas as pd
import requests
import streamlit as st
import numpy as np

# Local vectorization and similarity search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LangChain components for LLM interaction
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.messages import HumanMessage, AIMessage

# == UI Constants ==
HEADER_ICON = "🐙"
PROJECT_ICON = "📁"
IDEA_ICON = "💡"
CHAT_ICON = "🤖"
DOWNLOAD_ICON = "⬇️"

# GitHub API base URL
GITHUB_API_URL = "https://api.github.com"

def initialize_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        'projects_df': pd.DataFrame(),
        'ai_summary': None,
        'tfidf_vectorizer': None,
        'tfidf_matrix': None,
        'chat_history': [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

@st.cache_data(show_spinner="Searching GitHub for repositories...")
def search_github_repositories(keywords, languages, topics, per_page=50, token=None):
    """Searches GitHub repositories using the REST API."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    q_parts = []
    if keywords: q_parts.extend(keywords)
    if languages: q_parts.extend([f'language:"{lang}"' for lang in languages])
    if topics: q_parts.extend([f'topic:{t}' for t in topics])
    if not q_parts: return []
    query = ' '.join(q_parts)
    params = {"q": query, "per_page": per_page, "sort": "stars", "order": "desc"}
    response = requests.get(f"{GITHUB_API_URL}/search/repositories", headers=headers, params=params)
    response.raise_for_status()
    return response.json().get('items', [])

@st.cache_data(show_spinner=False)
def fetch_readme(owner, repo, token=None):
    """Fetches README content for a repository."""
    headers = {"Accept": "application/vnd.github.v3.raw"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/readme"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        return response.text if response.status_code == 200 else ""
    except requests.RequestException:
        return ""

def preprocess_project_data(repos, github_token=None, required_features=None):
    """Processes repository data and fetches READMEs."""
    rows = []
    if not repos: return pd.DataFrame()
    progress_bar = st.progress(0, text="Fetching READMEs and processing data...")
    for i, repo in enumerate(repos):
        readme_content = fetch_readme(repo['owner']['login'], repo['name'], token=github_token)
        textsearch = (f"{repo.get('name', '')} {repo.get('description', '')} {readme_content}").lower()
        matched_features = [f for f in required_features if f.lower() in textsearch] if required_features else []
        rows.append({
            'name': repo.get('name'), 'full_name': repo.get('full_name'), 'html_url': repo.get('html_url'),
            'description': repo.get('description'), # Keep it as None if that's what API returns
            'stars': repo.get('stargazers_count', 0),
            'language': repo.get('language', 'N/A'),
            'last_updated': pd.to_datetime(repo.get('updated_at')).strftime('%Y-%m-%d'),
            'readme': readme_content, 'matched_features': list(set(matched_features)),
            'match_count': len(set(matched_features)),
        })
        progress_bar.progress((i + 1) / len(repos), text=f"Processing: {repo.get('full_name')}")
    progress_bar.empty()
    df = pd.DataFrame(rows)
    df.sort_values(['match_count', 'stars'], ascending=[False, False], inplace=True)
    return df.reset_index(drop=True)

# --- MODIFIED AND FIXED FUNCTION ---
def build_tfidf_retriever(df):
    """Builds a TF-IDF vectorizer and matrix from project data."""
    if df.empty:
        return None, None

    # FIX: Fill any potential NaN values in text columns with an empty string
    # This prevents the vectorizer from crashing on missing descriptions.
    documents = (
        df['full_name'].fillna('') + ". " +
        df['description'].fillna('') + ". " +
        df['readme'].fillna('')
    ).tolist()

    n_documents = len(documents)
    min_df_param, max_df_param = (1, 1.0) if n_documents < 4 else (2, 0.7)
    vectorizer = TfidfVectorizer(
        stop_words='english', min_df=min_df_param, max_df=max_df_param
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix
# --- END OF MODIFICATION ---


def prepare_download_files(df):
    """Prepares project data for download."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"github_projects_{timestamp}"
    export_df = df.copy()
    export_df['matched_features'] = export_df['matched_features'].apply(lambda x: ', '.join(x))
    csv_bytes = export_df.drop(columns=['readme']).to_csv(index=False).encode('utf-8')
    json_bytes = export_df.drop(columns=['readme']).to_json(orient='records', indent=2).encode('utf-8')
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base_filename}.csv", csv_bytes)
        zf.writestr(f"{base_filename}.json", json_bytes)
    mem_zip.seek(0)
    return csv_bytes, json_bytes, mem_zip, base_filename

def horizontal_download_buttons(csv_bytes, json_bytes, mem_zip, base_filename):
    """Displays download buttons for the project list."""
    st.markdown("#### Download Project List")
    c1, c2, c3 = st.columns(3)
    c1.download_button(f"{DOWNLOAD_ICON} Download CSV", csv_bytes, f"{base_filename}.csv", "text/csv")
    c2.download_button(f"{DOWNLOAD_ICON} Download JSON", json_bytes, f"{base_filename}.json", "application/json")
    c3.download_button(f"{DOWNLOAD_ICON} Download ZIP", mem_zip, f"{base_filename}.zip", "application/zip")

def generate_ai_ideas(user_needs, df, llm_model, api_key):
    """Generates project upgrade ideas using an LLM."""
    if df.empty: return "No projects available for idea generation."
    context = ""
    for _, row in df.head(10).iterrows():
        context += (f"Project: {row['full_name']}\nDescription: {row['description'] or 'N/A'}\n"
                    f"Features matched: {', '.join(row['matched_features'])}\n\n")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert product manager."),
        ("human", "User's needs: '{user_needs}'.\n\nGitHub Projects:\n{project_context}\n\n"
                  "Provide a concise, actionable summary with:\n"
                  "- **Project Upgrades:** How to make individual projects market-ready.\n"
                  "- **Project Combinations:** Which projects can be combined for a full-scale product.\n"
                  "- **New Feature Ideas:** What essential features are missing.")
    ])
    llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key, temperature=0.3)
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"user_needs": user_needs, "project_context": context})

def handle_ai_chat(chat_history, df, vectorizer, matrix, llm_model, api_key):
    """
    Handles conversational AI chat using a more robust, two-step contextual retriever.
    """
    if vectorizer is None or matrix is None:
        return "The knowledge base is not built. Please search for projects first."

    latest_question = ""
    if chat_history and isinstance(chat_history[-1], HumanMessage):
        latest_question = chat_history[-1].content.lower()

    if not latest_question:
        return "Could not identify the latest question to process."

    relevant_indices = []
    searchable_names = df['full_name'].str.lower()
    for idx, name in searchable_names.items():
        repo_name = name.split('/')[-1]
        if repo_name in latest_question or name in latest_question:
            relevant_indices.append(idx)

    if not relevant_indices:
        query_vector = vectorizer.transform([latest_question])
        cosine_similarities = cosine_similarity(query_vector, matrix).flatten()
        k = min(4, len(cosine_similarities))
        top_indices = np.argsort(-cosine_similarities)[:k]
        relevant_indices = [i for i in top_indices if cosine_similarities[i] > 0.1]

    context = ""
    sources = []
    unique_indices = sorted(list(set(relevant_indices)))

    if unique_indices:
        for idx in unique_indices:
            doc = df.iloc[idx]
            context += (f"--- Source: {doc['full_name']} ---\n"
                        f"Description: {doc['description']}\n"
                        f"README snippet: {doc['readme'][:500]}...\n\n")
            sources.append({"name": doc['full_name'], "url": doc['html_url']})
    else:
        context = "Based on the provided context, there is no information about the project or topic you mentioned."

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the user's questions based on the provided context from GitHub projects and our ongoing conversation. If the context doesn't contain the answer, state that clearly and do not make up information."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "CONTEXT FOR MY LATEST QUESTION:\n{context}\n\nBased on our conversation and the context above, please answer my latest question."),
    ])
    
    llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key, temperature=0.2)
    chain = prompt_template | llm | StrOutputParser()
    
    response = chain.invoke({"chat_history": chat_history, "context": context})
    
    if sources:
        sources_md = ", ".join([f"[{s['name']}]({s['url']})" for s in sources])
        response += f"\n\n---\n*Sources:* {sources_md}"
        
    return response

def show_example_questions():
    """Displays a few example questions for the AI chat."""
    st.markdown("##### Example Questions:")
    st.markdown(
        """
        - "Which project has the most modern backend stack?"
        - "Summarize the one by YasinzHyper."
        - "Based on that, what features should I add to make it better?"
        - "Is there a project that already has user authentication?"
        """
    )
    st.divider()

def main():
    st.set_page_config(page_title="AI GitHub Project Explorer", page_icon=HEADER_ICON, layout="wide")
    st.title(f"{HEADER_ICON} AI-Powered GitHub Project Explorer")
    st.markdown("Find, analyze, and get AI-driven insights on GitHub projects tailored to your requirements.")
    initialize_state()

    all_languages = [
        "Assembly", "Bash", "C", "C#", "C++", "Clojure", "CoffeeScript", "Crystal", "CSS",
        "Dart", "DM", "Dockerfile", "Elixir", "Elm", "Erlang", "F#", "Go", "Groovy",
        "Haskell", "HTML", "Java", "JavaScript", "Julia", "Jupyter Notebook", "Kotlin",
        "Latex", "Less", "LiveScript", "Lua", "Makefile", "Markdown", "Matlab", "Objective-C",
        "OCaml", "Pascal", "Perl", "PHP", "PowerShell", "Prolog", "Puppet", "Python",
        "R", "Reason", "Ruby", "Rust", "Sass", "Scala", "Scheme", "SCSS", "Shell",
        "Solidity", "SQL", "Swift", "TeX", "TypeScript", "V", "Vala", "Verilog",
        "VHDL", "Vim script", "Vue", "WebAssembly"
    ]

    with st.sidebar:
        st.header("1. Project Requirements")
        
        st.info("All filters (Keywords, Languages, Topics) are combined. For best results, ensure your topics match your selected languages.", icon="💡")
        
        user_keywords = st.text_input("Keywords (e.g., real estate, ai, react)", "full stack real estate")
        
        selected_langs = st.multiselect(
            "Languages", 
            options=all_languages, 
            default=["JavaScript", "TypeScript"],
            help="Select common languages from the list."
        )
        
        custom_langs_str = st.text_input(
            "Add other languages", 
            placeholder="e.g., Zig, Nim, Mojo",
            help="Enter any languages not in the list, separated by commas."
        )
        
        custom_langs_list = [lang.strip() for lang in custom_langs_str.split(',') if lang.strip()]
        final_user_langs = sorted(list(set(selected_langs + custom_langs_list)))
        
        user_topics = st.text_input("Topics/Tags (comma-separated)", "", placeholder="e.g., django, flask, fastapi")
        
        num_projects = st.slider("Projects to retrieve", 10, 100, 25)
        github_token = st.text_input("GitHub Token", type="password", help="Use a GitHub 'Classic' PAT with no scopes.")
        st.divider()
        st.header("2. AI Settings")
        use_ai = st.checkbox("Enable AI Features", value=True)
        llm_model = st.selectbox("Gemini Model", ["gemini-2.5-pro","gemini-1.5-pro-latest", "gemini-1.0-pro"])
        gemini_api_key = st.text_input("Google AI API Key", type="password")
        st.divider()
        run_button = st.button("🔍 Search & Analyze Projects", use_container_width=True)

    if run_button:
        if use_ai and not gemini_api_key:
            st.error("Please enter your Google AI API Key to use AI features."); st.stop()
        with st.spinner("Searching GitHub and analyzing repositories..."):
            keywords_list = [k.strip() for k in user_keywords.split(',') if k.strip()]
            topics_list = [t.strip() for t in user_topics.split(',') if t.strip()]
            
            query_display = keywords_list + [f'lang:"{l}"' for l in final_user_langs] + [f"topic:{t}" for t in topics_list]
            st.info(f"**Executing Search:** `{' '.join(query_display)}`")
            try:
                repos = search_github_repositories(keywords_list, final_user_langs, topics_list, num_projects, github_token)
                if not repos:
                    st.warning("No projects found. Please broaden your search filters."); st.stop()
                df = preprocess_project_data(repos, github_token, list(set(keywords_list + topics_list)))
                st.session_state.projects_df = df
                if df.empty:
                    st.warning("No projects could be processed."); st.stop()
                st.success(f"Successfully processed {len(df)} projects.")
                if use_ai:
                    with st.spinner("Building local knowledge base for chat..."):
                        vectorizer, matrix = build_tfidf_retriever(df)
                        st.session_state.tfidf_vectorizer = vectorizer
                        st.session_state.tfidf_matrix = matrix
                        st.session_state.chat_history = []
                        st.session_state.ai_summary = None
            except Exception as e:
                st.error(f"An error occurred: {e}"); st.stop()

    if not st.session_state.projects_df.empty:
        df = st.session_state.projects_df
        st.subheader(f"{PROJECT_ICON} Found Projects")
        display_cols = ['name', 'stars', 'language', 'match_count', 'matched_features', 'description', 'html_url']
        st.dataframe(df[display_cols].fillna('N/A'), use_container_width=True, hide_index=True) # Added fillna for display
        csv, json, zip_f, fname = prepare_download_files(df)
        horizontal_download_buttons(csv, json, zip_f, fname)

        if use_ai and gemini_api_key:
            st.divider()
            st.subheader(f"{IDEA_ICON} AI-Generated Upgrade & Combination Ideas")
            if not st.session_state.ai_summary:
                 with st.spinner("Generating AI insights..."):
                    needs = f"{user_keywords}, {', '.join(final_user_langs)}"
                    st.session_state.ai_summary = generate_ai_ideas(needs, df, llm_model, gemini_api_key)
            st.markdown(st.session_state.ai_summary)
            st.download_button(
                label=f"{DOWNLOAD_ICON} Download AI Report", data=st.session_state.ai_summary,
                file_name=f"ai_upgrade_report_{datetime.now().strftime('%Y%m%d')}.md", mime="text/markdown"
            )

            st.divider()
            st.subheader(f"{CHAT_ICON} Chat with Project Data")
            
            for msg in st.session_state.chat_history:
                with st.chat_message(msg.type):
                    st.markdown(msg.content)

            if not st.session_state.chat_history:
                show_example_questions()

            if user_question := st.chat_input("Ask a follow-up question about the projects..."):
                st.session_state.chat_history.append(HumanMessage(content=user_question))
                with st.chat_message("human"):
                    st.markdown(user_question)

                with st.chat_message("ai"):
                    with st.spinner("Thinking..."):
                        response = handle_ai_chat(st.session_state.chat_history, df, 
                                                  st.session_state.tfidf_vectorizer, 
                                                  st.session_state.tfidf_matrix, 
                                                  llm_model, gemini_api_key)
                        st.markdown(response, unsafe_allow_html=True)
                
                st.session_state.chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()