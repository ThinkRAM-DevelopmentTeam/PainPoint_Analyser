#requirement installs: pip install streamlit psycopg2 resend # streamlit
# psycopg2-binary
# resend
# requests
# openai
# beautifulsoup4
# langchain-google-genai
# langchain
# langchain-community
# langchain-core
# langchain-text-splitters
# langchain-openai


import streamlit as st
import psycopg2
import json
import resend
import requests
import re
from openai import OpenAI
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pandas as pd

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Business Analyst AI")


# --- Initializing Clients ---
try:
    # It's assumed your secrets are configured for a deployed app.
    # For local development, you might handle these differently.
    resend.api_key = st.secrets["resend"]["api_key"]
    openai_client = OpenAI(api_key=st.secrets["openai"]["api_key"])
except KeyError as e:
    st.error(f"Missing secret: {e}. Please ensure your secrets are configured in Streamlit Cloud or locally in .streamlit/secrets.toml.")
    st.stop()


# --- Database Functions (Restored) ---
@st.cache_data(ttl=600)
def fetch_data_from_neon_db():
    try:
        conn = psycopg2.connect(st.secrets["connections"]["neon"]["url"])
        with conn.cursor() as cur:
            cur.execute("SELECT place_id, details_json FROM places_cache")
            records = cur.fetchall()
            return [{"place_id": rec[0], "details": rec[1]} for rec in records if rec[1] is not None]
    except (Exception, psycopg2.Error) as error:
        st.error(f"Error fetching data from Neon DB: {error}")
        return []

def update_email_in_db(place_id, email):
    try:
        conn = psycopg2.connect(st.secrets["connections"]["neon"]["url"])
        with conn.cursor() as cur:
            update_query = "UPDATE places_cache SET details_json = details_json || %s WHERE place_id = %s"
            email_json = json.dumps({'email': email})
            cur.execute(update_query, (email_json, place_id))
            conn.commit()
        st.cache_data.clear()
        return True
    except (Exception, psycopg2.Error) as error:
        st.error(f"Database Update Error: {error}")
        return False

# --- AI & Web Scraping Functions (Restored) ---
@st.cache_data(ttl=3600)
def get_website_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        return " ".join(soup.stripped_strings), response.text
    except requests.RequestException as e:
        return f"Error fetching website: {e}", None

def find_contact_links(base_url, html_content):
    if not html_content: return []
    soup = BeautifulSoup(html_content, 'html.parser')
    links = set()
    keywords = ['contact', 'about', 'impressum', 'kontakt', 'support', 'help']
    for a in soup.find_all('a', href=True):
        href = a['href'].lower()
        link_text = a.text.lower()
        if any(keyword in href or keyword in link_text for keyword in keywords):
            full_url = urljoin(base_url, a['href'])
            links.add(full_url)
    return list(links)

def find_email_with_openai(content):
    if content.startswith("Error"): return None, content
    prompt = f"""Scan the following website text and find the best single public contact email address (like info@, contact@, support@, hello@). Do not invent an email. Reply with only the email address and nothing else. If you cannot find a suitable email, reply with the single word: None. Website Text: "{content[:4000]}" """
    try:
        response = openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        potential_email = response.choices[0].message.content.strip()
        if re.match(r"[^@]+@[^@]+\.[^@]+", potential_email):
            return potential_email, None
        else:
            return None, "AI could not find a valid email address."
    except Exception as e:
        return None, f"An error occurred with the AI model: {e}"

def intelligently_find_email(base_url):
    stripped_content, full_html = get_website_content(base_url)
    if stripped_content.startswith("Error"): return None, stripped_content
    contact_urls = find_contact_links(base_url, full_html)
    urls_to_scan = contact_urls + [base_url]
    for url in urls_to_scan:
        content, _ = get_website_content(url)
        if not content.startswith("Error"):
            email, error = find_email_with_openai(content)
            if email: return email, None
    return None, "Could not find a valid email on the website or its contact pages."

# --- Email Sending and Generation (Restored) ---
def send_bulk_email(recipients, subject, body_template):
    success_count = 0; error_list = []
    for r in recipients:
        try:
            p_body = body_template.replace('{{name}}', r.get('name', 'there'))
            h_body = p_body.replace('\n', '<br>')
            params = {"from": st.secrets["email"]["sender_email"], "to": [r["email"]], "subject": subject, "html": f"<html><body>{h_body}</body></html>"}
            email = resend.Emails.send(params)
            if 'id' in email: success_count += 1
            else: error_list.append(f"{r['name']}: {email.get('message', 'Unknown error')}")
        except Exception as e:
            error_list.append(f"{r['name']}: {str(e)}")
    return success_count, error_list

def generate_email_with_llm(requirement):
    prompt = f"""You are an expert copywriter... [Your detailed prompt here] ...Compose the email body now."""
    try:
        response = openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            st.error("AI failed to generate a response. The response was empty.")
            return None
    except Exception as e:
        st.error(f"An unexpected AI error occurred: {e}")
        return None

# --- API Communication Functions ---
def search_businesses(params: dict):
    """Calls the FastAPI /business-search endpoint with a dynamic set of filters."""
    search_url = f"{API_BASE_URL}/business-search"
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the Business Search API: {e}")
        if e.response is not None:
            st.error(f"Backend Error: {e.response.json().get('detail', e.response.text)}")
        return []

def chat_with_analyst(business_data: list, question: str):
    """Calls the FastAPI /chat endpoint."""
    chat_url = f"{API_BASE_URL}/chat"
    payload = {
        "business_data": business_data,
        "user_question": question,
        "use_lean_payload": True
    }
    try:
        response = requests.post(chat_url, json=payload)
        response.raise_for_status()
        return response.json().get("answer", "No answer received from AI.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the AI Chat API: {e}")
        return "Failed to get a response from the AI."


# --- Main Streamlit App ---
st.title("Business Analyst AI Dashboard")
st.caption("A full-featured Streamlit UI for your FastAPI-powered Business Intelligence Service")

# --- Section 1: Business Search with All Filters ---
st.header("1. Find Businesses")

with st.form("search_form"):
    st.info("Select data fields, define your search area, and apply filters to find specific businesses.")

    # --- Field Selection ---
    available_fields = ["rating", "reviews", "opening_hours", "website", "formatted_phone_number", "email"]
    selected_fields = st.multiselect(
        "Select Data Fields to Retrieve",
        options=available_fields,
        default=["rating", "website", "formatted_phone_number"],
        help="Choosing fewer fields can speed up the search. Select fields required for your filters (e.g., 'rating' for rating filters)."
    )

    st.write("---")
    # --- Location & Keyword ---
    st.subheader("Search Area")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        keyword = st.text_input("Keyword", "restaurant")
    with col2:
        lat = st.number_input("Latitude", value=40.7128, format="%.4f")
    with col3:
        lng = st.number_input("Longitude", value=-74.0060, format="%.4f")
    with col4:
        radius = st.number_input("Radius (m)", value=5000)

    # --- Advanced Filters ---
    st.write("---")
    st.subheader("Advanced Filters")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        min_rating = st.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.1)
        max_rating = st.slider("Maximum Rating", 0.0, 5.0, 5.0, 0.1)
    with col_f2:
        min_reviews = st.number_input("Minimum Number of Reviews", 0, 10000, 0)
    with col_f3:
        days_of_week = ["--", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        open_on_day = st.selectbox("Open on Day", options=days_of_week)

    # --- Pain Points Filter ---
    pain_point_options = {"--": None, "Yes": True, "No": False}
    selected_pain_point_option = st.selectbox(
        "Has Customer Pain Points?",
        options=list(pain_point_options.keys()),
        help="Filter for businesses with (Yes) or without (No) negative reviews."
    )
    has_pain_points = pain_point_options[selected_pain_point_option]


    # --- Cost Control ---
    st.write("---")
    st.subheader("Cost & Depth Control")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        desired_total_results = st.number_input("Total Initial Results to Fetch", min_value=1, max_value=60, value=20)
    with col_c2:
        max_results_to_process = st.number_input("Limit Processing to Top N Results", min_value=1, max_value=60, value=10)

    submitted = st.form_submit_button("Search for Businesses", type="primary", use_container_width=True)

if submitted:
    if not selected_fields:
        st.error("You must select at least one data field to retrieve.")
    else:
        # Build the params dictionary dynamically based on user input
        search_params = {
            "keyword": keyword, "lat": lat, "lng": lng, "radius_m": radius,
            "fields": selected_fields,
            "desired_total_results": desired_total_results,
            "max_results_to_process": max_results_to_process
        }

        # --- FIX APPLIED HERE ---
        # This corrected logic now always sends the rating filter values.
        # This is safe because the refactored backend is built to handle these
        # default values (0.0 and 5.0) correctly. This synchronizes front and back ends.
        search_params['min_rating'] = min_rating
        search_params['max_rating'] = max_rating
        # --- END OF FIX ---

        # Add other optional filters only if they have been set to a non-default value
        if min_reviews > 0: search_params['min_reviews'] = min_reviews
        if has_pain_points is not None: search_params['has_pain_points'] = has_pain_points
        if open_on_day != "--": search_params['open_on_day'] = open_on_day

        with st.spinner("Searching for businesses via API..."):
            results = search_businesses(search_params)
            if results:
                df = pd.DataFrame(results)
                st.session_state['results_df'] = df
                st.success(f"Found and processed {len(results)} businesses matching your criteria.")
            else:
                if 'results_df' in st.session_state:
                    del st.session_state['results_df']
                st.warning("No businesses found for your specific criteria.")
                st.info(
                    """
                    **Tip:** If you're not getting results, try the following:
                    - **Broaden your search radius.**
                    - **Loosen the filters:** Set ratings from 0.0 to 5.0 and "Has Customer Pain Points?" to '--'.
                    - **Try a more general keyword** (e.g., "food" instead of a specific restaurant name).
                    - **Increase the "Total Initial Results to Fetch"** to search a larger initial pool of candidates.
                    """
                )

# --- Section 2: Display Results & AI Analyst (Restored) ---
if 'results_df' in st.session_state:
    st.divider()
    st.header("2. Analyze Results")
    results_df = st.session_state['results_df']

    st.subheader("Search Results")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.subheader("Chat with AI Analyst")
    st.info("Ask a question about the data shown above. The AI will provide an analysis.")

    user_question = st.text_input("Your question:", placeholder="e.g., Which business has the lowest rating?")

    if st.button("Ask AI Analyst"):
        if user_question:
            # The backend may return slightly different data structures now, so ensure this works
            # Using .to_dict('records') is generally robust.
            business_data_list = results_df.to_dict('records')
            with st.spinner("AI is analyzing the data..."):
                answer = chat_with_analyst(business_data_list, user_question)
                st.markdown("#### AI Analyst's Answer:")
                st.success(answer)
        else:
            st.warning("Please enter a question for the AI.")
