import streamlit as st
import psycopg2
import json
import resend
import requests
import re
from openai import OpenAI # Import OpenAI
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
# time is no longer needed for rate limiting in this version

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Email Finder & Sender")

# --- Initializing Clients ---
try:
    resend.api_key = st.secrets["resend"]["api_key"]
    # UPDATED: Initialize OpenAI client
    openai_client = OpenAI(api_key=st.secrets["openai"]["api_key"])
except KeyError as e:
    st.error(f"Missing secret: {e}. Please check your .streamlit/secrets.toml file.")
    st.stop()


# --- Database Functions (Unchanged) ---
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

# --- AI & Web Scraping Functions ---
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

# --- REPLACED: Function to find email using OpenAI ---
def find_email_with_openai(content):
    """Uses OpenAI to find an email address in the provided text content."""
    if content.startswith("Error"): return None, content
    
    prompt = f"""Scan the following website text and find the best single public contact email address (like info@, contact@, support@, hello@). Do not invent an email. Reply with only the email address and nothing else. If you cannot find a suitable email, reply with the single word: None. Website Text: "{content[:4000]}" """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        potential_email = response.choices[0].message.content.strip()
        if re.match(r"[^@]+@[^@]+\.[^@]+", potential_email):
            return potential_email, None
        else:
            return None, "AI could not find a valid email address."
    except Exception as e:
        return None, f"An error occurred with the AI model: {e}"

def intelligently_find_email(base_url):
    """Intelligently searches a website for an email by checking contact pages first."""
    stripped_content, full_html = get_website_content(base_url)
    if stripped_content.startswith("Error"): return None, stripped_content
    
    contact_urls = find_contact_links(base_url, full_html)
    urls_to_scan = contact_urls + [base_url]
    for url in urls_to_scan:
        content, _ = get_website_content(url)
        if not content.startswith("Error"):
            # UPDATED: Call the OpenAI function
            email, error = find_email_with_openai(content)
            if email: return email, None
    return None, "Could not find a valid email on the website or its contact pages."


# --- Email Sending and Generation ---
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

# --- REPLACED: Email Generation with OpenAI ---
def generate_email_with_llm(requirement):
    """Generates an email body using OpenAI."""
    prompt = f"""
    You are an expert copywriter specializing in professional business communication.
    Your task is to compose a clear, concise, and professional email body based on the user's specified purpose.

    **User's Purpose:** "{requirement}"

    **Instructions:**
    1. Start the email *exactly* with `Hi {{{{name}}}},`.
    2. The tone should be professional, courteous, and engaging.
    3. Clearly articulate the user's purpose.
    4. End the email *exactly* with `Best regards,`.
    5. Do NOT include a subject line.

    Compose the email body now.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            st.error("AI failed to generate a response. The response was empty.")
            return None
    except Exception as e:
        st.error(f"An unexpected AI error occurred: {e}")
        return None


# --- Main Streamlit App (UI code remains largely the same) ---
st.title("AI-Powered Email Finder & Campaign Sender")

if 'recipients' not in st.session_state: st.session_state.recipients = []

st.header("1. Build Your Recipient List")
col1, col2 = st.columns([2, 1])

with col1:
    # Database selection UI (unchanged)
    st.subheader("Select from Database")
    customer_data = fetch_data_from_neon_db()
    if not customer_data: st.info("No data fetched from database.")
    else:
        for record in customer_data:
            details, place_id = record.get("details", {}), record.get("place_id")
            if not details or not place_id: continue
            name = details.get('name', 'N/A')
            with st.expander(f"**{name}**"):
                st.write(f"**Phone:** {details.get('formatted_phone_number', 'N/A')}")
                st.write(f"**Address:** {details.get('formatted_address', 'N/A')}")
                st.write(f"**Website:** {details.get('website', 'N/A')}")
                email_status_placeholder = st.empty()
                if details.get('email'):
                    email_status_placeholder.success(f"**Email:** {details.get('email')}")
                    st.checkbox("Add to recipient list", key=f"select_{place_id}")
                elif details.get('website'):
                    session_key = f"searched_{place_id}"
                    if session_key not in st.session_state:
                        with email_status_placeholder.container():
                            with st.spinner(f"Automatically scanning {details.get('website')} for an email..."):
                                email, error = intelligently_find_email(details.get('website'))
                                if email and update_email_in_db(place_id, email):
                                    st.session_state[session_key] = {'status': 'found', 'email': email}
                                    st.rerun()
                                else:
                                    st.warning(error or "Could not find an email.")
                                    st.session_state[session_key] = {'status': 'not_found', 'error': error}
                    else:
                        result = st.session_state[session_key]
                        if result['status'] == 'not_found':
                            email_status_placeholder.warning(result.get('error') or "Could not find an email.")
                else:
                    email_status_placeholder.error("No email or website available for this record.")
        
        selected_db = [r['details'] for r in customer_data if st.session_state.get(f"select_{r.get('place_id')}")]
        manual_adds = [r for r in st.session_state.get('recipients', []) if 'manual' in r]
        st.session_state.recipients = selected_db + manual_adds

with col2:
    # Manual add form (unchanged)
    st.subheader("Manually Add Customer")
    with st.form("manual_form", clear_on_submit=True):
        name = st.text_input("Name*"); email = st.text_input("Email*")
        if st.form_submit_button("Add to List") and name and email:
            new = {"name": name, "email": email, "manual": True}
            if not any(r['email'] == email for r in st.session_state.recipients):
                st.session_state.recipients.append(new); st.rerun()
            else: st.warning("Email already on list.")

st.subheader("Current Recipient List")
if not st.session_state.get('recipients'): st.info("Your recipient list is empty.")
else:
    # Recipient list display (unchanged)
    unique_recipients = {r['email']: r for r in st.session_state.recipients}.values()
    st.session_state.recipients = list(unique_recipients)
    for r in st.session_state.recipients: st.write(f"- **{r.get('name')}** ({r.get('email')})")
    if st.button("Clear List"):
        st.session_state.recipients = []; [st.session_state.pop(k) for k in list(st.session_state.keys()) if k.startswith('select_')]; st.rerun()


# --- Section 2: Compose and Send ---
st.header("2. Compose and Send Email")

if 'send_status' in st.session_state:
    status = st.session_state.pop('send_status')
    s_count, e_list = status['success'], status['errors']
    if s_count > 0: st.success(f"Campaign Finished! Successfully sent {s_count} emails.")
    if e_list: st.error("Some emails failed to send:"); [st.write(f"- {err}") for err in e_list]
    st.divider()

if st.session_state.get('recipients'):
    subject = st.text_input("Email Subject", "Following up")
    requirement = st.text_area("Describe the email's purpose:", height=150, placeholder="e.g., Pitch our new AI-powered analytics tool and ask for a 15-minute demo call next week.")
    
    if 'email_body' not in st.session_state:
        if st.button("✨ Generate Email Body with AI", type="primary") and requirement:
            with st.spinner("AI is writing..."):
                generated_body = generate_email_with_llm(requirement)
            if generated_body:
                st.session_state.email_body = generated_body
                st.rerun()
    else:
        st.subheader("Generated Email Template")
        st.info("You can edit the text below before sending.")
        st.session_state.email_body = st.text_area("Email Body", st.session_state.email_body, height=300, label_visibility="collapsed")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✨ Regenerate with AI", use_container_width=True):
                if requirement:
                    with st.spinner("AI is writing a new version..."):
                        new_body = generate_email_with_llm(requirement)
                    if new_body:
                        st.session_state.email_body = new_body
                        st.rerun()
                else:
                    st.warning("Please describe the email's purpose above before regenerating.")
        
        with col2:
            if st.button("✅ Send Email to All", type="primary", use_container_width=True):
                with st.spinner("Sending emails..."):
                    s, e = send_bulk_email(st.session_state.recipients, subject, st.session_state.email_body)
                    st.session_state.send_status = {'success': s, 'errors': e}
                    st.session_state.recipients = []
                    del st.session_state['email_body']
                    for k in list(st.session_state.keys()):
                        if k.startswith('select_'):
                            del st.session_state[k]
                    st.rerun()