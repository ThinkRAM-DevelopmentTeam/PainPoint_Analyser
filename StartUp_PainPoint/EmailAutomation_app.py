import streamlit as st
import psycopg2
import json
import resend
import requests
import re
import time
from openai import OpenAI
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timedelta, timezone
from textblob import TextBlob

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Business Finder & Sender")

# --- Initializing Clients ---
try:
    resend.api_key = st.secrets["resend"]["api_key"]
    openai_client = OpenAI(api_key=st.secrets["openai"]["api_key"])
    google_api_key = st.secrets["google"]["api_key"]
    db_connection_url = st.secrets["connections"]["neon"]["url"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Please check your .streamlit/secrets.toml file.")
    st.stop()

# --- Database Functions ---
def get_db_connection():
    """Establishes a new database connection."""
    try:
        return psycopg2.connect(db_connection_url)
    except psycopg2.Error as e:
        st.error(f"Database connection error: {e}")
        return None

def fetch_data_from_neon_db():
    conn = get_db_connection()
    if not conn: return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT place_id, details_json FROM places_cache")
            records = cur.fetchall()
            return [{"place_id": rec[0], "details": rec[1]} for rec in records if rec[1] is not None]
    except psycopg2.Error as error:
        st.error(f"Error fetching data from Neon DB: {error}")
        return []
    finally:
        if conn: conn.close()

def update_details_in_db(place_id, new_details):
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT details_json FROM places_cache WHERE place_id = %s", (place_id,))
            record = cur.fetchone()
            if record:
                current_details = record[0]
                current_details.update(new_details)
                updated_json_string = json.dumps(current_details) # Convert dict to JSON string
                cur.execute("UPDATE places_cache SET details_json = %s WHERE place_id = %s", (updated_json_string, place_id))
                conn.commit()
                st.cache_data.clear()
                return True
            return False
    except psycopg2.Error as error:
        st.error(f"Database Update Error: {error}")
        return False
    finally:
        if conn: conn.close()


# --- AI & Web Scraping Functions (Email Finding) ---
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
    keywords = ['contact', 'about', 'impressum', 'kontakt', 'support', 'help', 'legal']
    for a in soup.find_all('a', href=True):
        href = a['href'].lower()
        link_text = a.text.lower()
        if any(keyword in href or keyword in link_text for keyword in keywords):
            full_url = urljoin(base_url, a['href'])
            links.add(full_url)
    return list(links)[:5]

def find_email_with_openai(content, source_url):
    if not content or content.startswith("Error"):
        return None, None, "Could not fetch website content."

    potential_emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", content)
    if not potential_emails:
        return None, None, "No potential email patterns found on page."

    prompt = f"""
    You are a highly accurate data extraction assistant. Your task is to find the single best contact email address from the provided website text.
    **CRITICAL RULES:**
    1.  **DO NOT GUESS OR INVENT an email address.** If no valid email is present in the text, you MUST reply with the single word: "None".
    2.  Analyse the text and identify the most appropriate public-facing contact email (e.g., info@, contact@, hello@, support@, or even a public gmail/outlook if that's what the business uses).
    3.  Your reply MUST be a JSON object with two keys: "email" and "reasoning".
    4.  The "email" key's value should be the single email address you found. If none found, it must be null.
    5.  The "reasoning" key should briefly explain why you chose that email.
    
    **Website Text:**
    "{content[:4000]}"
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        result_json = json.loads(response.choices[0].message.content)
        email = result_json.get("email")

        if email and isinstance(email, str) and re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return email, source_url, None
        else:
            return None, None, "AI analyzed the page but did not find a suitable email."
    except (json.JSONDecodeError, KeyError):
        return None, None, "AI response was not in the expected format."
    except Exception as e:
        return None, None, f"An error occurred with the AI model: {e}"


def intelligently_find_email(base_url):
    stripped_content, full_html = get_website_content(base_url)
    if stripped_content.startswith("Error"):
        return None, None, stripped_content

    email, source_url, error = find_email_with_openai(stripped_content, base_url)
    if email:
        return email, source_url, None

    contact_urls = find_contact_links(base_url, full_html)
    for url in contact_urls:
        content, _ = get_website_content(url)
        if not content.startswith("Error"):
            email, source_url, error = find_email_with_openai(content, url)
            if email:
                return email, source_url, None

    return None, None, "Could not find a valid email on the website or its contact pages."


# --- Email Sending and Generation ---
def send_bulk_email(recipients, subject, body_template):
    success_count, error_list = 0, []
    for r in recipients:
        try:
            personalized_body = body_template.replace('{{name}}', r.get('name', 'there').split(' ')[0])
            html_body = personalized_body.replace('\n', '<br>')
            params = {
                "from": f"Dhruv from UpRez <{st.secrets['email']['sender_email']}>",
                "to": [r["email"]],
                "subject": subject,
                "html": f"<html><body>{html_body}</body></html>"
            }
            email = resend.Emails.send(params)
            if 'id' in email:
                success_count += 1
            else:
                error_list.append(f"{r['name']}: {email.get('message', 'Unknown error')}")
        except Exception as e:
            error_list.append(f"{r['name']}: {str(e)}")
    return success_count, error_list

def generate_email_with_llm(requirement):
    prompt = f"""You are an expert copywriter specializing in B2B outreach... [Your detailed prompt here] ...Compose the email body now."""
    try:
        response = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            st.error("AI failed to generate a response.")
            return None
    except Exception as e:
        st.error(f"An unexpected AI error occurred: {e}")
        return None


# --- BUSINESS SEARCH & ANALYSIS FUNCTIONS ---
def search_nearby_places_sync(params: dict, desired_total_results: int) -> list:
    all_places = []
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    while len(all_places) < desired_total_results:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            all_places.extend(data.get("results", []))
            pagetoken = data.get("next_page_token")
            if not pagetoken or len(all_places) >= desired_total_results:
                break
            params["pagetoken"] = pagetoken
            time.sleep(2)
        except requests.RequestException as e:
            st.error(f"Error calling Google Nearby Search API: {e}")
            break
    return all_places[:desired_total_results]

def get_place_details_sync(place_id: str) -> dict:
    conn = get_db_connection()
    if not conn: return {}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT details_json, fetched_at FROM places_cache WHERE place_id = %s", (place_id,))
            record = cur.fetchone()
            if record:
                details_json, fetched_at = record
                age = datetime.now(timezone.utc) - fetched_at
                if age < timedelta(days=30):
                    st.sidebar.info(f"Cache HIT for place_id: {place_id}")
                    cached_data = details_json
                    cached_data['data_source'] = f"DB Cache (fetched {age.days} days ago)"
                    return cached_data
                st.sidebar.warning(f"Cache STALE for {place_id}. Refetching.")

        st.sidebar.info(f"Cache MISS for {place_id}. Calling Google API.")
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        full_fields = "place_id,name,formatted_address,rating,user_ratings_total,url,formatted_phone_number,website,reviews,opening_hours"
        params = {"key": google_api_key, "place_id": place_id, "fields": full_fields}
        response = requests.get(url, params=params)
        response.raise_for_status()
        result = response.json().get("result", {})

        if result:
            result_json_str = json.dumps(result)
            current_time = datetime.now(timezone.utc)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO places_cache (place_id, details_json, fetched_at) VALUES (%s, %s, %s)
                    ON CONFLICT (place_id) DO UPDATE SET details_json = %s, fetched_at = %s;
                """, (place_id, result_json_str, current_time, result_json_str, current_time))
                conn.commit()
            result['data_source'] = "Live Google API (saved to cache)"
        return result
    except (Exception, psycopg2.Error, requests.RequestException) as e:
        st.error(f"Error in get_place_details_sync for {place_id}: {type(e).__name__} - {e}")
        return {}
    finally:
        if conn: conn.close()

def analyze_reviews_for_pain_points(reviews: list) -> list:
    if not reviews: return []
    pain_points = []
    for review in reviews:
        text = review.get("text")
        if not text: continue
        analysis = TextBlob(text)
        if analysis.sentiment.polarity < -0.1:
            pain_points.append({"sentiment": analysis.sentiment.polarity, "text": text, "author": review.get("author_name")})
    return pain_points

def chat_with_ai_analyst(business_data, user_question, use_lean_payload):
    if use_lean_payload:
        lean_business_data = [
            {"name": b.get("name"), "rating": b.get("rating"), "total_ratings": b.get("user_ratings_total"),
             "pain_points": [p['text'] for p in b.get("pain_points", [])],
             "phone": b.get("formatted_phone_number"), "website": b.get("website"), "address": b.get("formatted_address")}
            for b in business_data]
        data_to_send = json.dumps(lean_business_data, indent=2)
        system_prompt = "Answer ONLY based on the summarized JSON data. If info is not in the data, say it's not available."
    else:
        data_to_send = json.dumps(business_data, indent=2)
        system_prompt = "Answer ONLY based on the complete JSON data. Do not make up information."
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the business data:\n{data_to_send}"},
                {"role": "user", "content": f"Please answer this question: {user_question}"}])
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"An error with OpenAI: {str(e)}")
        return None


# --- Main Streamlit App ---
st.title("AI-Powered Business Finder & Outreach Tool")

# --- Initialize Session State ---
if 'recipients' not in st.session_state: st.session_state.recipients = []
if 'manual_recipients' not in st.session_state: st.session_state.manual_recipients = []
if 'search_results' not in st.session_state: st.session_state.search_results = []
if 'ai_analysis' not in st.session_state: st.session_state.ai_analysis = ""

tab1, tab2 = st.tabs(["🔎 Find & Analyze New Customers", "✉️ Build List & Send Campaigns"])

with tab1:
    st.header("0. Find New Customers")
    st.info("Find local businesses, analyze their online presence, and identify potential leads based on specific criteria.")

    with st.form("search_form"):
        st.subheader("Search Criteria")
        col1, col2 = st.columns(2)
        with col1:
            keyword = st.text_input("Business Type / Keyword*", value="restaurant")
            lat = st.number_input("Latitude*", value=48.8566, format="%.4f")
        with col2:
            radius_m = st.number_input("Search Radius (meters)*", min_value=500, max_value=50000, value=5000, step=500)
            lng = st.number_input("Longitude*", value=2.3522, format="%.4f")

        st.subheader("Filtering Criteria")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            min_rating = st.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.1)
            max_rating = st.slider("Maximum Rating", 0.0, 5.0, 5.0, 0.1)
        with col_f2:
            min_reviews = st.number_input("Minimum Number of Reviews", min_value=0, value=10, step=1)
            has_pain_points = st.checkbox("Must have negative reviews ('Pain Points')")
        with col_f3:
            desired_total_results = st.number_input("Max search results to fetch", min_value=1, max_value=60, value=10)

        submitted = st.form_submit_button("🔍 Search for Businesses", type="primary")

    if submitted:
        with st.spinner("Searching for businesses... This may take a moment."):
            search_params = {"key": google_api_key, "location": f"{lat},{lng}", "radius": radius_m, "keyword": keyword}
            nearby_places = search_nearby_places_sync(search_params, desired_total_results)
            if not nearby_places:
                st.warning("No businesses found matching your initial search criteria.")
            else:
                pre_filtered_places = [p for p in nearby_places if
                                       (p.get("rating", 0) >= min_rating and
                                        p.get("rating", 5) <= max_rating and
                                        p.get("user_ratings_total", 0) >= min_reviews)]
                st.write(f"Found {len(nearby_places)} businesses, {len(pre_filtered_places)} matched pre-filters. Fetching details...")
                final_results = []
                progress_bar = st.progress(0, text="Fetching details and analyzing reviews...")
                for i, place in enumerate(pre_filtered_places):
                    details = get_place_details_sync(place["place_id"])
                    if not details: continue
                    details["pain_points"] = analyze_reviews_for_pain_points(details.get("reviews", []))
                    if has_pain_points and not details["pain_points"]: continue
                    final_results.append(details)
                    progress_bar.progress((i + 1) / len(pre_filtered_places), text=f"Processing {details.get('name', 'N/A')}")
                st.session_state.search_results = final_results
                st.success(f"Search complete! Found {len(final_results)} businesses matching all criteria.")

    if st.session_state.search_results:
        st.divider()
        st.subheader(f"✅ Displaying {len(st.session_state.search_results)} Search Results")
        for res in st.session_state.search_results:
            place_id = res.get("place_id")
            with st.expander(f"**{res.get('name')}** - Rating: {res.get('rating', 'N/A')} ({res.get('user_ratings_total', 0)} reviews)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Address:** {res.get('formatted_address', 'N/A')}")
                    st.write(f"**Phone:** {res.get('formatted_phone_number', 'N/A')}")
                    st.write(f"**Website:** {res.get('website', 'N/A')}")
                    st.markdown(f"**Google Maps:** [View on Map]({res.get('url', '#')})")
                    st.caption(f"Data Source: {res.get('data_source', 'N/A')}")
                    
                    email_placeholder = st.empty()
                    if res.get('email'):
                        email_placeholder.success(f"**Email:** {res.get('email')}")
                    elif res.get('website'):
                        session_key = f"searched_{place_id}"
                        if session_key not in st.session_state:
                            with email_placeholder.container():
                                with st.spinner(f"Scanning {res.get('website')}..."):
                                    email, source_url, error = intelligently_find_email(res.get('website'))
                                    if email:
                                        update_details_in_db(place_id, {'email': email})
                                        st.session_state[session_key] = {'status': 'found', 'email': email, 'source': source_url}
                                        st.rerun()
                                    else:
                                        st.warning(error)
                                        st.session_state[session_key] = {'status': 'not_found', 'error': error}
                        else:
                            result = st.session_state[session_key]
                            if result['status'] == 'found':
                                email_placeholder.success(f"**Email:** {result['email']} (Found at: {result['source']})")
                            else:
                                email_placeholder.warning(result['error'])
                with col2:
                    st.markdown("**Opening Hours:**")
                    opening_hours = res.get("opening_hours", {}).get("weekday_text", [])
                    if opening_hours:
                        for day in opening_hours:
                            st.write(day)
                    else:
                        st.write("Not available.")

                    st.markdown("**Reviews:**")
                    reviews = res.get("reviews", [])
                    if reviews:
                        positive = next((r for r in reviews if r.get('rating', 0) >= 4), None)
                        negative = next((r for r in reviews if r.get('rating', 0) <= 2), None)
                        if positive: st.success(f"**Positive:** \"{positive['text'][:150]}...\" - *{positive['author_name']}*")
                        if negative: st.warning(f"**Negative:** \"{negative['text'][:150]}...\" - *{negative['author_name']}*")
                    else:
                        st.write("No reviews available.")

        st.divider()
        st.subheader("🤖 Chat with AI Analyst")
        with st.form("chat_form"):
            user_question = st.text_input("Your Question", "Which business has the most negative reviews? List its name and website.")
            use_lean = st.checkbox("Use summarized data for AI (cheaper & faster)", value=True)
            if st.form_submit_button("Ask AI") and user_question:
                with st.spinner("AI is analyzing the data..."):
                    st.session_state.ai_analysis = chat_with_ai_analyst(st.session_state.search_results, user_question, use_lean)
        if st.session_state.ai_analysis:
            st.markdown(st.session_state.ai_analysis)

with tab2:
    st.header("1. Build Your Recipient List")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Select from Database")
        customer_data = fetch_data_from_neon_db()
        if not customer_data: st.info("No data in database. Use the 'Find Customers' tab.")
        else:
            for record in customer_data:
                details, place_id = record.get("details", {}), record.get("place_id")
                if not details or not place_id: continue
                with st.expander(f"**{details.get('name', 'N/A')}**"):
                    st.write(f"**Website:** {details.get('website', 'N/A')}")
                    email_placeholder = st.empty()
                    if details.get('email'):
                        email_placeholder.success(f"**Email:** {details.get('email')}")
                        st.checkbox("Add to recipient list", key=f"select_{place_id}")
                    elif details.get('website'):
                        session_key = f"searched_{place_id}"
                        if session_key not in st.session_state:
                            if st.button("Find Email", key=f"find_{place_id}"):
                                with st.spinner(f"Scanning {details.get('website')}..."):
                                    email, source, error = intelligently_find_email(details.get('website'))
                                    st.session_state[session_key] = {'status': 'found' if email else 'not_found', 'email': email, 'source': source, 'error': error}
                                    if email: update_details_in_db(place_id, {'email': email})
                                    st.rerun()
                        else:
                            result = st.session_state[session_key]
                            if result['status'] == 'found':
                                email_placeholder.success(f"**Email:** {result['email']} (Found at: {result['source']})")
                                st.checkbox("Add to recipient list", key=f"select_{place_id}")
                            else:
                                email_placeholder.warning(result['error'])
                    else:
                        email_placeholder.error("No website to search for an email.")
            
            # --- FIX: ROBUST RECIPIENT LIST BUILDING ---
            # 1. Get all items selected via checkbox from the database
            selected_db = [r['details'] for r in customer_data if st.session_state.get(f"select_{r.get('place_id')}")]
            
            # 2. Create a dictionary keyed by name to handle overrides
            final_recipients_dict = {}

            # 3. Add database selections first
            for recipient in selected_db:
                # Use a case-insensitive name as the key
                key = recipient.get('name', '').lower()
                final_recipients_dict[key] = recipient

            # 4. Add/overwrite with manual entries (manual entries take precedence)
            for recipient in st.session_state.manual_recipients:
                key = recipient.get('name', '').lower()
                final_recipients_dict[key] = recipient

            # 5. The final list is the dictionary's values
            st.session_state.recipients = list(final_recipients_dict.values())

    with col2:
        st.subheader("Manually Add or Edit Customer")
        with st.form("manual_form", clear_on_submit=True):
            name = st.text_input("Name*", help="To edit, enter an existing name with a new email.")
            email = st.text_input("Email*")
            if st.form_submit_button("Add / Update in List"):
                if name and email:
                    # Modify the separate manual_recipients list
                    st.session_state.manual_recipients = [
                        r for r in st.session_state.manual_recipients
                        if r.get('name', '').lower() != name.lower()
                    ]
                    st.session_state.manual_recipients.append({"name": name, "email": email, "manual": True})
                    st.rerun()
                else:
                    st.warning("Name and Email are required.")

    st.subheader("Current Recipient List")
    if st.session_state.get('recipients'):
        for r in st.session_state.recipients: st.write(f"- **{r.get('name')}** ({r.get('email')})")
        if st.button("Clear List"):
            st.session_state.recipients = []
            st.session_state.manual_recipients = [] # Also clear the manual list
            keys_to_del = [k for k in st.session_state if k.startswith('select_')]
            for k in keys_to_del: del st.session_state[k]
            st.rerun()
    else: st.info("Your recipient list is empty.")
    
    st.divider()
    st.header("2. Compose and Send Email")
    if 'send_status' in st.session_state:
        s, e = st.session_state.pop('send_status').values()
        if s > 0: st.success(f"Campaign Finished! Successfully sent {s} emails.")
        if e: st.error("Some emails failed:"); [st.write(f"- {err}") for err in e]
    
    if st.session_state.get('recipients'):
        subject = st.text_input("Email Subject", "Quick Question")
        requirement = st.text_area("Email's purpose:", height=150, placeholder="e.g., Pitch our new AI-powered analytics tool...")
        if 'email_body' not in st.session_state:
            if st.button("✨ Generate Email Body with AI", type="primary") and requirement:
                with st.spinner("AI is writing..."):
                    st.session_state.email_body = generate_email_with_llm(requirement)
                    st.rerun()
        else:
            st.subheader("Generated Email Template")
            st.info("Edit the text below. Use {{name}} as a placeholder.")
            st.session_state.email_body = st.text_area("Email Body", st.session_state.email_body, height=300, label_visibility="collapsed")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✨ Regenerate with AI", use_container_width=True):
                    if requirement:
                        with st.spinner("AI is writing a new version..."):
                            st.session_state.email_body = generate_email_with_llm(requirement)
                            st.rerun()
                    else: st.warning("Please describe the email's purpose first.")
            with c2:
                if st.button(f"✅ Send Email to {len(st.session_state.recipients)} Recipients", type="primary", use_container_width=True):
                    with st.spinner("Sending emails..."):
                        s, e = send_bulk_email(st.session_state.recipients, subject, st.session_state.email_body)
                        st.session_state.send_status = {'success': s, 'errors': e}
                        # Clear all lists and selections after sending
                        st.session_state.recipients = []
                        st.session_state.manual_recipients = []
                        del st.session_state['email_body']
                        keys_to_del = [k for k in st.session_state if k.startswith(('select_', 'searched_'))]
                        for k in keys_to_del: del st.session_state[k]
                        st.rerun()
