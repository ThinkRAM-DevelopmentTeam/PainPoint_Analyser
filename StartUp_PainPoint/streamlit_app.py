import os
import pandas as pd
import streamlit as st
import re
from textblob import TextBlob
from datetime import datetime
import io
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# == UI Constants ==
PRIMARY_COLOR = "#0061a8"
ACCENT_COLOR = "#8dcefa"
HEADER_ICON = "🩹"
PAINPOINT_ICON = "🧩"
CHAT_ICON = "🤖"
SUMMARY_ICON = "📝"
DOWNLOAD_ICON = "⬇️"

def initialize_state():
    defaults = {
        'scraped_df': pd.DataFrame(),
        'chat_history': [],
        'last_answer_sources': [],
        'vectorizer': None,
        'X': None,
        'corpus': [],
        'last_summary': None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def sidebar_config():
    with st.sidebar:
        st.header("Settings")
        st.markdown("Adjust what to scrape, pain keywords and more.")
        with st.expander("Reddit Scraper"):
            default_subs = [
                'startups', 'entrepreneur', 'Entrepreneurship', 'StartupAccelerators',
                'venturecapital', 'crowdfunding', 'startup_resources', 'smallbusiness',
                'SideProject', 'freelance', 'Startup_Ideas'
            ]
            subreddits_text = st.text_area("Subreddits", value=", ".join(default_subs), height=80)
            time_filter = st.selectbox("Reddit time filter", ['day', 'week', 'month', 'year', 'all'], index=2)
            post_limit = st.slider("Posts per subreddit", min_value=10, max_value=500, value=150, step=10)
            output_rows = st.slider("Table rows to show", min_value=10, max_value=2000, value=150, step=10)
            pain_keywords = st.text_area(
                "Pain keywords (comma-separated)",
                value=", ".join([
                    'churn','onboarding','pmf','fundraising','runway',
                    'technical debt','scaling','lead generation','invoice','bug'
                ]), height=35,
            )
        with st.expander("AI & Gemini"):
            use_llm = st.toggle("Enable Gemini AI chat", value=True)
            llm_model = st.text_input("Gemini model", value="gemini-2.5-pro")
            gemini_api_key = st.text_input("GEMINI_API_KEY", type="password")
            ai_summary = st.checkbox("AI summarize first N posts", value=False, help="Get a Gemini summary of first posts")
            num_posts = st.number_input("How many posts to summarize?", min_value=3, max_value=50, value=10, step=1, key="num_posts_summarize")
        with st.expander("Reddit API auth", expanded=False):
            reddit_client_id = st.text_input("REDDIT_CLIENT_ID", value=os.getenv('REDDIT_CLIENT_ID', ""))
            reddit_client_secret = st.text_input("REDDIT_CLIENT_SECRET", value=os.getenv('REDDIT_CLIENT_SECRET', ""), type="password")
            reddit_user_agent = st.text_input("REDDIT_USER_AGENT", value=os.getenv('REDDIT_USER_AGENT', "startup_pain_points_app/1.0"))
            reddit_username = st.text_input("Reddit username (optional)")
            reddit_password = st.text_input("Reddit password (optional)", type="password")
        with st.expander("Help & About", expanded=False):
            st.markdown("* After scraping, you can chat, summarize or download results. Your keys/private data never leave your machine.")
        run = st.button("🌐 Scrape Reddit")
    return (
        subreddits_text, time_filter, post_limit, output_rows, pain_keywords,
        use_llm, llm_model, gemini_api_key, ai_summary, num_posts,
        reddit_client_id, reddit_client_secret, reddit_user_agent,
        reddit_username, reddit_password, run
    )

def extract_pain_points(text, pain_regexes):
    if not text: return []
    hits = [rx.pattern.strip('\\b') for rx in pain_regexes if rx.search(text)]
    return sorted(set(hits))

def analyze_sentiment(text):
    score = TextBlob(text or "").sentiment.polarity
    if score > 0.2: return 'Positive'
    if score < -0.2: return 'Negative'
    return 'Neutral'

def calculate_engagement(post):
    try:
        return float(getattr(post, 'upvote_ratio', 0)) * float(post.score) * float(post.num_comments)
    except Exception:
        return float(post.score) + float(post.num_comments)

def scrape_subreddits(config):
    (
        subreddits_text, time_filter, post_limit, pain_keywords,
        reddit_client_id, reddit_client_secret, reddit_user_agent,
        reddit_username, reddit_password, status_box, progress
    ) = config
    import praw
    subreddits = [s.strip() for s in subreddits_text.split(',') if s.strip()]
    keywords = [k.strip() for k in pain_keywords.split(',') if k.strip()]
    pain_regexes = [re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in keywords]
    reddit_kwargs = {
        "client_id": reddit_client_id.strip(),
        "client_secret": reddit_client_secret.strip(),
        "user_agent": reddit_user_agent.strip() or "startup_pain_points_app/1.0",
    }
    if reddit_username.strip() and reddit_password.strip():
        reddit_kwargs.update(username=reddit_username.strip(), password=reddit_password.strip())
    reddit = praw.Reddit(**reddit_kwargs)
    _ = reddit.read_only
    rows = []
    total = len(subreddits)
    for idx, name in enumerate(subreddits, 1):
        status_box.info(f"Scraping r/{name} ({idx}/{total})...")
        try:
            for post in reddit.subreddit(name).top(time_filter=time_filter, limit=int(post_limit)):
                combined = (post.title or "") + "\n\n" + (post.selftext or "")
                pains = extract_pain_points(combined, pain_regexes)
                rows.append({
                    'subreddit': name,
                    'title': post.title,
                    'content': post.selftext,
                    'permalink': f"https://www.reddit.com{post.permalink}",
                    'created_utc': pd.to_datetime(post.created_utc, unit='s'),
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'upvote_ratio': getattr(post, 'upvote_ratio', None),
                    'engagement_score': calculate_engagement(post),
                    'sentiment': analyze_sentiment(combined),
                    'pain_points': pains,
                    'pain_point_count': len(pains)
                })
        except Exception as e:
            status_box.warning(f"Failed scraping r/{name}: {e}")
        progress.progress(idx / total, text=f"Scraped {idx}/{total} subreddits")
    return pd.DataFrame(rows)

def update_vectorizer_and_corpus(df):
    if not df.empty:
        corpus = (df['title'].fillna('') + "\n\n" + df['content'].fillna('')).tolist()
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')
        X = vectorizer.fit_transform(corpus)
        st.session_state.corpus = corpus
        st.session_state.vectorizer = vectorizer
        st.session_state.X = X
    else:
        st.session_state.corpus = []
        st.session_state.vectorizer = None
        st.session_state.X = None

def prepare_download_files(df, time_filter):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"startup_pain_points_{time_filter}_{timestamp}"
    export_df = df.copy()
    if not export_df.empty:
        export_df['pain_points_joined'] = export_df['pain_points'].apply(lambda v: "; ".join(v))
    csv_bytes = export_df.to_csv(index=False).encode('utf-8') if not export_df.empty else b''
    json_str = export_df.to_json(orient='records', lines=False, date_format='iso') if not export_df.empty else "[]"
    json_bytes = json_str.encode('utf-8')
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w") as zf:
        zf.writestr(base_filename + ".csv", csv_bytes)
        zf.writestr(base_filename + ".json", json_bytes)
    mem_zip.seek(0)
    return csv_bytes, json_bytes, mem_zip, base_filename

def horizontal_download_buttons(csv_bytes, json_bytes, mem_zip, base_filename):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📄 CSV", data=csv_bytes, file_name=f"{base_filename}.csv",
            mime="text/csv", use_container_width=True, help="Download CSV"
        )
    with col2:
        st.download_button(
            "📝 JSON", data=json_bytes, file_name=f"{base_filename}.json",
            mime="application/json", use_container_width=True, help="Download JSON"
        )
    with col3:
        st.download_button(
            "🗜️ ZIP", data=mem_zip, file_name=f"{base_filename}.zip",
            mime="application/zip", use_container_width=True, help="Download ZIP (CSV + JSON)"
        )

def ai_summarize_posts(df, num_posts, model, api_key):
    """Use Gemini to summarize the first num_posts posts."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.prompts import ChatPromptTemplate
        selected = df.head(num_posts)
        # Form context from titles and first part of content
        context = ""
        for idx, row in selected.iterrows():
            context += f"r/{row['subreddit']} | Title: {row['title']}\nContent: {row['content'][:500]}\n\n"
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.2,
            transport="rest"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following Reddit posts from founders and entrepreneurs. Highlight pain points, common themes, and anything that stands out in a concise manner."),
            ("human", "{context}\n\nSummary:"),
        ])
        chain = prompt | llm
        response = chain.invoke({"context": context})
        return response.content
    except Exception as e:
        return f"AI summarization error: {e}"

def show_example_questions():
    with st.expander("💡 See example questions", expanded=False):
        st.markdown(
            "- What are the top challenges founders mention?\n"
            "- Are there frequent issues with onboarding?\n"
            "- What funding obstacles are trending this week?\n"
            "- What's the mood or sentiment of most high-engagement posts?\n"
            "- What technical pain points repeat?"
        )

def handle_ai_summary_and_chat(ai_summary, num_posts, use_llm, llm_model, gemini_api_key, df):
    st.markdown(f"## {SUMMARY_ICON} AI Summarization and {CHAT_ICON} Chat")
    st.caption("Get a quick summary of key pain points, then ask questions interactively about the scraped data.")
    # AI SUMMARIZATION
    if ai_summary:
        st.info(f"Gemini will now generate a concise summary of the first {num_posts} posts:")
        if gemini_api_key and not df.empty:
            with st.spinner("AI is summarizing posts..."):
                summary = ai_summarize_posts(df, num_posts, llm_model, gemini_api_key)
                st.session_state.last_summary = summary
                st.markdown(f"**{SUMMARY_ICON} Gemini Summary:**\n\n{summary}")
        elif not gemini_api_key:
            st.warning("Enter your Gemini API key to use AI summarization.")
        elif df.empty:
            st.warning("Nothing to summarize yet. Scrape Reddit first.")
    # --- AI CHAT --- #
    st.divider()
    st.markdown(f"#### {CHAT_ICON} Interactive Q&A")
    show_example_questions()
    if st.session_state.corpus:
        st.info(f"Knowledge base ready with {len(st.session_state.corpus)} posts. Try chatting!")
    else:
        st.warning("Knowledge base is empty. Run scraping first.")
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)
    user_q = st.chat_input("Ask about founder challenges, validation, tech, etc.")
    if user_q:
        st.session_state.chat_history.append(("user", user_q))
        with st.chat_message("user"):
            st.markdown(user_q)
        answer = ""
        source_keywords = ['source', 'link', 'where', 'post', 'reference']
        if any(keyword in user_q.lower() for keyword in source_keywords) and st.session_state.last_answer_sources:
            lines = ["Here are the sources for my previous answer:"]
            for m in st.session_state.last_answer_sources:
                lines.append(f"- **r/{m['subreddit']}**: [{m['title']}]({m['permalink']}) (Score: {m['score']})")
            answer = "\n".join(lines)
        elif not st.session_state.corpus:
            answer = "The knowledge base is empty. Please run a scrape first."
        elif not use_llm or not gemini_api_key:
            answer = "Enable Gemini AI and provide your API key for chat."
        else:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langchain.prompts import ChatPromptTemplate
                q_vec = st.session_state.vectorizer.transform([user_q.strip()])
                sims = cosine_similarity(q_vec, st.session_state.X).ravel()
                best_indices = sims.argsort()[::-1][:5]
                relevant_indices = [i for i in best_indices if sims[i] > 0.1]
                if not relevant_indices:
                    answer = "No relevant information found. Try a different question."
                else:
                    context = ""
                    sources = []
                    for idx in relevant_indices:
                        post = df.iloc[idx]
                        context += f"--- Post from r/{post['subreddit']} ---\nTitle: {post['title']}\nContent: {post['content'][:1500]}\n\n"
                        sources.append(post.to_dict())
                    st.session_state.last_answer_sources = sources
                    llm = ChatGoogleGenerativeAI(
                        model=llm_model,
                        google_api_key=gemini_api_key,
                        temperature=0.2,
                        transport="rest"
                    )
                    prompt = ChatPromptTemplate.from_messages([
                        ("system",
                         "You are a helpful startup analyst. Answer the user's question based only on the Reddit post context. Synthesize info, avoid quoting. If the context does not answer, say so clearly."
                        ),
                        ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"),
                    ])
                    chain = prompt | llm
                    response = chain.invoke({"context": context, "question": user_q})
                    answer = response.content + "\n\n*Say 'sources' for citations.*"
            except ImportError:
                answer = "Required AI libraries missing. Try: pip install langchain-google-genai google-generativeai"
            except Exception as e:
                answer = f"AI answer error: {e}"
        st.session_state.chat_history.append(("assistant", answer))
        st.rerun()

def main():
    st.set_page_config(
        page_title="Startup Pain Points AI",
        page_icon=HEADER_ICON,
        layout="wide"
    )
    st.markdown(
        f"""
        <style>
            [data-testid="stHeader"] {{
                background: linear-gradient(90deg, {PRIMARY_COLOR}, {ACCENT_COLOR});
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title(f"{HEADER_ICON} Startup Pain Points AI")
    st.caption("Discover founder challenges and market pain points. Use Gemini AI to summarize and explore insights ⚡️.")

    initialize_state()
    (
        subreddits_text, time_filter, post_limit, output_rows, pain_keywords,
        use_llm, llm_model, gemini_api_key, ai_summary, num_posts,
        reddit_client_id, reddit_client_secret, reddit_user_agent,
        reddit_username, reddit_password, run
    ) = sidebar_config()

    status_box = st.empty()
    table_box = st.empty()
    downloads_box = st.container()
    ai_box = st.container()

    if run:
        status_box.info("Scraping in progress...")
        try:
            progress = st.progress(0, text="Scraping subreddits...")
            config = (
                subreddits_text, time_filter, post_limit, pain_keywords,
                reddit_client_id, reddit_client_secret, reddit_user_agent,
                reddit_username, reddit_password, status_box, progress
            )
            df = scrape_subreddits(config)
            status_box.success("Scraping finished.")
        except Exception as e:
            st.error(f"Reddit auth or scraping error: {e}")
            st.stop()
        if not df.empty:
            df['has_pain'] = df['pain_point_count'] > 0
            df = df.sort_values(['has_pain', 'engagement_score'], ascending=[False, False]).reset_index(drop=True)
            st.session_state.scraped_df = df
            update_vectorizer_and_corpus(df)
        else:
            st.session_state.scraped_df = pd.DataFrame()
            update_vectorizer_and_corpus(pd.DataFrame())
        display_cols = [
            'subreddit', 'title', 'permalink', 'created_utc', 'score', 'num_comments',
            'engagement_score', 'sentiment', 'pain_point_count', 'pain_points'
        ]
        table_box.markdown(f"### {PAINPOINT_ICON} Pain Points Table")
        table_box.dataframe(st.session_state.scraped_df[display_cols].head(int(output_rows)), use_container_width=True)
        with downloads_box:
            csv_bytes, json_bytes, mem_zip, base_filename = prepare_download_files(st.session_state.scraped_df, time_filter)
            st.markdown("#### " + DOWNLOAD_ICON + " Download Results")
            horizontal_download_buttons(csv_bytes, json_bytes, mem_zip, base_filename)
            st.caption("All data processed locally. Your keys/data are never sent to any server.")

    with ai_box:
        df_for_chat = st.session_state.scraped_df
        handle_ai_summary_and_chat(ai_summary, num_posts, use_llm, llm_model, gemini_api_key, df_for_chat)

if __name__ == "__main__":
    main()
