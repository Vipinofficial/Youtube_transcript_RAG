# youtube_chatmodel.py
import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="YouTube Chat Model", page_icon="🎥", layout="wide")
st.title("🎥 YouTube Chat Model")
st.write("Chat with the content of multiple YouTube videos using Gemini.")

# API Key input
api_key = st.text_input("Enter your Google API Key", type="password")

# Store vectorstore in session
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------------------------
# Helper functions
# ---------------------------
def extract_video_id(url: str):
    """Extract video ID from YouTube URL."""
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return url

def load_transcripts(video_ids):
    """Load transcripts from YouTube videos."""
    docs = []
    failed_videos = []
    for vid in video_ids:
        try:
            loader = YoutubeLoader.from_youtube_url(
                f"https://www.youtube.com/watch?v={vid}",
                add_video_info=True
            )
            video_docs = loader.load()
            if video_docs:
                docs.extend(video_docs)
            else:
                failed_videos.append(vid)
        except Exception as e:
            failed_videos.append(vid)
            st.error(f"❌ Error loading {vid}: {e}")

    if failed_videos:
        st.warning(f"⚠️ No transcripts found for: {', '.join(failed_videos)}")
    return docs

def create_vectorstore(docs, api_key):
    """Create FAISS vectorstore from documents."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# ---------------------------
# YouTube URL input
# ---------------------------
urls = st.text_area("Enter YouTube URLs (one per line)")

# ---------------------------
# Process Videos
# ---------------------------
if st.button("Process Videos"):
    if not api_key or not urls.strip():
        st.error("Please provide both API key and at least one YouTube URL.")
    else:
        os.environ["GOOGLE_API_KEY"] = api_key
        video_ids = [extract_video_id(u.strip()) for u in urls.strip().split("\n") if u.strip()]
        st.write(f"📥 Processing {len(video_ids)} videos...")
        docs = load_transcripts(video_ids)
        if docs:
            st.session_state.vectorstore = create_vectorstore(docs, api_key)
            st.success("✅ Videos processed! You can now chat with them.")
        else:
            st.error("❌ No transcripts could be processed.")

# ---------------------------
# Chat Section
# ---------------------------
if st.session_state.vectorstore:
    query = st.text_input("💬 Ask a question about the videos:")
    if query:
        retriever = st.session_state.vectorstore.as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.2)
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {question}"
        )
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])
        final_prompt = prompt_template.format(context=context, question=query)
        response = llm.invoke(final_prompt)
        st.markdown(f"**Answer:** {response.content}")
