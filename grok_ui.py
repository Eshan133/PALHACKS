import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import Client

# ------------------------------
# File Reading Function (PDF, DOCX, XLSX)
# ------------------------------
def extract_text_from_file(file):
    filename = file.name.lower()
    text = ""
    if filename.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif filename.endswith(".docx"):
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file)
        rows = []
        for _, row in df.iterrows():
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            d = str(row.get("details", "")).strip()
            if q and a:
                entry = f"Q: {q}\nA: {a}"
                if d and d.lower() != "nan":
                    entry += f"\nDetails: {d}"
                rows.append(entry)
        text = "\n\n".join(rows)
    else:
        raise ValueError("Unsupported file type. Only PDF, DOCX, and XLSX are supported.")
    return text

# ------------------------------
# Text Chunking Function
# ----------------------
def get_text_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

# ------------------------------
# Vector Store Functions
# ----------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
KL_VECTOR_DIR = "kl_vector_index"
POLICY_VECTOR_DIR = "policy_vector_index"

def get_vector_store(chunks, db_type):
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    if db_type == "kl":
        vector_dir = KL_VECTOR_DIR
        index_file = "kl_index.faiss"
        chunks_file = "kl_chunks.txt"
    elif db_type == "policy":
        vector_dir = POLICY_VECTOR_DIR
        index_file = "policy_index.faiss"
        chunks_file = "policy_chunks.txt"
    else:
        raise ValueError("Invalid db_type. Use 'kl' or 'policy'.")
    os.makedirs(vector_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(vector_dir, index_file))
    with open(os.path.join(vector_dir, chunks_file), "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.replace("\n", " ") + "\n")

def load_vector_store():
    databases = {"kl": None, "policy": None}
    # Load KL database
    kl_index_path = os.path.join(KL_VECTOR_DIR, "kl_index.faiss")
    kl_chunks_path = os.path.join(KL_VECTOR_DIR, "kl_chunks.txt")
    if os.path.exists(kl_index_path) and os.path.exists(kl_chunks_path):
        index = faiss.read_index(kl_index_path)
        with open(kl_chunks_path, "r", encoding="utf-8") as f:
            chunks = [line.strip() for line in f.readlines()]
        databases["kl"] = {"index": index, "chunks": chunks}
    # Load Policy database
    policy_index_path = os.path.join(POLICY_VECTOR_DIR, "policy_index.faiss")
    policy_chunks_path = os.path.join(POLICY_VECTOR_DIR, "policy_chunks.txt")
    if os.path.exists(policy_index_path) and os.path.exists(policy_chunks_path):
        index = faiss.read_index(policy_index_path)
        with open(policy_chunks_path, "r", encoding="utf-8") as f:
            chunks = [line.strip() for line in f.readlines()]
        databases["policy"] = {"index": index, "chunks": chunks}
    # If both databases are empty, prompt for upload
    if not databases["kl"] and not databases["policy"]:
        st.warning("⚠️ No vector databases found. Please upload and process documents first.")
        uploaded_files = st.file_uploader("Upload PDF, DOCX, or XLSX Files", type=["pdf", "docx", "xlsx"], accept_multiple_files=True, key="reupload")
        db_type = st.selectbox("Select Database", ["Knowledge Library (KL)", "Policy"], key="reupload_db")
        db_type = "kl" if db_type == "Knowledge Library (KL)" else "policy"
        if uploaded_files:
            with st.spinner("Processing uploaded documents..."):
                all_text = ""
                for file in uploaded_files:
                    all_text += extract_text_from_file(file)
                chunks = get_text_chunks(all_text)
                get_vector_store(chunks, db_type)
                st.success(f"Processing complete! Documents added to {db_type.upper()} database.")
        else:
            st.stop()
    return databases

# ------------------------------
# Helper Function to Extract Answers from Chunks
# ------------------------------
def extract_answer_from_chunk(chunk):
    """Extract the answer from a chunk if it follows the Q: ... \nA: ... format."""
    if chunk.startswith("Q:"):
        lines = chunk.split("\n")
        for line in lines:
            if line.startswith("A:"):
                answer = line.replace("A:", "").strip()
                return answer
    first_sentence = chunk.split(".")[0].strip()
    return first_sentence[:100] if first_sentence else "No answer extracted"

# ------------------------------
# Ollama/Gemma Call
# ----------------------
client = Client()

def ask_gemma(context, question, is_batch=False):
    max_context_chars = 2000
    max_history_chars = 1000
    if len(context) > max_context_chars:
        context = context[:max_context_chars]

    history = ""
    if 'chat_history' in st.session_state:
        recent_history = st.session_state.chat_history[-5:]
        for turn in recent_history:
            turn_text = f"User: {turn['question']}\nAssistant: {turn['answer']}\n"
            if len(history) + len(turn_text) <= max_history_chars:
                history += turn_text
            else:
                break

    if is_batch:
        prompt = f"""
You are an assistant answering questions based on the provided document context.
Return a structured response with two parts:
1. Answer: Exactly one of "Yes", "No", or "Not available in the context".
2. Details: A detailed explanation supporting the answer, citing the context where possible.
- For questions that can be answered with a clear yes or no based on the context, use "Yes" or "No".
- For questions where the context does not provide a clear yes/no answer or any relevant information, use "Not available in the context".
- In Details, explain your answer, referencing the context or noting its limitations.
Format your response exactly as:
Answer: [Yes/No/Not available in the context]
Details: [Your detailed explanation]

Document Context:
{context}

Current Question: {question}
"""
    else:
        prompt = f"""
You are an assistant that answers questions based on provided document context and conversation history.
Use the document context to provide accurate answers. Refer to the conversation history to ensure coherence and avoid repetition.
If the answer is not in the context, say "Not available in the context." If the question relates to prior conversation, address it appropriately.

Document Context:
{context}

Conversation History:
{history}

Current Question: {question}

Answer:
"""
    response = client.chat(
        model="gemma3",
        messages=[{"role": "user", "content": prompt}]
    )
    response_text = response['message']['content'].strip()

    if is_batch:
        answer = "Not available in the context"
        details = response_text
        lines = response_text.split("\n")
        answer_found = False
        details_start = 0
        for i, line in enumerate(lines):
            if line.startswith("Answer:"):
                answer_candidate = line.replace("Answer:", "").strip()
                if answer_candidate in ["Yes", "No", "Not available in the context"]:
                    answer = answer_candidate
                    answer_found = True
                if i + 1 < len(lines) and lines[i + 1].startswith("Details:"):
                    details = lines[i + 1].replace("Details:", "").strip()
                    details_start = i + 2
                break
        if answer_found and details_start > 0:
            details += "\n" + "\n".join(lines[details_start:]).strip()
        return answer, details
    return response_text

# ------------------------------
# User Query Handler
# ----------------------
def user_input(user_question):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    databases = load_vector_store()
    question_vector = model.encode([user_question]).astype("float32")
    context = ""
    # Try KL database first
    if databases["kl"]:
        index = databases["kl"]["index"]
        chunks = databases["kl"]["chunks"]
        D, I = index.search(question_vector, k=5)
        contexts = [chunks[i] for i in I[0] if i < len(chunks)]
        context = "\n".join(contexts)
        answer = ask_gemma(context, user_question)
        if answer != "Not available in the context":
            st.session_state.chat_history.append({"question": user_question, "answer": answer, "context": context})
            for turn in st.session_state.chat_history:
                st.markdown(f"**You:** {turn['question']}")
                st.markdown(f"**Gemma:** {turn['answer']}")
            if st.button("🧹 Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
            return
    # If KL has no answer or doesn't exist, try Policy database
    if databases["policy"]:
        index = databases["policy"]["index"]
        chunks = databases["policy"]["chunks"]
        D, I = index.search(question_vector, k=5)
        contexts = [chunks[i] for i in I[0] if i < len(chunks)]
        context = "\n".join(contexts)
        answer = ask_gemma(context, user_question)
    else:
        answer = "Not available in the context"
    st.session_state.chat_history.append({"question": user_question, "answer": answer, "context": context})
    for turn in st.session_state.chat_history:
        st.markdown(f"**You:** {turn['question']}")
        st.markdown(f"**Gemma:** {turn['answer']}")
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ------------------------------
# Streamlit Login UI
# ----------------------
def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    user_type = st.radio("User Type", ["Admin", "Student"])
    if st.button("Login"):
        if user_type == "Admin" and username == "" and password == "":
            st.session_state.logged_in = True
            st.session_state.user_type = "admin"
            st.rerun()
        elif user_type == "Student" and username == "" and password == "":
            st.session_state.logged_in = True
            st.session_state.user_type = "student"
            st.rerun()
        else:
            st.error("Invalid credentials")

# ------------------------------
# Admin Dashboard
# ----------------------
def admin_dashboard():
    st.header("Chat with Files using Gemma 🧠")
    user_question = st.text_input("Ask a question:")
    if user_question:
        user_input(user_question)
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.rerun()
    with st.sidebar:
        st.title("Menu:")
        db_type = st.selectbox("Select Database", ["Knowledge Library (KL)", "Policy"])
        db_type = "kl" if db_type == "Knowledge Library (KL)" else "policy"
        uploaded_files = st.file_uploader(f"Upload PDF, DOCX, or XLSX Files for {db_type.upper()} Database", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not uploaded_files:
                st.error("Please upload at least one file.")
            else:
                with st.spinner("Processing..."):
                    all_text = ""
                    for file in uploaded_files:
                        all_text += extract_text_from_file(file)
                    chunks = get_text_chunks(all_text)
                    get_vector_store(chunks, db_type)
                    st.success(f"Processing complete! Documents added to {db_type.upper()} database.")

# ------------------------------
# Student Dashboard
# ----------------------
def student_dashboard():
    st.header("Chat with Files using Gemma 🧠")
    st.subheader("Batch QnA Processor")
    delay = st.slider("Set delay between questions (seconds)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
    batch_file = st.file_uploader("Upload .xlsx file with 'Questions' column", type=["xlsx"], key="student_batch")
    if batch_file and st.button("Run Batch QnA", key="run_student_batch"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        with st.spinner("Answering questions..."):
            df = pd.read_excel(batch_file)
            questions = df['Questions'].dropna().tolist()
            if not questions:
                st.error("No questions found in the uploaded file.")
                return
            results = []
            databases = load_vector_store()
            import time
            for i, q in enumerate(questions):
                time.sleep(delay)
                q_vector = model.encode([q]).astype("float32")
                context = ""
                answer = "Not available in the context"
                details = ""
                score = None
                candidate_chunks = []
                source = "None"
                # Try KL database first
                if databases["kl"]:
                    index = databases["kl"]["index"]
                    chunks = databases["kl"]["chunks"]
                    D, I = index.search(q_vector, k=5)
                    contexts = [chunks[idx] for idx in I[0] if idx < len(chunks)]
                    if contexts:
                        context = "\n".join(contexts)
                        answer, details = ask_gemma(context, q, is_batch=True)
                        score = float(D[0][0]) if D.size > 0 else None
                        candidate_chunks = [(ctx, idx) for ctx, idx in zip(contexts, I[0]) if idx < len(chunks)]
                    if answer != "Not available in the context":
                        source = "KL"
                        # KL provided an answer, proceed with candidate answers
                        candidate_answers = []
                        if answer in ["Yes", "No"]:
                            for chunk, _ in candidate_chunks:
                                candidate = extract_answer_from_chunk(chunk)
                                if candidate and candidate not in candidate_answers:
                                    candidate_answers.append(candidate)
                        candidate_answers_str = ", ".join(candidate_answers[:3]) if candidate_answers else "None"
                        results.append({
                            "Question": q,
                            "Answer": answer,
                            "Details": details,
                            "FAISS Score": score,
                            "Candidate Answers": candidate_answers_str,
                            "Source": source
                        })
                        progress_value = (i + 1) / len(questions)
                        progress_bar.progress(min(progress_value, 1.0))
                        status_text.text(f"Processing question {i + 1} of {len(questions)}")
                        continue
                # If KL has no answer or doesn't exist, try Policy database
                if databases["policy"]:
                    index = databases["policy"]["index"]
                    chunks = databases["policy"]["chunks"]
                    D, I = index.search(q_vector, k=5)
                    contexts = [chunks[idx] for idx in I[0] if idx < len(chunks)]
                    if contexts:
                        context = "\n".join(contexts)
                        answer, details = ask_gemma(context, q, is_batch=True)
                        score = float(D[0][0]) if D.size > 0 else None
                        candidate_chunks = [(ctx, idx) for ctx, idx in zip(contexts, I[0]) if idx < len(chunks)]
                    if answer != "Not available in the context":
                        source = "Policy"
                # Process candidate answers
                candidate_answers = []
                if answer in ["Yes", "No"]:
                    for chunk, _ in candidate_chunks:
                        candidate = extract_answer_from_chunk(chunk)
                        if candidate and candidate not in candidate_answers:
                            candidate_answers.append(candidate)
                candidate_answers_str = ", ".join(candidate_answers[:3]) if candidate_answers else "None"
                results.append({
                    "Question": q,
                    "Answer": answer,
                    "Details": details,
                    "FAISS Score": score,
                    "Candidate Answers": candidate_answers_str,
                    "Source": source
                })
                progress_value = (i + 1) / len(questions)
                progress_bar.progress(min(progress_value, 1.0))
                status_text.text(f"Processing question {i + 1} of {len(questions)}")
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "batch_qna_results.csv", "text/csv")
    user_question = st.text_input("Ask a question:")
    if user_question:
        user_input(user_question)
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.rerun()

# ------------------------------
# Main
# ----------------------
def main():
    if not st.session_state.get("logged_in", False):
        login()
        return
    if st.session_state.user_type == "admin":
        admin_dashboard()
    elif st.session_state.user_type == "student":
        student_dashboard()

if __name__ == "__main__":
    main()