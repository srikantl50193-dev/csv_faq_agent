"""Hello Agent - CSV FAQ Assistant (Streamlit + LangChain pandas dataframe agent).

Run locally:  streamlit run app.py
"""
import io
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
LOG_FILE = "qa_log.csv"
HISTORY_TURNS = 3      # past Q&A pairs passed along so follow-up questions work
MAX_AGENT_STEPS = 8    # lets the agent retry a failed query, but not loop forever

# Step 5: the system prompt. {count} and {file_map} are filled in by build_agent();
# keep the rest of this text free of curly braces.
SYSTEM_PROMPT = """You are a strict data assistant for internal company FAQ and policy files.
You have {count} pandas dataframe(s), named df1, df2, ... They come from these files:
{file_map}

RULES:
1. Always inspect the dataframes with pandas before answering. Never answer from general knowledge.
2. Search text case-insensitively, for example:
   df1[df1['Question'].str.contains('return', case=False, na=False)]
   If you are unsure which dataframe holds the answer, search all of them.
3. When returning text, prefer columns named like Answer, Policy, Description, or Details.
4. For numeric questions, compute the result with pandas (sum, mean, count, min, max). Never estimate.
5. Answer in simple English and quote the actual values from the data.
6. End your answer with a line in this form: Source: <file name>, row <row index>
7. If nothing in the data answers the question, reply exactly:
   I could not find this information in the uploaded files.
"""


# ---------- Data loading (Step 4) ----------

@st.cache_data(show_spinner=False)
def load_csv(data: bytes) -> pd.DataFrame:
    """Read CSV bytes, tolerating common encodings, and tidy column names."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    df.columns = df.columns.astype(str).str.strip()
    return df


# ---------- Agent (Steps 6-7) ----------

def build_agent(api_key: str, dfs: list, names: list):
    llm = ChatOpenAI(model=MODEL, temperature=0, api_key=api_key)
    file_map = "\n".join(f"- df{i}: {name}" for i, name in enumerate(names, start=1))
    prefix = SYSTEM_PROMPT.format(count=len(dfs), file_map=file_map)
    # LangChain calls .format() on the prefix again, so escape any braces
    # (e.g. from file names) to keep them as literal text.
    prefix = prefix.replace("{", "{{").replace("}", "}}")

    return create_pandas_dataframe_agent(
        llm,
        dfs,                          # always a list, so names are df1, df2, ...
        agent_type="tool-calling",
        prefix=prefix,                # system prompt goes in as a real system message
        allow_dangerous_code=True,    # required: the agent executes pandas code. Run locally only.
        return_intermediate_steps=True,
        max_iterations=MAX_AGENT_STEPS,
        agent_executor_kwargs={"handle_parsing_errors": True},
        verbose=True,                 # prints the agent's steps in your terminal
    )


def build_query(question: str, history: list) -> str:
    """Add the last few exchanges so follow-ups like 'what about gold?' make sense."""
    recent = history[-HISTORY_TURNS:]
    if not recent:
        return question
    context = "\n".join(f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent)
    return (
        "Earlier conversation (use only to understand the question, not as a data source):\n"
        f"{context}\n\nCurrent question: {question}"
    )


# ---------- Stretch ideas: show evidence, log Q&A ----------

def render_steps(steps: list) -> None:
    """Show the pandas code the agent ran and the rows it got back."""
    if not steps:
        return
    with st.expander("🔍 How the agent found this"):
        for action, observation in steps:
            tool_input = action.tool_input
            code = tool_input.get("query", tool_input) if isinstance(tool_input, dict) else tool_input
            st.code(str(code), language="python")
            st.text(str(observation)[:2000])


def log_qa(question: str, answer: str, files: list) -> None:
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "files": "; ".join(files),
        "question": question,
        "answer": answer,
    }])
    row.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)


# ---------- UI (Step 3) ----------

def main() -> None:
    st.set_page_config(page_title="Hello Agent - CSV FAQ", page_icon="📊", layout="wide")
    st.title("📊 Hello Agent: CSV FAQ Assistant")
    st.caption("Upload FAQ or policy CSV files and ask questions. Answers come only from your files.")

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", "")
        )
        show_steps = st.toggle("Show how the agent found each answer", value=True)
        if st.button("🧹 Clear chat"):
            st.session_state.history = []

    uploaded_files = st.file_uploader(
        "Upload one or more CSV files", type=["csv"], accept_multiple_files=True
    )

    # Rebuilt from the uploader on every run, so removed files disappear too.
    datasets = {}
    for uploaded in uploaded_files or []:
        try:
            datasets[uploaded.name] = load_csv(uploaded.getvalue())
        except Exception as e:
            st.error(f"Could not read {uploaded.name}: {e}")

    if datasets:
        st.subheader("📄 Data previews")
        for name, df in datasets.items():
            with st.expander(f"{name}  ({len(df)} rows, {len(df.columns)} columns)"):
                st.dataframe(df.head(5), width="stretch")

        choice = st.selectbox("Which file should the agent search?", ["All files", *datasets])
        names = list(datasets) if choice == "All files" else [choice]
        dfs = [datasets[n] for n in names]

    st.divider()

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if show_steps:
                render_steps(turn["steps"])

    ready = bool(api_key) and bool(datasets)
    if not api_key:
        st.info("Enter your OpenAI API key in the sidebar to start.")
    elif not datasets:
        st.info("Upload at least one CSV file to start.")

    question = st.chat_input("Ask a question about your files...", disabled=not ready)
    if not question or not question.strip():
        return
    question = question.strip()

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your files..."):
            try:
                agent = build_agent(api_key, dfs, names)
                result = agent.invoke(
                    {"input": build_query(question, st.session_state.history)}
                )
                answer = result["output"]
                steps = result.get("intermediate_steps", [])
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                return
        st.write(answer)
        if show_steps:
            render_steps(steps)

    st.session_state.history.append({"question": question, "answer": answer, "steps": steps})
    log_qa(question, answer, names)


if __name__ == "__main__":
    main()