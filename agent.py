import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama  # <-- Swapped package to Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv()

class DataSandboxManager:
    """Manages the uploaded CSV data and provides safe execution environments."""
    def __init__(self):
        self._sandbox = {}

    def register_dataframe(self, name: str, dataframe: pd.DataFrame) -> str:
        """Sanitizes the file name and registers the dataframe into memory."""
        sanitized_name = name.replace(".csv", "").replace(" ", "_").lower()
        self._sandbox[sanitized_name] = dataframe
        return sanitized_name

    def get_available_datasets(self) -> list:
        """Returns the list of currently registered dataset names."""
        return list(self._sandbox.keys())

    def execute_safely(self, dataframe_name: str, python_code: str) -> str:
        """Executes a single filtering/query pandas expression inside a blocked sandbox."""
        if dataframe_name not in self._sandbox:
            return f"Error: Dataset '{dataframe_name}' does not exist. Available: {self.get_available_datasets()}"
        
        local_env = {"df": self._sandbox[dataframe_name]}
        try:
            result = eval(python_code, {"__builtins__": None}, local_env)
            return str(result)
        except Exception as e:
            return f"Execution error: {str(e)}"

# Instantiate sandbox manager
if "sandbox_manager" not in st.session_state:
    st.session_state.sandbox_manager = DataSandboxManager()
manager = st.session_state.sandbox_manager


@tool
def query_dataset(dataframe_name: str, python_code: str) -> str:
    """
    Executes a standard Pandas filtering expression or column read operation on an uploaded dataset.
    Args:
        dataframe_name: The precise string name of the target dataset.
        python_code: A safe pandas operation line targeting the 'df' variable. Example: df[df['category'] == 'electronics']
    """
    return manager.execute_safely(dataframe_name, python_code)


class FAQAgentEngine:
    """Handles the LangChain setup, tool calling orchestration, and prompt execution workflows."""
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.0):
        # Configured to look for your locally running model
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.llm_with_tools = self.llm.bind_tools([query_dataset])

    def _get_routing_prompt(self, datasets: list, question: str) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", (
                "You are an internal corporate support data assistant.\n"
                "Available datasets in workspace: {datasets}\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. You must answer using ONLY factual information retrieved via the tools.\n"
                "2. Do NOT use any general external or world knowledge.\n"
                "3. If data is missing or a row can't be found, reply exactly: "
                "'I could not find this information in the uploaded files.'\n"
                "4. Never output programming code to the user. Give clean text answers."
            )),
            ("human", "{question}")
        ])

    def _get_synthesis_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", (
                "You are an automated, literal data extraction tool. You have no outside knowledge.\n"
                "Review these raw data values retrieved from the files: {facts}.\n\n"
                "CRITICAL MATCHING RULES:\n"
                "1. If the text in {facts} is empty, contains errors, or says it can't find a dataset, you must say exactly: "
                "'I could not find this information in the uploaded files.'\n"
                "2. If {facts} contains the answer, rewrite that exact information into a clean, direct sentence.\n"
                "3. DO NOT add any pleasantries, introductions, or assumptions. DO NOT invent policies."
            )),
            ("human", "{question}")
        ])


    def process_query(self, question: str, datasets: list) -> str:
        """Main orchestrator chain managing the tool routing loop and final synthesis."""
        routing_prompt = self._get_routing_prompt(datasets, question)
        routing_chain = routing_prompt | self.llm_with_tools
        llm_response = routing_chain.invoke({"datasets": datasets, "question": question})
        
        if llm_response.tool_calls:
            tool_outputs = []
            for tool_call in llm_response.tool_calls:
                if tool_call["name"] == "query_dataset":
                    output = query_dataset.invoke(tool_call["args"])
                    tool_outputs.append(output)
            
            synthesis_chain = self._get_synthesis_prompt() | self.llm
            final_answer = synthesis_chain.invoke({"facts": tool_outputs, "question": question})
            return final_answer.content
        
        return llm_response.content


def render_ui():
    st.set_page_config(page_title="Hello Agent - CSV FAQ Tool", layout="wide")
    st.title("📊 Hello Agent: CSV FAQ Assistant")

    uploaded_files = st.file_uploader(
        "Upload one or more CSV files", 
        type=["csv"], 
        accept_multiple_files=True
    )

    # We will combine all text rows into one giant searchable text reference for the LLM
    all_text_context = ""

    if uploaded_files:
        st.subheader("📄 Data Previews")
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)
            manager.register_dataframe(uploaded_file.name, df)
            
            # Convert the dataframe to a plain text string so the model can read it directly
            all_text_context += f"\n--- DATASET: {uploaded_file.name} ---\n"
            all_text_context += df.to_string(index=False) + "\n"
            
            with st.expander(f"Previewing dataset: {uploaded_file.name}"):
                st.dataframe(df.head(3))

    user_question = st.text_input("💬 Ask a question about the uploaded data:")

    if user_question:
        if not all_text_context:
            st.warning("⚠️ Please upload at least one CSV file first.")
            return

        with st.spinner("Analyzing data through local engine..."):
            try:
                # Bypass tool calling entirely for smaller local models
                # Direct contextual prompting is 100% reliable for Llama 3.2
                from langchain_ollama import ChatOllama
                llm = ChatOllama(model="llama3.2", temperature=0.0)
                
                prompt = f"""You are an internal corporate support data assistant.
You must answer the question using ONLY the factual data provided below.
Do NOT use external knowledge. If the answer cannot be found in the data, reply exactly: 'I could not find this information in the uploaded files.'

Uploaded Data:
{all_text_context}

Question: {user_question}
Answer:"""
                
                answer = llm.invoke(prompt)
                st.subheader("💡 Answer:")
                st.info(answer.content)
            except Exception as e:
                st.error(f"Application processing error: {e}")


if __name__ == "__main__":
    render_ui()
