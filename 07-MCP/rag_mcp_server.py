import os
import glob
import sqlite3
import pandas as pd
from dotenv import load_dotenv

from fastmcp import FastMCP
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

DATA_FOLDER = os.getenv("DATA_FOLDER", "test")

# Initialize MCP server
mcp = FastMCP(name="RAG Agent Server")

class RAGSystem:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.vectorstore = None
        self.retriever = None
        self.sqlite_connection = None
        self.table_info = {}
        self.schema_description = ""
        self.sql_schema = ""
        self._initialize()

    def _initialize(self):
        # Discover files
        csv_files = glob.glob(os.path.join(self.data_folder, "**/*.csv"), recursive=True)
        excel_files = glob.glob(os.path.join(self.data_folder, "**/*.xlsx"), recursive=True)
        excel_files += glob.glob(os.path.join(self.data_folder, "**/*.xls"), recursive=True)
        all_files = csv_files + excel_files

        if not all_files:
            print(f"[WARNING] No data files found in folder: {self.data_folder}")

        all_documents = []
        all_file_info = []

        for file_path in all_files:
            docs, info = self._create_documents_from_file(file_path)
            if docs:
                all_documents.extend(docs)
                all_file_info.append(info)

        # Build schema description by file
        self.schema_description = "Available Data:\n\n"
        for info in all_file_info:
            filename = os.path.basename(info["file_path"])
            self.schema_description += f"File: {filename}\n"
            self.schema_description += f"Columns: {', '.join(info['columns'])}\n"
            self.schema_description += f"Rows: {info['total_rows']}\n\n"

        # Build vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.from_documents(documents=all_documents, embedding=embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 7})

        # Build in-memory SQLite from files
        self.sqlite_connection = sqlite3.connect(":memory:", check_same_thread=False)
        for file_path in all_files:
            df = self._load_file(file_path)
            if df is not None:
                table_name = os.path.splitext(os.path.basename(file_path))[0] \
                                .replace(" ", "_").replace("-", "_")
                df.to_sql(table_name, self.sqlite_connection, if_exists="replace", index=False)
                self.table_info[table_name] = df.columns.tolist()

        # Build SQL schema description
        self.sql_schema = "Tables:\n\n"
        for table_name, cols in self.table_info.items():
            self.sql_schema += f"Table: {table_name}\n"
            self.sql_schema += f"Columns: {', '.join(cols)}\n\n"

    def _load_file(self, file_path):
        try:
            if file_path.endswith(".csv"):
                return pd.read_csv(file_path)
            elif file_path.endswith((".xlsx", ".xls")):
                return pd.read_excel(file_path)
            else:
                return None
        except Exception as e:
            print(f"[ERROR] Failed loading file {file_path}: {e}")
            return None

    def _get_file_info(self, dataframe, file_path):
        return {
            "columns": dataframe.columns.tolist(),
            "total_rows": len(dataframe),
            "file_path": file_path
        }

    def _create_documents_from_file(self, file_path):
        df = self._load_file(file_path)
        if df is None or df.empty:
            return [], None

        info = self._get_file_info(df, file_path)
        documents = []
        for idx, row in df.iterrows():
            content_lines = []
            for col in df.columns:
                cell = row[col]
                if pd.notna(cell):
                    content_lines.append(f"{col}: {cell}")
            if content_lines:
                doc = Document(
                    page_content="\n".join(content_lines),
                    metadata={
                        "source": file_path,
                        "row_number": int(idx),
                        "filename": os.path.basename(file_path)
                    }
                )
                documents.append(doc)
        return documents, info

    def retrieve_from_vectors(self, query: str) -> str:
        retrieved = self.retriever.invoke(query)
        return "\n\n---\n\n".join([d.page_content for d in retrieved])

    def run_sql_query(self, question: str, retry_count: int = 0) -> str:
        if retry_count > 2:
            return "Unable to generate valid SQL query after multiple attempts"

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        prompt = f"""Generate ONLY a valid SQLite query. No explanations, no markdown, no extra text.

{self.sql_schema}

Question: {question}

Return ONLY the SQL query without any formatting or code blocks."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        sql_query = resp.content.strip()
        if sql_query.lower().startswith("```sql"):
            sql_query = sql_query.split("```sql")[-1].split("```")[0].strip()
        lines = [l.strip() for l in sql_query.splitlines() if l.strip() and not l.strip().startswith("--")]
        sql_query = " ".join(lines).split(";")[0]

        cursor = self.sqlite_connection.cursor()
        try:
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]
            result_text = f"SQL: {sql_query}\n\nResults ({len(rows)} rows):\nColumns: {', '.join(cols)}\n\n"
            for row in rows[:20]:
                result_text += str(row) + "\n"
            return result_text
        except Exception as e:
            print(f"[WARNING] SQL execution failed: {e} – retrying...")
            return self.run_sql_query(question, retry_count + 1)

    def generate_answer(self, context: str, question: str) -> str:
        system_msg = f"""Answer using the context provided.

Database Info:
{self.schema_description}

Be concise and accurate."""
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        resp = llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {question}")
        ])
        return resp.content

# Instantiate RAG system
print(f"[INFO] Initializing RAGSystem with data folder: {DATA_FOLDER}")
rag_system = RAGSystem(DATA_FOLDER)
print("[INFO] RAGSystem initialized.")

# Define MCP tools & resources

@mcp.tool()
def vector_search(query: str) -> str:
    """Do vector retrieval + answer generation."""
    context = rag_system.retrieve_from_vectors(query)
    return rag_system.generate_answer(context, query)

@mcp.tool()
def sql_query(question: str) -> str:
    """Run SQL query over tables and generate answer."""
    context = rag_system.run_sql_query(question)
    return rag_system.generate_answer(context, question)

@mcp.tool()
def smart_query(question: str) -> str:
    """Decide whether to run vector search or SQL and answer."""
    sql_keywords = ['count', 'total', 'sum', 'average', 'avg', 'max', 'min', 
                    'top', 'highest', 'lowest', 'how many', 'all', 'most', 'least']
    needs_sql = any(kw in question.lower() for kw in sql_keywords)
    if needs_sql:
        context = rag_system.run_sql_query(question)
    else:
        context = rag_system.retrieve_from_vectors(question)
    return rag_system.generate_answer(context, question)

@mcp.resource("schema://database")
def get_database_schema() -> str:
    """Get the schema description of the in-memory SQLite database."""
    return rag_system.sql_schema

@mcp.resource("schema://files")
def get_file_schema() -> str:
    """Get the description of the loaded data files."""
    return rag_system.schema_description

# Server startup
if __name__ == "__main__":
    print("[INFO] Starting MCP server (STDIO transport) …")
    mcp.run(transport="stdio")
