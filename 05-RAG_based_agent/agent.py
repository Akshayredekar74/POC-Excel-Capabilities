import os
import json
import pandas as pd
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
import sqlite3
from dotenv import load_dotenv
import numpy as np

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
connections.connect(host="localhost", port="19530")
COLLECTION_NAME = "documents"

# CRITICAL: Track only user-uploaded files
user_uploaded_tables = set()  # Only tables from user uploads
file_metadata = {}  # Maps table names to original file info
chat_history_store = []

def setup_milvus():
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000)
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

def chunk_structured_data(df, filename, chunk_size=10):
    """Optimized chunking with summary for context"""
    chunks = []
    
    # Add a summary chunk for better retrieval
    summary = {
        'text': f"Dataset: {filename}\nColumns: {', '.join(df.columns.tolist())}\nTotal rows: {len(df)}\nSummary:\n{df.describe().to_string()}",
        'metadata': json.dumps({
            'filename': filename,
            'type': 'summary',
            'total': len(df)
        })
    }
    chunks.append(summary)
    
    # Chunk the data
    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i+chunk_size]
        json_str = chunk_df.to_json(orient='records', indent=2)
        chunks.append({
            'text': json_str,
            'metadata': json.dumps({
                'filename': filename,
                'rows': f'{i}-{i+len(chunk_df)}',
                'total': len(df),
                'type': 'data'
            })
        })
    return chunks

def load_file(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        return None
    return df

def index_file(file_path):
    """Index user-uploaded file and mark it as user-uploaded"""
    df = load_file(file_path)
    filename = os.path.basename(file_path)
    table_name = filename.replace('.', '_').replace(' ', '_')
    
    # CRITICAL: Mark this as a user-uploaded table
    user_uploaded_tables.add(table_name)
    
    # Store file metadata
    file_metadata[table_name] = {
        'original_filename': filename,
        'columns': df.columns.tolist(),
        'row_count': len(df),
        'is_user_uploaded': True
    }
    
    chunks = chunk_structured_data(df, filename)
    
    collection = Collection(COLLECTION_NAME)
    texts = [c['text'] for c in chunks]
    metadatas = [c['metadata'] for c in chunks]
    
    vectors = embeddings.embed_documents(texts)
    
    collection.insert([vectors, texts, metadatas])
    collection.load()
    
    conn = sqlite3.connect('data.db')
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    
    return table_name, df.columns.tolist()

def get_user_tables():
    """Get only user-uploaded tables"""
    return list(user_uploaded_tables)

def search_milvus(query, top_k=3):
    """Enhanced search with similarity scores and embeddings"""
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    query_vector = embeddings.embed_query(query)
    
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text", "metadata"]
    )
    
    docs_with_scores = []
    for hits in results:
        for hit in hits:
            similarity_score = 1 / (1 + hit.distance)
            docs_with_scores.append({
                'text': hit.entity.get('text'),
                'metadata': json.loads(hit.entity.get('metadata')),
                'similarity_score': similarity_score,
                'distance': hit.distance,
                'embedding_sample': query_vector[:5]
            })
    
    return docs_with_scores, query_vector[:5]

def execute_sql(sql_query, user_tables_only=True):
    """Execute SQL with support for multiple queries - returns structured results"""
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    
    # Get only user-uploaded tables
    if user_tables_only:
        tables = get_user_tables()
    else:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
    
    try:
        # Split multiple queries
        queries = [q.strip() for q in sql_query.split(';') if q.strip()]
        
        if len(queries) == 0:
            conn.close()
            return [], tables
        
        results = []
        for query in queries:
            try:
                result_df = pd.read_sql_query(query, conn)
                
                # Identify which table this query is for
                query_lower = query.lower()
                matched_table = None
                for table in tables:
                    if table.lower() in query_lower:
                        matched_table = table
                        break
                
                # Only include results from user-uploaded tables
                if matched_table and matched_table in user_uploaded_tables:
                    results.append({
                        'query': query,
                        'table': matched_table,
                        'dataframe': result_df,
                        'success': True
                    })
            except Exception as e:
                # Only report errors for user tables
                query_lower = query.lower()
                is_user_table = any(table.lower() in query_lower for table in user_uploaded_tables)
                if is_user_table:
                    results.append({
                        'query': query,
                        'table': None,
                        'error': str(e),
                        'success': False
                    })
        
        conn.close()
        return results, tables
    except Exception as e:
        conn.close()
        raise Exception(f"SQL Error: {str(e)}\nQuery: {sql_query}")

def dataframe_to_markdown(df, max_rows=50):
    """Convert DataFrame to markdown table format"""
    if len(df) > max_rows:
        df = df.head(max_rows)
        truncated = True
    else:
        truncated = False
    
    # Create header
    headers = df.columns.tolist()
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    
    # Create data rows
    data_rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(val) for val in row.values) + " |"
        data_rows.append(row_str)
    
    result = "\n".join([header_row, separator] + data_rows)
    
    if truncated:
        result += f"\n\n*Showing first {max_rows} rows of {len(df)} total*"
    
    return result

def get_conversation_context(max_history=3):
    """Get recent conversation history for context"""
    if len(chat_history_store) == 0:
        return ""
    
    recent_history = chat_history_store[-max_history:]
    context_parts = []
    for entry in recent_history:
        context_parts.append(f"User: {entry['query']}\nAssistant: {entry['response'][:200]}...")
    
    return "\n\n".join(context_parts)

class AgentState(TypedDict):
    query: str
    decision: str
    context: str
    sql_query: str
    result: str
    tables: List[str]
    retrieval_info: Dict[str, Any]
    conversation_history: str
    structured_results: List[Dict[str, Any]]

def router_node(state: AgentState):
    """Enhanced router with conversation context"""
    query = state['query']
    conv_history = get_conversation_context()
    
    prompt = f"""You are a routing agent. Analyze the query and decide the approach.
Recent conversation:
{conv_history if conv_history else 'No previous context'}
Current query: {query}
Rules:
- RETRIEVE: Use for questions about data structure, schema, column names, or general information
- SQL: Use for aggregations, filtering, calculations, specific data queries, or displaying records
Respond with ONE word only (RETRIEVE or SQL):"""
    
    response = llm.invoke(prompt)
    decision = response.content.strip()
    
    return {"decision": decision, "conversation_history": conv_history}

def retrieve_node(state: AgentState):
    """Enhanced retrieval with transparency"""
    query = state['query']
    docs_with_scores, query_embedding = search_milvus(query, top_k=3)
    
    context_parts = []
    retrieval_details = []
    
    for idx, doc in enumerate(docs_with_scores, 1):
        context_parts.append(doc['text'])
        retrieval_details.append({
            'rank': idx,
            'similarity_score': round(doc['similarity_score'], 4),
            'distance': round(doc['distance'], 4),
            'content_preview': doc['text'][:200] + "...",
            'metadata': doc['metadata']
        })
    
    retrieval_info = {
        'method': 'RAG (Retrieval-Augmented Generation)',
        'query_embedding_sample': [round(x, 4) for x in query_embedding],
        'num_documents_retrieved': len(docs_with_scores),
        'documents': retrieval_details
    }
    
    context = "\n\n".join(context_parts)
    
    return {"context": context, "retrieval_info": retrieval_info}

def sql_node(state: AgentState):
    """FIXED: Enhanced SQL generation - only uses user-uploaded tables"""
    query = state['query']
    conv_history = state.get('conversation_history', '')
    
    # CRITICAL: Only get user-uploaded tables
    tables = get_user_tables()
    
    if not tables:
        return {
            "sql_query": "", 
            "tables": [], 
            "retrieval_info": {
                'method': 'SQL Query Generation',
                'status': 'error',
                'error': 'No user-uploaded files found'
            },
            "structured_results": [],
            "result": "No files have been uploaded yet. Please upload files first."
        }
    
    conn = sqlite3.connect('data.db')
    
    # Get schemas ONLY for user-uploaded tables
    all_schemas = []
    for table in tables:
        schema_info = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
        columns = schema_info['name'].tolist()
        sample_data = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 2", conn)
        
        # Include original filename in schema
        original_name = file_metadata.get(table, {}).get('original_filename', table)
        all_schemas.append(f"Table: {table} (Original file: {original_name})\nColumns: {', '.join(columns)}\nSample:\n{sample_data.to_string(index=False)}")
    
    conn.close()
    
    schema_text = "\n\n".join(all_schemas)
    
    # Enhanced prompt - explicitly mentions available tables
    prompt = f"""Generate SQLite query/queries for the request.

IMPORTANT: Only these tables are available (user-uploaded files):
{', '.join(tables)}

Schema:
{schema_text}

{f"Context: {conv_history[:300]}" if conv_history else ""}

Query: {query}

CRITICAL RULES:
1. ONLY use the tables listed above: {', '.join(tables)}
2. If the query asks for data from "EACH file" or "ALL files", generate SEPARATE queries for EACH of these tables: {', '.join(tables)}
3. Each query should be separated by a semicolon (;)
4. Use double quotes for column names with spaces
5. Return ONLY the SQL queries, no explanations

Examples:
- "show top 10 from each file" → {'; '.join([f'SELECT * FROM {t} LIMIT 10' for t in tables[:2]])};
- "total from first file" → SELECT COUNT(*) FROM {tables[0] if tables else 'table'};

SQL:"""
    
    response = llm.invoke(prompt)
    sql_query = response.content.strip()
    
    # Clean up SQL query
    if "```sql" in sql_query:
        sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql_query:
        sql_query = sql_query.split("```")[1].split("```")[0].strip()
    
    sql_query = sql_query.replace('`', '"')
    
    # Count number of queries
    num_queries = len([q for q in sql_query.split(';') if q.strip()])
    
    retrieval_info = {
        'method': 'SQL Query Generation',
        'approach': 'Multi-table query execution' if num_queries > 1 else 'Single query execution',
        'generated_query': sql_query,
        'tables_used': tables,
        'num_queries': num_queries
    }
    
    try:
        structured_results, _ = execute_sql(sql_query, user_tables_only=True)
        retrieval_info['status'] = 'success'
        return {
            "sql_query": sql_query, 
            "tables": tables, 
            "retrieval_info": retrieval_info,
            "structured_results": structured_results
        }
    except Exception as e:
        retrieval_info['status'] = 'error'
        retrieval_info['error'] = str(e)
        return {
            "sql_query": sql_query, 
            "result": f"Error: {str(e)}", 
            "tables": tables, 
            "retrieval_info": retrieval_info,
            "structured_results": []
        }

def answer_node(state: AgentState):
    """FIXED: Enhanced answer generation with proper table formatting"""
    query = state['query']
    decision = state['decision']
    conv_history = state.get('conversation_history', '')
    
    if decision == "RETRIEVE":
        context = state.get('context', '')
        prompt = f"""Answer concisely using the context.
{f"Previous: {conv_history[:200]}" if conv_history else ""}
Context: {context[:1000]}
Question: {query}
Answer:"""
        response = llm.invoke(prompt)
        final_result = response.content
    else:
        # Handle structured SQL results
        structured_results = state.get('structured_results', [])
        
        if not structured_results:
            final_result = state.get('result', 'No results found')
        else:
            result_parts = []
            for idx, res in enumerate(structured_results, 1):
                if res['success']:
                    df = res['dataframe']
                    table_name = res['table']
                    
                    # Get original filename
                    original_name = file_metadata.get(table_name, {}).get('original_filename', table_name)
                    
                    result_parts.append(f"### 📊 Results from: {original_name}\n")
                    result_parts.append(f"**Query:** `{res['query']}`\n")
                    
                    if len(df) > 0:
                        markdown_table = dataframe_to_markdown(df)
                        result_parts.append(markdown_table)
                        result_parts.append(f"\n**Total records:** {len(df)}\n")
                    else:
                        result_parts.append("*No data found*\n")
                else:
                    result_parts.append(f"### ❌ Error in query {idx}\n")
                    result_parts.append(f"**Query:** `{res['query']}`\n")
                    result_parts.append(f"**Error:** {res['error']}\n")
            
            final_result = "\n".join(result_parts)
    
    # Store in chat history
    chat_history_store.append({
        'query': query,
        'response': final_result,
        'decision': decision
    })
    
    return {"result": final_result}

def route_decision(state: AgentState):
    decision = state.get('decision', '')
    if decision == "RETRIEVE":
        return "retrieve"
    else:
        return "sql"

def create_agent():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("answer", answer_node)
    
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "sql": "sql"
        }
    )
    
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("sql", "answer")
    workflow.add_edge("answer", END)
    
    return workflow.compile()

def process_query(query, agent):
    """Enhanced query processing with detailed transparency"""
    state = {"query": query}
    result = agent.invoke(state)
    
    response_parts = []
    
    # Method used
    method = result.get('decision', 'Unknown')
    response_parts.append(f"**🔍 Method Used:** {method}")
    
    # Show retrieval info
    if 'retrieval_info' in result:
        info = result['retrieval_info']
        response_parts.append(f"\n**📊 Approach:** {info.get('method', 'N/A')}")
        
        if method == "RETRIEVE":
            response_parts.append(f"\n**🎯 Documents Retrieved:** {info.get('num_documents_retrieved', 0)}")
            response_parts.append(f"**🔢 Query Embedding (sample):** {info.get('query_embedding_sample', [])}")
            
            for doc in info.get('documents', [])[:2]:
                response_parts.append(f"\n**Document {doc['rank']}:**")
                response_parts.append(f"- Similarity Score: {doc['similarity_score']}")
                response_parts.append(f"- Distance: {doc['distance']}")
                response_parts.append(f"- Preview: {doc['content_preview']}")
        
        elif method == "SQL":
            num_queries = info.get('num_queries', 1)
            response_parts.append(f"\n**📝 Number of Queries:** {num_queries}")
            response_parts.append(f"\n**💻 Generated SQL:**\n```sql\n{info.get('generated_query', 'N/A')}\n```")
            response_parts.append(f"\n**✅ Status:** {info.get('status', 'N/A')}")
            
            if info.get('status') == 'error':
                response_parts.append(f"**❌ Error Details:** {info.get('error', 'Unknown error')}")
    
    # Final answer with results
    response_parts.append(f"\n---\n\n**💡 Results:**\n\n{result['result']}")
    
    return "\n".join(response_parts)

def clear_chat_history():
    """Clear the conversation history"""
    global chat_history_store
    chat_history_store = []

def reset_system():
    """Reset the entire system including user uploads"""
    global user_uploaded_tables, file_metadata, chat_history_store
    user_uploaded_tables = set()
    file_metadata = {}
    chat_history_store = []
    
    # Clear database
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")
    conn.commit()
    conn.close()