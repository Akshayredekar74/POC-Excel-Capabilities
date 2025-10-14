"""
AI Agent with SQL Database using LangGraph
"""

import os
import duckdb
import polars as pl
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from cleaner import DataCleaner
from tools import SQLTools

load_dotenv()

class AgentState(TypedDict):
    question: str
    schema_info: str
    sql_query: str
    query_result: str
    answer: str
    error: str
    iteration: int
    cleaning_report: str

class SQLAgent:
    def __init__(self, db_path=':memory:'):
        self.conn = duckdb.connect(database=db_path)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            temperature=0
        )
        self.tools = SQLTools(self.conn)
        self.max_iterations = 3
        self.cleaning_report = []
        
    def load_data(self, file_path):
        """Load and clean data from file"""
        if file_path.endswith('.csv'):
            try:
                df = pl.read_csv(file_path, infer_schema_length=10000, truncate_ragged_lines=True)
            except:
                df = pl.read_csv(file_path, encoding="utf8-lossy", infer_schema_length=10000, truncate_ragged_lines=True)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pl.read_excel(file_path)
        else:
            return None
        
        # Clean the dataframe with intelligent type detection
        df_cleaned, cleaning_report = DataCleaner.clean_dataframe(df)
        self.cleaning_report = cleaning_report
        
        # Remove existing view/table if exists
        try:
            self.conn.execute("DROP VIEW IF EXISTS data")
        except:
            pass
        try:
            self.conn.execute("DROP TABLE IF EXISTS data")
        except:
            pass
        
        # Register as view in DuckDB
        self.conn.register("data", df_cleaned.to_pandas())
        
        return df_cleaned
    
    def get_schema(self, state: AgentState) -> AgentState:
        """Get comprehensive schema information"""
        table_info = self.tools.get_table_info()
        
        if "error" in table_info:
            state['error'] = table_info['error']
            return state
        
        schema_text = f"""Database: DuckDB
Table: data
Total Rows: {table_info['row_count']}

Columns ({len(table_info['columns'])}):
"""
        
        for col in table_info['columns']:
            schema_text += f"\n  - {col['name']} (Type: {col['type']})"
        
        schema_text += f"\n\nSample Data (5 rows):\n{table_info['sample_data']}"
        
        state['schema_info'] = schema_text
        state['iteration'] = 0
        state['cleaning_report'] = self._format_cleaning_report()
        return state
    
    def _format_cleaning_report(self) -> str:
        """Format cleaning report for display"""
        if not self.cleaning_report:
            return "No cleaning operations performed"
        
        report = []
        for item in self.cleaning_report:
            report.append(f"- {item['column']}: {item['action']}")
        return "\n".join(report)
    
    def generate_sql(self, state: AgentState) -> AgentState:
        """Generate SQL query with error context"""
        error_context = ""
        if state.get('error'):
            error_context = f"""
PREVIOUS ATTEMPT FAILED:
Error: {state['error']}
Previous SQL: {state.get('sql_query', 'N/A')}

INSTRUCTIONS TO FIX:
1. Check column names match schema exactly (case-sensitive)
2. Verify column data types from schema
3. Use CAST() if type conversion needed
4. Check for syntax errors
5. Ensure calculations use numeric columns
"""
        
        sql_prompt = f"""You are a SQL expert. Convert the question to a DuckDB SQL query.

{state['schema_info']}

Question: {state['question']}
{error_context}

CRITICAL RULES:
- Table name is 'data'
- Use EXACT column names from schema above
- Return ONLY the SQL query, no explanation or markdown
- Use proper DuckDB syntax
- For calculations, ensure columns are numeric type
- If column is VARCHAR but contains numbers, use: CAST(column_name AS DOUBLE)

SQL Query:"""

        response = self.llm.invoke(sql_prompt)
        sql = response.content.strip()
        
        # Clean up SQL
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql:
            sql = sql.split("```")[1].split("```")[0].strip()
        
        sql = sql.replace('`', '').strip()
        if sql.endswith(';'):
            sql = sql[:-1]
        
        state['sql_query'] = sql
        state['error'] = ""
        return state
    
    def execute_sql(self, state: AgentState) -> AgentState:
        """Execute SQL query"""
        try:
            result_df = self.conn.execute(state['sql_query']).fetchdf()
            
            display_rows = min(20, len(result_df))
            total_rows = len(result_df)
            
            result_text = f"Query returned {total_rows} rows (showing {display_rows}):\n\n"
            result_text += result_df.head(display_rows).to_string(index=False)
            
            state['query_result'] = result_text
            state['error'] = ""
        except Exception as e:
            state['error'] = str(e)
            state['query_result'] = ""
        
        state['iteration'] += 1
        return state
    
    def generate_answer(self, state: AgentState) -> AgentState:
        """Generate natural language answer"""
        if state.get('error'):
            state['answer'] = f"Query failed after {state['iteration']} attempts.\n\nFinal Error: {state['error']}\n\nPlease check your question or try rephrasing it."
            return state
        
        answer_prompt = f"""Based on the SQL query results, provide a clear and direct answer.

Question: {state['question']}

SQL Query Used:
{state['sql_query']}

Query Results:
{state['query_result']}

Provide a concise answer that directly addresses the question. Use specific numbers and facts from the results."""

        response = self.llm.invoke(answer_prompt)
        state['answer'] = response.content
        return state
    
    def should_retry(self, state: AgentState) -> str:
        """Decide whether to retry on error"""
        if state.get('error') and state['iteration'] < self.max_iterations:
            return "retry"
        elif state.get('error'):
            return "failed"
        else:
            return "success"
    
    def create_graph(self):
        """Create LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("get_schema", self.get_schema)
        workflow.add_node("generate_sql", self.generate_sql)
        workflow.add_node("execute_sql", self.execute_sql)
        workflow.add_node("generate_answer", self.generate_answer)
        
        # Set entry point
        workflow.set_entry_point("get_schema")
        
        # Add edges
        workflow.add_edge("get_schema", "generate_sql")
        workflow.add_edge("generate_sql", "execute_sql")
        
        # Conditional retry logic
        workflow.add_conditional_edges(
            "execute_sql",
            self.should_retry,
            {
                "retry": "generate_sql",
                "success": "generate_answer",
                "failed": "generate_answer"
            }
        )
        
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def query(self, question: str):
        """Execute query through agent"""
        graph = self.create_graph()
        
        initial_state = {
            "question": question,
            "schema_info": "",
            "sql_query": "",
            "query_result": "",
            "answer": "",
            "error": "",
            "iteration": 0,
            "cleaning_report": ""
        }
        
        final_state = graph.invoke(initial_state)
        
        return {
            "answer": final_state['answer'],
            "sql": final_state['sql_query'],
            "result": final_state['query_result'],
            "error": final_state.get('error', ''),
            "iterations": final_state['iteration'],
            "schema": final_state['schema_info'],
            "cleaning": final_state.get('cleaning_report', '')
        }
