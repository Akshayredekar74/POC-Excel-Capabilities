"""
SQL Tools for Database Operations
"""

import duckdb
from typing import Dict, Any

class SQLTools:
    
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get comprehensive table information"""
        try:
            # Get column info
            schema_info = self.conn.execute("PRAGMA table_info('data')").fetchall()
            columns = [{"name": col[1], "type": col[2]} for col in schema_info]
            
            # Get row count
            row_count = self.conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]
            
            # Get sample data
            sample_data = self.conn.execute("SELECT * FROM data LIMIT 5").fetchdf()
            
            return {
                "columns": columns,
                "row_count": row_count,
                "sample_data": sample_data.to_string(index=False)
            }
        except:
            return {"error": "Could not fetch table info"}
    
    def execute_aggregation(self, column: str, operation: str) -> Any:
        """
        Execute aggregation operations
        Operations: SUM, AVG, COUNT, MIN, MAX
        """
        operations = ['SUM', 'AVG', 'COUNT', 'MIN', 'MAX']
        op = operation.upper()
        
        if op not in operations:
            return {"error": f"Invalid operation. Use: {', '.join(operations)}"}
        
        try:
            query = f"SELECT {op}({column}) as result FROM data"
            result = self.conn.execute(query).fetchone()[0]
            return {"operation": op, "column": column, "result": result}
        except Exception as e:
            return {"error": str(e)}
    
    def get_distinct_values(self, column: str, limit: int = 10) -> Dict[str, Any]:
        """Get distinct values from a column"""
        try:
            query = f"SELECT DISTINCT {column} FROM data LIMIT {limit}"
            result = self.conn.execute(query).fetchdf()
            return {
                "column": column,
                "distinct_values": result[column].tolist(),
                "count": len(result)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def group_by_aggregate(self, group_column: str, agg_column: str, operation: str = "SUM") -> Dict[str, Any]:
        """Group by one column and aggregate another"""
        try:
            query = f"""
            SELECT {group_column}, {operation}({agg_column}) as result
            FROM data
            GROUP BY {group_column}
            ORDER BY result DESC
            LIMIT 10
            """
            result = self.conn.execute(query).fetchdf()
            return {
                "group_by": group_column,
                "aggregation": f"{operation}({agg_column})",
                "results": result.to_string(index=False)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def filter_data(self, column: str, operator: str, value: Any) -> Dict[str, Any]:
        """Filter data based on condition"""
        try:
            query = f"SELECT * FROM data WHERE {column} {operator} {value} LIMIT 20"
            result = self.conn.execute(query).fetchdf()
            return {
                "filter": f"{column} {operator} {value}",
                "rows_returned": len(result),
                "results": result.to_string(index=False)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def execute_custom_query(self, query: str) -> Dict[str, Any]:
        """Execute custom SQL query"""
        try:
            result = self.conn.execute(query).fetchdf()
            return {
                "query": query,
                "rows_returned": len(result),
                "results": result.head(20).to_string(index=False)
            }
        except Exception as e:
            return {"error": str(e), "query": query}
    
    def get_column_stats(self, column: str) -> Dict[str, Any]:
        """Get statistics for a numeric column"""
        try:
            query = f"""
            SELECT 
                COUNT({column}) as count,
                AVG({column}) as mean,
                MIN({column}) as min,
                MAX({column}) as max,
                STDDEV({column}) as stddev
            FROM data
            """
            result = self.conn.execute(query).fetchone()
            return {
                "column": column,
                "count": result[0],
                "mean": result[1],
                "min": result[2],
                "max": result[3],
                "stddev": result[4]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_top_n(self, column: str, n: int = 10, order: str = "DESC") -> Dict[str, Any]:
        """Get top N rows ordered by column"""
        try:
            query = f"SELECT * FROM data ORDER BY {column} {order} LIMIT {n}"
            result = self.conn.execute(query).fetchdf()
            return {
                "column": column,
                "order": order,
                "top_n": n,
                "results": result.to_string(index=False)
            }
        except Exception as e:
            return {"error": str(e)}
