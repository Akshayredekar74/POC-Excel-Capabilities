"""
Test script to verify AI Agent setup
"""

import os
from dotenv import load_dotenv

print("Testing AI Agent Setup...")
print("=" * 50)

# Test 1: Check .env file
print("\n1. Checking .env file...")
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    print("   ✓ GOOGLE_API_KEY found")
else:
    print("   ✗ GOOGLE_API_KEY not found in .env")
    print("   Please create .env file with: GOOGLE_API_KEY=your_key")

# Test 2: Check imports
print("\n2. Checking imports...")
try:
    import polars as pl
    print("   ✓ polars")
except ImportError:
    print("   ✗ polars - Run: pip install polars")

try:
    import duckdb
    print("   ✓ duckdb")
except ImportError:
    print("   ✗ duckdb - Run: pip install duckdb")

try:
    from langgraph.graph import StateGraph
    print("   ✓ langgraph")
except ImportError:
    print("   ✗ langgraph - Run: pip install langgraph")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("   ✓ langchain-google-genai")
except ImportError:
    print("   ✗ langchain-google-genai - Run: pip install langchain-google-genai")

try:
    import gradio as gr
    print("   ✓ gradio")
except ImportError:
    print("   ✗ gradio - Run: pip install gradio")

# Test 3: Test data cleaning
print("\n3. Testing data cleaner...")
try:
    from cleaner import DataCleaner
    
    # Create test data
    test_df = pl.DataFrame({
        "price": ["$100", "$200.50", "$300"],
        "quantity": ["10", "20", "30"],
        "active": ["yes", "no", "yes"]
    })
    
    cleaned_df, report = DataCleaner.clean_dataframe(test_df)
    print(f"   ✓ Cleaned {len(report)} columns")
    for item in report:
        print(f"     - {item['column']}: {item['action']}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Test SQL tools
print("\n4. Testing SQL tools...")
try:
    from tools import SQLTools
    import duckdb
    
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE data (id INT, value VARCHAR)")
    conn.execute("INSERT INTO data VALUES (1, 'test'), (2, 'demo')")
    
    tools = SQLTools(conn)
    info = tools.get_table_info()
    print(f"   ✓ SQL tools working")
    print(f"     - Found {len(info['columns'])} columns")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 5: Test agent
print("\n5. Testing agent...")
try:
    from agent import SQLAgent
    
    agent = SQLAgent()
    
    # Create test CSV
    test_file = "test_data.csv"
    test_df = pl.DataFrame({
        "Product": ["A", "B", "C"],
        "Price": ["$10", "$20", "$30"],
        "Quantity": ["5", "10", "15"]
    })
    test_df.write_csv(test_file)
    
    # Load and test
    agent.load_data(test_file)
    print("   ✓ Agent initialized and data loaded")
    
    # Cleanup
    os.remove(test_file)
    
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 50)
print("Setup verification complete!")
print("\nTo run the app: python app.py")
