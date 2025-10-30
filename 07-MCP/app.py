import chainlit as cl
from rag_mcp_server import RAGSystem  # your class
from rag_mcp_server import vector_search, sql_query, smart_query  
# Instantiate your system
rag_system = RAGSystem(data_folder="test")

@cl.on_message
async def handle_user_message(message: cl.Message):
    user_input = message.content

    # Decide which tool to call (you could also call smart_query always)
    # e.g. use smart_query
    result = await smart_query(user_input)

    # Send back the answer
    await cl.Message(content=result).send()
