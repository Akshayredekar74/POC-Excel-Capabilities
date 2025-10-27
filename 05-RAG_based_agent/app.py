import gradio as gr
from agent import setup_milvus, index_file, create_agent, process_query, clear_chat_history

agent = None
uploaded_files = {}

def initialize_system(files):
    global agent, uploaded_files
    
    if files is None or len(files) == 0:
        return "Please select at least one file"
    
    try:
        setup_milvus()
        clear_chat_history()  # Clear previous session
        
        uploaded_files = {}
        file_info = []
        
        for file in files:
            table_name, columns = index_file(file.name)
            uploaded_files[table_name] = columns
            file_info.append(f"{table_name}: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
        
        agent = create_agent()
        
        return f"loaded {len(files)} file(s):\n\n" + "\n".join(file_info) + "\n\n✨"
    except Exception as e:
        return f"Error: {str(e)}"

def chat(message, history):
    global agent
    
    if agent is None:
        return "Please upload a file first using the Upload section above"
    
    if not message or message.strip() == "":
        return "Please enter a question"
    
    try:
        response = process_query(message, agent)
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"

def clear_session():
    """Clear chat history and return empty chatbot"""
    clear_chat_history()
    return []

# Custom CSS for better UI
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.upload-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""
    # Multi-File RAG & SQL Agent
    
    Upload your CSV/Excel files and ask questions. The agent will automatically choose between:
    - **RAG**: For schema exploration and data understanding
    - **SQL**: For aggregations, filtering, and calculations
    """)
    
    with gr.Accordion("File Upload Section", open=True):
        with gr.Row():
            with gr.Column(scale=3):
                file_input = gr.File(
                    label="Upload Multiple CSV/Excel Files", 
                    file_types=[".csv", ".xlsx", ".xls"], 
                    file_count="multiple"
                )
            with gr.Column(scale=1):
                upload_btn = gr.Button("Load Files", variant="primary", size="lg")
        
        status = gr.Textbox(
            label="System Status", 
            interactive=False, 
            lines=6,
            placeholder="Upload files to get started..."
        )
    
    gr.Markdown("---")
    
    gr.Markdown("### Chat Interface")
    
    chatbot = gr.Chatbot(
        label="Conversation", 
        height=450, 
        type="messages",
        show_copy_button=True
    )
    
    msg = gr.Textbox(
        label="Your Question", 
        placeholder="Try: 'What columns are in the dataset?' or 'What is the total sales?'", 
        lines=2
    )
    
    with gr.Row():
        submit = gr.Button("Send", variant="primary", scale=2)
        clear = gr.Button("Clear Chat", variant="secondary", scale=1)
    
    gr.Markdown("""
    ### Example Questions:
    - "What columns are available in the dataset?" (RAG)
    - "Show me the top 5 rows" (SQL)
    - "What is the total sales by category?" (SQL)
    - "Describe the dataset structure" (RAG)
    """)
    
    # Event handlers
    upload_btn.click(
        fn=initialize_system,
        inputs=[file_input],
        outputs=[status]
    )
    
    def respond(message, chat_history):
        bot_response = chat(message, chat_history)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_response})
        return "", chat_history
    
    submit.click(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    clear.click(
        fn=clear_session,
        inputs=None,
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)