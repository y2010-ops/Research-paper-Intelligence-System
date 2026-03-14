import streamlit as st
import requests
import json

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="RPAI System", layout="wide")

st.title("Research Paper Intelligence System 🧠")

# Sidebar - Ingestion
with st.sidebar:
    st.header("📄 Upload Papers")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file and st.button("Ingest Paper"):
        with st.spinner("Uploading & Processing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                response = requests.post(f"{API_URL}/ingest/upload", files=files)
                
                if response.status_code == 200:
                    st.success(f"Successfully started processing {uploaded_file.name}")
                else:
                    st.error(f"Failed: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
                
    st.divider()
    st.markdown("**System Status**")
    try:
        status = requests.get(f"{API_URL}/")
        if status.status_code == 200:
            st.success("Backend Online ✅")
            data = status.json()
            st.caption(f"LLM: {data.get('llm')}")
            st.caption(f"Embeddings: {data.get('embeddings')}")
        else:
            st.error("Backend Offline ❌")
    except:
        st.error("Backend unreachable ❌")

# Main Chat Interface
st.subheader("Query the Knowledge Base")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question (e.g., 'What is the Transformer architecture?')"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Get AIs response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Thinking... (this may take a moment for RAG + Graph search)"):
            try:
                response = requests.post(f"{API_URL}/query", json={"query": prompt})
                
                if response.status_code == 200:
                    result = response.json()
                    full_response = result.get("final_answer", "No answer provided.")
                    
                    # Display Sources if available (RAG)
                    rag_data = result.get("rag_response", {})
                    sources = rag_data.get("sources", [])
                    
                    # Display Graph foundings (Graph)
                    graph_data = result.get("graph_response", {})
                    graph_answer = graph_data.get("answer", "")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Expanders for details
                    with st.expander("📚 Retrieved Sources (RAG)"):
                        if sources:
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1}** (Score: {source.get('score', 0):.2f})")
                                st.caption(source.get('text')[:300] + "...")
                                st.divider()
                        else:
                            st.info("No text sources used.")
                            
                    with st.expander("🕸️ Knowledge Graph Data"):
                        if graph_answer:
                            st.write(graph_answer)
                            if graph_data.get("data"):
                                st.json(graph_data["data"])
                        else:
                            st.info("No graph data used.")
                            
                else:
                    full_response = f"Error {response.status_code}: {response.text}"
                    message_placeholder.error(full_response)
                    
            except Exception as e:
                full_response = f"Connection Failed: {str(e)}"
                message_placeholder.error(full_response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
