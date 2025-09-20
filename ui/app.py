"""
Streamlit UI for RAG System
Interactive web interface for document ingestion and querying
"""

import streamlit as st
import requests
import json
import pandas as pd
from pathlib import Path
import time
import os
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

def main():
    """Main application function"""
    st.title("📚 End-to-End RAG System")
    st.markdown("Retrieval-Augmented Generation System for PDF Documents")
    
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Dashboard"
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Dashboard", "📤 Upload Documents", "❓ Query Documents", "📊 System Stats"],
            index=["🏠 Dashboard", "📤 Upload Documents", "❓ Query Documents", "📊 System Stats"].index(st.session_state.page)
        )
        
        # Update session state when selectbox changes
        if page != st.session_state.page:
            st.session_state.page = page
        
        st.header("API Status")
        if st.button("Check API Health"):
            check_api_health()
    
    # Main content
    if st.session_state.page == "🏠 Dashboard":
        show_dashboard()
    elif st.session_state.page == "📤 Upload Documents":
        show_upload_page()
    elif st.session_state.page == "❓ Query Documents":
        show_query_page()
    elif st.session_state.page == "📊 System Stats":
        show_stats_page()

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            st.success("✅ API is running and healthy!")
        else:
            st.error("❌ API is not responding properly")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure it's running on port 8000")
    except Exception as e:
        st.error(f"❌ Error checking API health: {str(e)}")

def show_dashboard():
    """Show the main dashboard"""
    st.header("🏠 Dashboard")
    
    # Check API status
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            st.success("✅ RAG System is running")
            
            # Get system stats
            stats_response = requests.get(f"{API_BASE_URL}/stats")
            if stats_response.status_code == 200:
                stats = stats_response.json()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Documents", stats.get('vector_database', {}).get('total_documents', 0))
                with col2:
                    st.metric("System Status", stats.get('status', 'Unknown'))
                with col3:
                    st.metric("Collection", stats.get('vector_database', {}).get('collection_name', 'Unknown'))
        else:
            st.error("❌ RAG System is not responding")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to RAG System. Please start the API server.")
        
    # Quick actions
    st.header("Quick Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Upload New Document"):
            st.session_state.page = "📤 Upload Documents"
            
    with col2:
        if st.button("❓ Ask a Question"):
            st.session_state.page = "❓ Query Documents"

def show_upload_page():
    """Show document upload page"""
    st.header("📤 Upload Documents")
    
    # Back to dashboard button
    if st.button("← Back to Dashboard"):
        st.session_state.page = "🏠 Dashboard"
        st.rerun()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to be processed and added to the knowledge base"
    )
    
    if uploaded_file is not None:
        st.info(f"Selected file: {uploaded_file.name}")
        
        # Upload button
        if st.button("🚀 Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # Prepare file for upload
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}
                    
                    # Send to API
                    response = requests.post(f"{API_BASE_URL}/ingest", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Document processed successfully!")
                        
                        # Show results
                        st.subheader("Processing Results")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Pages Processed", result.get('pages_processed', 0))
                        with col2:
                            st.metric("Chunks Created", result.get('chunks_created', 0))
                        with col3:
                            st.metric("Embeddings Generated", result.get('embeddings_generated', 0))
                            
                        # Show metadata
                        if 'doc_ids' in result:
                            st.subheader("Document IDs")
                            st.code(json.dumps(result['doc_ids'], indent=2))
                            
                    else:
                        st.error(f"❌ Error processing document: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Batch upload
    st.header("📚 Batch Upload")
    st.info("For multiple documents, use the API endpoint directly or upload them one by one.")
    
    # Clear documents
    st.header("🗑️ Clear Documents")
    if st.button("Clear All Documents", type="secondary"):
        if st.button("⚠️ Confirm Clear", type="primary"):
            try:
                response = requests.delete(f"{API_BASE_URL}/documents")
                if response.status_code == 200:
                    st.success("✅ All documents cleared successfully!")
                else:
                    st.error(f"❌ Error clearing documents: {response.text}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def show_query_page():
    """Show document query page"""
    st.header("❓ Query Documents")
    
    # Back to dashboard button
    if st.button("← Back to Dashboard"):
        st.session_state.page = "🏠 Dashboard"
        st.rerun()
    
    # Query input
    query = st.text_area(
        "Ask a question about your documents:",
        placeholder="e.g., What are the main benefits of RAG systems?",
        height=100
    )
    
    # Query parameters
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results to retrieve", min_value=1, max_value=20, value=5)
    with col2:
        if st.button("🔍 Search", type="primary"):
            if query.strip():
                process_query(query, top_k)
            else:
                st.warning("Please enter a question first.")

def process_query(query: str, top_k: int):
    """Process a query and display results"""
    with st.spinner("Searching documents..."):
        try:
            # Send query to API
            response = requests.post(f"{API_BASE_URL}/query", json={
                "question": query,
                "top_k": top_k
            })
            
            if response.status_code == 200:
                result = response.json()
                display_query_results(result)
            else:
                st.error(f"❌ Error processing query: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def display_query_results(result: Dict):
    """Display query results"""
    st.success("✅ Query processed successfully!")
    
    # Answer
    st.subheader("🤖 Answer")
    st.write(result['answer'])
    
    # Confidence score
    confidence = result.get('confidence_score', 0)
    st.metric("Confidence Score", f"{confidence:.2%}")
    
    # Citations
    if result.get('citations'):
        st.subheader("📚 Sources")
        citations_df = pd.DataFrame(result['citations'])
        st.dataframe(citations_df[['source', 'page', 'similarity_score']])
        
        # Show detailed citations
        for i, citation in enumerate(result['citations']):
            with st.expander(f"Source {i+1}: {citation['source']} (Page {citation['page']})"):
                st.write("**Text Snippet:**")
                st.write(citation['text_snippet'])
                st.write(f"**Similarity Score:** {citation['similarity_score']:.3f}")
    
    # Source documents
    if result.get('source_documents'):
        st.subheader("📄 Source Documents")
        for doc in result['source_documents']:
            st.write(f"- {doc}")

def show_stats_page():
    """Show system statistics page"""
    st.header("📊 System Statistics")
    
    # Back to dashboard button
    if st.button("← Back to Dashboard"):
        st.session_state.page = "🏠 Dashboard"
        st.rerun()
    
    try:
        # Get system stats
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            
            # Display stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Vector Database")
                db_stats = stats.get('vector_database', {})
                st.metric("Total Documents", db_stats.get('total_documents', 0))
                st.metric("Collection Name", db_stats.get('collection_name', 'Unknown'))
                st.metric("Persist Directory", db_stats.get('persist_directory', 'Unknown'))
                
            with col2:
                st.subheader("System Status")
                st.metric("Status", stats.get('status', 'Unknown'))
                st.metric("Configuration", "Loaded" if stats.get('config') else 'Not Loaded')
                
            # Configuration details
            if stats.get('config'):
                st.subheader("Configuration Details")
                config = stats['config']
                
                # Display config in expandable sections
                with st.expander("Ingestion Settings"):
                    st.json(config.get('ingestion', {}))
                    
                with st.expander("Embedding Settings"):
                    st.json(config.get('embeddings', {}))
                    
                with st.expander("Vector Database Settings"):
                    st.json(config.get('vector_db', {}))
                    
        else:
            st.error(f"❌ Error getting stats: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please start the server.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
