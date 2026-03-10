import streamlit as st
import chromadb
import os
import re

# Initialize ChromaDB client from the local persistent directory
db_path = "./chroma_db"

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=db_path)

try:
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name="doc_chunks_v7")
except Exception as e:
    st.error(f"Failed to connect to Chroma DB: {e}")
    st.stop()

st.title("Custom RAG chatbot")
st.write("Ask a question, and I'll retrieve the most relevant sections from the document (CoE Kavach Ver 3.2 manual.pdf).")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "image":
            st.image(message["content"], caption=message.get("caption", ""))
        else:
            st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the document..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Query the Chroma DB collection
            # n_results limits the number of chunks to retrieve
            results = collection.query(
                query_texts=[prompt],
                n_results=10 
            )
            
            if results and results['documents'] and len(results['documents'][0]) > 0:
                retrieved_texts = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                # Filter out chunks that are too far (irrelevant)
                valid_indices = [i for i, dist in enumerate(distances) if dist < 1.5]
                
                if not valid_indices:
                    response = "I'm sorry, but I couldn't find any information about that in the provided document. Please ask a question related to the Kavach Manual."
                else:
                    # We will display text, and queue images to be displayed via st.image later
                    # so the Streamlit UI flow feels natural
                    images_to_display = []
                    seen_images = set()
                    
                    seen_paragraphs = set()
                    response_paragraphs = []
                    
                    for i in valid_indices:
                        text = retrieved_texts[i]
                        meta = metadatas[i]
                        
                        # Deduplicate overlapping text paragraphs (blocks), not single physical lines
                        for paragraph in re.split(r'\n{2,}', text):
                            para_clean = paragraph.strip()
                            if para_clean and para_clean not in seen_paragraphs:
                                seen_paragraphs.add(para_clean)
                                response_paragraphs.append(para_clean)
                        
                        if meta.get("images"):
                            img_paths = meta["images"].split(",")
                            for img_path in img_paths:
                                clean_path = img_path.strip()
                                if clean_path and os.path.exists(clean_path) and clean_path not in seen_images:
                                    seen_images.add(clean_path)
                                    images_to_display.append({"path": clean_path, "caption": f"Image from Page {meta.get('page_number', 'Unknown')}"})
                    
                    # Join unique paragraphs into the final response
                    response = "\n\n".join(response_paragraphs)
            else:
                response = "I couldn't find any relevant information in the document for your query."
                
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display any images associated with the chunks
            if 'images_to_display' in locals() and images_to_display:
                for img_data in images_to_display:
                    st.image(img_data["path"], caption=img_data["caption"])
                    # Store image in session state message for persistence upon rerun using a custom type
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "type": "image", 
                        "content": img_data['path'], 
                        "caption": img_data['caption']
                    })
            
        except Exception as e:
            st.error(f"Error querying the database: {e}")

# Inject custom CSS to pin the disclaimer at the very bottom of the page
st.markdown(
    """
    <style>
    .disclaimer {
        position: fixed;
        bottom: 0px;
        left: 0;
        width: 100%;
        text-align: center;
        color: #888;
        font-size: 12px;
        padding: 5px;
        background-color: transparent;
        z-index: 100;
    }
    /* Add some padding to the main content so it doesn't overlap with the fixed footer */
    .main .block-container {
        padding-bottom: 80px;
    }
    </style>
    <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> Generative AI can make mistakes. Please verify important information regarding train operation procedures manually.
    </div>
    """,
    unsafe_allow_html=True
)
