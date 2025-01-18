import streamlit as st
import os
import tempfile
from llama_parse import LlamaParse
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from operator import itemgetter
import uuid
import re
import time
import os
import nest_asyncio
nest_asyncio.apply()
from dotenv import load_dotenv
load_dotenv()

os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv('LLAMA_CLOUD_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# Initialize session state
if 'chroma_db' not in st.session_state:
    st.session_state.chroma_db = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

def init_llm():
    return ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0,
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )

def init_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )

def categorize_data(data):
    categories = {}
    table_count = 1
    text_count = 1
    for item in data:
        content = item.text
        if re.search(r"\|", content):
            category = f"table{table_count}"
            categories.setdefault(category, []).append(content)
            table_count += 1
        else:
            category = f"text{text_count}"
            categories.setdefault(category, []).append(content)
            text_count += 1
    return categories

def create_summaries(llm, text_docs, table_docs):
    prompt_text = """
    You are an assistant tasked with summarizing tables and text particularly for semantic retrieval.
    These summaries will be embedded and used to retrieve the raw text or table elements.
    Give a detailed summary of the table or text below that is well optimized for retrieval.
    For any tables also add in a one line description of what the table is about besides the summary.
    Do not add additional words like Summary: etc.

    Table or text chunk:
    {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    summarize_chain = (
        {"element": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    text_summaries = []
    table_summaries = []
    
    for doc in text_docs:
        text_summaries.append(summarize_chain.invoke({'element': doc}))
        time.sleep(3)
    
    for doc in table_docs:
        table_summaries.append(summarize_chain.invoke({'element': doc}))
        time.sleep(3)
    return text_summaries, table_summaries

def process_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
        
    data = LlamaParse(result_type="markdown").load_data(tmp_path)
    os.unlink(tmp_path)
    
    category_data = categorize_data(data)
    
    text_docs = []
    table_docs = []
    
    for item in category_data.keys():
        if item.startswith('table'):
            table_docs.append(category_data[item][0])
        else:
            text_docs.append(category_data[item][0])
            
    llm = init_llm()
    text_summaries, table_summaries = create_summaries(llm, text_docs, table_docs)
    
    # Initialize Chroma with persistence
    chroma_db = Chroma(
        collection_name="financial_qa",
        embedding_function=init_embeddings(),
        persist_directory="./chroma_db"
    )
    
    # Add documents to Chroma
    documents = []
    
    # Add text documents
    for i, (summary, content) in enumerate(zip(text_summaries, text_docs)):
        doc_id = f"text_{i}"
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "type": "text",
                    "summary": summary,
                    "id": doc_id
                }
            )
        )
    
    # Add table documents
    for i, (summary, content) in enumerate(zip(table_summaries, table_docs)):
        doc_id = f"table_{i}"
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "type": "table",
                    "summary": summary,
                    "id": doc_id
                }
            )
        )
    
    # Add all documents to Chroma
    chroma_db.add_documents(documents)
    return chroma_db

def create_qa_chain(llm, chroma_db):
    def multimodal_prompt_function(data_dict):
        documents = data_dict["context"]
        formatted_texts = "\n\n".join([doc.page_content for doc in documents])
        
        messages = []
        text_message = {
            "type": "text",
            "text": (
                f"""You are a financial analyst tasked with understanding detailed information and trends from financial documents,
                    data tables, and financial statements.
                    Use the provided context information to answer the user's question about financial metrics and performance.
                    Only use the information from the provided context and be precise with numbers and calculations.
                    Give accurate and coherent responses for the user questions.
                    If you cannot find specific information in the context, please say so.

                    User question:
                    {data_dict['question']}

                    Context documents:
                    {formatted_texts}

                    Answer:
                """
            ),
        }
        messages.append(text_message)
        return [HumanMessage(content=messages)]

    def retrieve_docs(query_dict):
        query = query_dict['input']
        return chroma_db.similarity_search(query, k=1)

    multimodal_rag = (
        {
            "context": itemgetter('context'),
            "question": itemgetter('input'),
        }
        | RunnableLambda(multimodal_prompt_function)
        | llm
        | StrOutputParser()
    )

    retrieve_docs_chain = RunnableLambda(retrieve_docs)

    return (
        RunnablePassthrough.assign(context=retrieve_docs_chain)
        .assign(answer=multimodal_rag)
    )

def main():
    on = st.toggle("Guide Lines")
    if on:
        st.sidebar.markdown("""
    ### How to use this application:
    1. Upload your financial PDF document (preferably containing P&L statements and financial tables)
    2. Ask questions about the financial data in natural language
    3. View the answer and supporting information from the document
    
    #### Example questions:
    - "What was the total revenue for Q1 2024?"
    - "How do operating expenses compare between Q3 and Q4?"
    - "What is the trend in gross profit margin over the past year?"
    """)
    st.title("Financial Document QA System")
    
    uploaded_file = st.file_uploader("Upload your financial PDF document", type="pdf")
    
    if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
        with st.spinner('Processing document... This may take a few minutes.'):
            st.session_state.chroma_db = process_document(uploaded_file)
            st.session_state.processed_files.append(uploaded_file.name)
        st.success('Document processed successfully!')
    
    if st.session_state.chroma_db:
        question = st.text_area("Ask a question about the financial data:")
        if question:
            with st.spinner('Analyzing...'):
                llm = init_llm()
                qa_chain = create_qa_chain(llm, st.session_state.chroma_db)
                response = qa_chain.invoke({"input":question})
                
                st.markdown("### Answer")
                st.write(response['answer'])
                
                st.markdown("### Supporting Information")
                for doc in response['context']:
                    # Extract all table-like sections using the improved regex
                    matches = re.finditer(r"\n\n\|(.*?)(\|\n\n|\n\n\|)", doc.page_content, re.DOTALL)

                    # Iterate through matches to process each table section
                    current_table = []
                    for match in matches:
                        content = match.group(1).strip()  # Extract the matched content
                        delimiter = match.group(2)

                        if delimiter == "|\n\n":  # Standard table ending
                            current_table.append(content)
                            st.markdown("\n\n| " + " |\n\n| ".join(current_table) + " |\n\n")
                            current_table = []
                        elif delimiter == "\n\n|":  # Start of a new table
                            if current_table:
                                st.markdown("\n\n| " + " |\n\n| ".join(current_table) + " |\n\n")
                            current_table = [content]

                    # Handle leftover content if the loop ends mid-table
                    if current_table:
                        st.markdown("\n\n| " + " |\n\n| ".join(current_table) + " |\n\n")

if __name__ == "__main__":
    main()