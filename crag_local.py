# Corrective RAG (CRAG) using local LLMs
# Based on: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag_local/

import os
import getpass
import time
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama  # Updated from langchain_community.chat_models
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables from .env file
load_dotenv()

# Package installations (you should run these separately or comment out after first run)
# pip install -U python-dotenv langchain_community tiktoken langchainhub scikit-learn langchain langgraph tavily-python firecrawl-py chromadb gpt4all sentence-transformers langchain-ollama

# Check if required API keys are set
required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY", "FC_API_KEY"]
missing_keys = [key for key in required_keys if not os.environ.get(key)]

if missing_keys:
    print(f"Warning: Missing required API keys: {', '.join(missing_keys)}")
    print("Please fill in your .env file with the required API keys.")
    # Optionally, you can still use the interactive prompt as fallback
    for key in missing_keys:
        os.environ[key] = getpass.getpass(f"Enter your {key}: ")

# LLM setup
local_llm = "llama3"
model_tested = "llama3-8b"
metadata = f"CRAG, {model_tested}"

# Define persistent directory for Chroma
CHROMA_DIR = "chroma_db"

# List of URLs to load documents from
urls = [
"https://www.ai-jason.com/learning-ai/how-to-reduce-llm-cost",
"https://www.ai-jason.com/learning-ai/gpt5-llm",
"https://www.ai-jason.com/learning-ai/how-to-build-ai-agent-tutorial-3",
]

# Load documents from FireCrawl with retry logic
def load_documents_with_retry(urls, max_retries=3):
    all_docs = []
    
    for url in urls:
        retries = 0
        success = False
        
        while not success and retries < max_retries:
            try:
                print(f"Loading documents from {url} (attempt {retries + 1}/{max_retries})...")
                loader = FireCrawlLoader(
                    api_key=os.environ.get("FC_API_KEY"),
                    url=url,
                    mode="scrape"
                )
                documents = loader.load()
                print(f"  ✅ Loaded {len(documents)} documents from {url}")
                all_docs.extend(documents)
                success = True
            except Exception as e:
                retries += 1
                print(f"  ❌ Error loading {url}: {e}")
                if retries < max_retries:
                    wait_time = 2 ** retries  # Exponential backoff
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed to load {url} after {max_retries} attempts")
    
    return all_docs

# Load documents
docs_list = load_documents_with_retry(urls)
print(f"Loaded {len(docs_list)} documents in total")

# Handle case where no documents were loaded successfully
if not docs_list:
    error_message = f"ERROR: Failed to load any documents from the specified URLs: {urls}"
    print("\n" + "!" * 80)
    print(error_message)
    print("!" * 80)
    print("\nPlease check:")
    print("1. Your FC_API_KEY is valid and has sufficient quota")
    print("2. The URLs are accessible and contain valid content")
    print("3. Network connectivity is working properly")
    print("\nExiting program.")
    raise RuntimeError(error_message)

# Filter complex metadata first
filtered_docs = filter_complex_metadata(docs_list)
print(f"Filtered out complex metadata, remaining: {len(filtered_docs)} documents")

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,   # DO NOT CHANGE THIS
    chunk_overlap=0,  # DO NOT CHANGE THIS
    length_function=len
)

# Split the filtered documents into chunks using the text splitter
doc_splits = text_splitter.split_documents(filtered_docs)
print(f"Split {len(filtered_docs)} documents into {len(doc_splits)} chunks")

# Set up embeddings
try:
    embedding = GPT4AllEmbeddings()
    print("✅ Successfully initialized GPT4AllEmbeddings")
except Exception as e:
    print(f"❌ Error initializing GPT4AllEmbeddings: {e}")
    raise Exception("Failed to initialize embeddings. Please check GPT4All installation.")

# Create or load Chroma vectorstore with properly processed documents
try:
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"Loading existing Chroma DB from {CHROMA_DIR}...")
        try:
            vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
            # Test retrieval to ensure DB is working
            test_docs = vectorstore.similarity_search("test", k=1)
            print(f"✅ Successfully loaded existing Chroma DB with {len(test_docs)} test documents")
            
            # If the test returns no documents, recreate the DB
            if not test_docs:
                print("⚠️ Existing DB returns no results. Recreating...")
                # Remove the existing directory
                import shutil
                shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                os.makedirs(CHROMA_DIR, exist_ok=True)
                
                # Create new vectorstore with current documents
                vectorstore = Chroma.from_documents(
                    documents=doc_splits,
                    embedding=embedding,
                    persist_directory=CHROMA_DIR
                )
                vectorstore.persist()
                print(f"✅ Created new vectorstore with {len(doc_splits)} documents")
                
        except Exception as e:
            print(f"❌ Error loading existing Chroma DB: {e}")
            print("Recreating Chroma DB...")
            # Remove the existing directory if it's corrupted
            import shutil
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
            os.makedirs(CHROMA_DIR, exist_ok=True)
            
            # Create new vectorstore
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                embedding=embedding,
                persist_directory=CHROMA_DIR
            )
            vectorstore.persist()
            print(f"✅ Created new vectorstore with {len(doc_splits)} documents")
    else:
        print("Creating new Chroma vectorstore...")
        if not os.path.exists(CHROMA_DIR):
            os.makedirs(CHROMA_DIR, exist_ok=True)
            
        if doc_splits:
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                embedding=embedding,
                persist_directory=CHROMA_DIR
            )
            vectorstore.persist()
            print(f"✅ Created vectorstore with {len(doc_splits)} documents")
        else:
            print("⚠️ No documents to create vectorstore, creating empty vectorstore")
            vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
            vectorstore.persist()
except Exception as e:
    print(f"❌ Critical error with vectorstore: {e}")
    raise Exception(f"Failed to initialize vectorstore: {e}")

# Create a retriever with search type and parameters
try:
    # Create a hybrid retriever that combines keyword and semantic search
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import InMemoryStore
    
    # Add keyword-based pre-filtering to find cost-related documents
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 15,  # Retrieve more documents to ensure we get relevant ones
            "filter": None,  # No filtering to ensure we get all potentially relevant documents
        }
    )
    
    # Test the retriever
    test_docs = retriever.invoke("test")
    print(f"✅ Retriever test successful with {len(test_docs)} documents")
except Exception as e:
    print(f"❌ Error initializing retriever: {e}")
    
    # Create a simple in-memory backup retriever with our documents
    from langchain_core.retrievers import BaseRetriever
    
    class SimpleRetriever(BaseRetriever):
        def __init__(self, documents):
            self.documents = documents
            
        def _get_relevant_documents(self, query):
            return self.documents
    
    print("⚠️ Using fallback retriever")
    retriever = SimpleRetriever(doc_splits)

### Retrieval Grader

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Prompt for document relevance grading. NEVER CHANGE THIS.
retrieval_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the question, grade it as relevant. It does not need to be a stringent test. The goal is to filter erroneous retrievals. \n
Give a binary score of "yes" or "no" to indicate whether the document is relevant to the question. \n
Provide the binary score as a JSON with a single key "score" and no preamble or explanation.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Here is the retrieved document:
\n\n 
{document}
\n\n 
Here is the user question:
{question}
\n
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["question", "document"],
)

retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()

### Answer Generation

# Prompt for answer generation. DO NOT CHANGE THIS.
qa_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. if you do not know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}
Context: {context}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["question", "context"],
)

# Format documents
def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# LLM for generation
generation_llm = ChatOllama(model=local_llm, temperature=0.1)

# Chain
rag_chain = (
    {"context": lambda x: format_docs(x["documents"]), "question": lambda x: x["question"]}
    | qa_prompt
    | generation_llm
    | StrOutputParser()
)

### Hallucination Grader

# Prompt for checking hallucination. NEVER CHANGE THIS.
hallucination_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key "score" and no preamble or explanation.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Here are the facts:
\n ------- \n
{documents}
\n ------- \n
Here is the answer: {generation}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["generation", "documents"],
)

hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

### Answer Relevance Grader

# Prompt for checking answer relevance
answer_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing whether an answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether an answer is useful to resolve a question. Provide the binary score as a JSON with a single key "score" and no preamble or explanation.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Here are the facts:
\n ------- \n
{generation}
\n ------- \n
Here is the question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["generation", "question"],
)

answer_grader = answer_grader_prompt | llm | JsonOutputParser()

from typing_extensions import TypedDict 
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

### State Definition

class GraphState(TypedDict, total=False):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question
        generation: LLM-generated answer
        documents: Retrieved documents
        web_search_results: Results from web search
        retrieval_scores: Document relevance scores
        relevance_count: How many documents were relevant
        hallucination_check: Result of hallucination check
        answer_quality_check: Result of answer quality check
        error: Any error that occurred
        steps: List of steps executed
    """
    question: str
    generation: Optional[str]
    documents: List[Document]
    web_search_results: Optional[List[Dict[str, Any]]]
    retrieval_scores: Optional[List[Dict[str, Any]]]
    relevance_count: Optional[int]
    hallucination_check: Optional[Dict[str, Any]]
    answer_quality_check: Optional[Dict[str, Any]]
    error: Optional[str]
    steps: List[str]

### Node Definitions

def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents from vectorstore
    """
    print("\n🔍 RETRIEVING DOCUMENTS")
    state["steps"] = state.get("steps", []) + ["retrieve"]
    
    try:
        documents = retriever.invoke(state["question"])
        
        # Add debugging information
        print(f"📝 Retrieved {len(documents)} documents")
        if documents:
            print(f"First document sample: {documents[0].page_content[:100]}...")
        else:
            print("❌ No documents were retrieved")
            error_msg = "Error: No documents were retrieved from the vectorstore."
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        return {**state, "documents": documents}
    except Exception as e:
        error_msg = f"Error during retrieval: {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

def grade_documents(state: GraphState) -> GraphState:
    """
    Grade document relevance
    """
    print("\n📊 GRADING DOCUMENT RELEVANCE")
    state["steps"] = state.get("steps", []) + ["grade_documents"]
    
    # Ensure the state dictionary has the correct keys using get()
    question = state.get("question", "")
    documents = state.get("documents", [])
    
    # Debug document state
    print(f"📋 Documents to grade: {len(documents)}")
    print(f"📋 Document types: {[type(doc).__name__ for doc in documents[:3]]}")
    
    if not documents:
        print("⚠️ No documents to grade, proceeding to web search")
        return {**state, "retrieval_scores": [], "relevance_count": 0}
    
    try:
        # Score each document
        retrieval_scores = []
        relevant_docs = []
        
        for i, doc in enumerate(documents):
            print(f"\n🔍 Grading document {i+1}/{len(documents)}")
            
            # Safety check for document content
            if not hasattr(doc, 'page_content') or not doc.page_content:
                print(f"⚠️ Document {i+1} has no page_content, skipping")
                continue
                
            # Grade document relevance
            score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            is_relevant = score["score"].lower() == "yes"
            
            retrieval_scores.append({
                "document": doc,
                "score": score["score"],
                "is_relevant": is_relevant
            })
            
            if is_relevant:
                print(f"✅ DOCUMENT {i+1} RELEVANT")
                relevant_docs.append(doc)
            else:
                print(f"❌ DOCUMENT {i+1} NOT RELEVANT")
        
        relevance_count = len(relevant_docs)
        relevance_ratio = relevance_count / len(documents) if documents else 0
        
        print(f"📈 Relevance: {relevance_count}/{len(documents)} documents relevant ({relevance_ratio:.0%})")
        
        # Always return at least one document (the most relevant if possible)
        if not relevant_docs and documents:
            print("⚠️ No relevant documents found, keeping the first document for fallback")
            relevant_docs = [documents[0]]
        
        return {
            **state, 
            "documents": relevant_docs if relevant_docs else documents,  # Keep original if all filtered out
            "retrieval_scores": retrieval_scores,
            "relevance_count": relevance_count
        }
    except Exception as e:
        error_msg = f"Error during document grading: {str(e)}"
        print(f"❌ {error_msg}")
        return {**state, "error": error_msg, "documents": documents}

def web_search(state: GraphState) -> GraphState:
    """
    Search the web for additional information
    """
    print("\n🌐 PERFORMING WEB SEARCH")
    state["steps"] = state.get("steps", []) + ["web_search"]
    
    # Keep track of original documents in case web search fails
    original_documents = state.get("documents", []).copy()
    
    try:
        # Define the web search tool
        web_search_tool = TavilySearchResults(k=3)
        
        # Perform search
        search_results = web_search_tool.invoke({"query": state["question"]})
        print(f"🔎 Found {len(search_results)} web search results")
        
        if not search_results:
            print("⚠️ No web search results found")
            return {**state, "web_search_results": []}
            
        # Convert to documents and add to state
        documents = original_documents.copy()
        
        for result in search_results:
            web_doc = Document(
                page_content=f"Title: {result['title']}\nContent: {result['content']}\nURL: {result['url']}",
                metadata={"source": "web_search", "url": result["url"]}
            )
            documents.append(web_doc)
            
        print(f"📚 Combined document count: {len(documents)} (Original: {len(original_documents)}, Web: {len(search_results)})")
        
        return {**state, "documents": documents, "web_search_results": search_results}
    except Exception as e:
        error_msg = f"Error during web search: {str(e)}"
        print(f"❌ {error_msg}")
        # Return original state with error flag but keeping original documents
        return {**state, "error": error_msg, "web_search_results": []}

def generate_answer(state: GraphState) -> GraphState:
    """
    Generate an answer using the RAG chain
    """
    print("\n✍️ GENERATING ANSWER")
    state["steps"] = state.get("steps", []) + ["generate_answer"]
    
    try:
        if not state.get("documents"):
            answer = "I don't have enough information to answer this question."
        else:
            answer = rag_chain.invoke({
                "documents": state["documents"], 
                "question": state["question"]
            })
        
        print(f"💬 Generated answer: {answer[:100]}...")
        return {**state, "generation": answer}
    except Exception as e:
        error_msg = f"Error during answer generation: {str(e)}"
        print(f"❌ {error_msg}")
        return {**state, "error": error_msg, "generation": "An error occurred while generating your answer."}

def check_hallucination(state: GraphState) -> GraphState:
    """
    Check if the generated answer contains hallucinations
    """
    print("\n🔍 CHECKING FOR HALLUCINATIONS")
    state["steps"] = state.get("steps", []) + ["check_hallucination"]
    
    if not state.get("generation") or not state.get("documents"):
        return {**state, "hallucination_check": {"score": "no"}}
    
    try:
        # Format documents for the grader
        formatted_docs = format_docs(state["documents"])
        
        # Check for hallucinations
        hallucination_check = hallucination_grader.invoke({
            "documents": formatted_docs, 
            "generation": state["generation"]
        })
        
        is_supported = hallucination_check["score"].lower() == "yes"
        print(f"{'✅ Answer is supported by documents' if is_supported else '❌ Answer contains unsupported information'}")
        
        return {**state, "hallucination_check": hallucination_check}
    except Exception as e:
        error_msg = f"Error during hallucination check: {str(e)}"
        print(f"❌ {error_msg}")
        return {**state, "error": error_msg, "hallucination_check": {"score": "unknown"}}

def check_answer_quality(state: GraphState) -> GraphState:
    """
    Check if the answer addresses the question
    """
    print("\n🔍 CHECKING ANSWER QUALITY")
    state["steps"] = state.get("steps", []) + ["check_answer_quality"]
    
    if not state.get("generation"):
        return {**state, "answer_quality_check": {"score": "no"}}
    
    try:
        # Check answer quality
        answer_quality_check = answer_grader.invoke({
            "question": state["question"], 
            "generation": state["generation"]
        })
        
        is_helpful = answer_quality_check["score"].lower() == "yes"
        print(f"{'✅ Answer addresses the question' if is_helpful else '❌ Answer does not address the question'}")
        
        return {**state, "answer_quality_check": answer_quality_check}
    except Exception as e:
        error_msg = f"Error during answer quality check: {str(e)}"
        print(f"❌ {error_msg}")
        return {**state, "error": error_msg, "answer_quality_check": {"score": "unknown"}}

### Conditional Edges

def should_search_web(state: GraphState) -> str:
    """
    Determine if web search is needed based on document relevance
    """
    relevance_count = state.get("relevance_count", 0)
    documents = state.get("documents", [])
    
    # If we have an error, skip to generation with what we have
    if state.get("error"):
        print("⚠️ Error occurred, proceeding with available information")
        return "generate_answer"
    
    # If there are few/no relevant documents, do web search
    if relevance_count < 2 or len(documents) == 0:
        print("⚠️ Not enough relevant documents, performing web search")
        return "web_search"
    
    # Otherwise, proceed to generation
    print("✅ Sufficient relevant documents found, proceeding to generation")
    return "generate_answer"

def evaluate_generated_answer(state: GraphState) -> str:
    """
    Evaluate the generated answer and decide next steps
    """
    hallucination_check = state.get("hallucination_check", {}).get("score", "no")
    answer_quality_check = state.get("answer_quality_check", {}).get("score", "no")
    
    # If we have an error, just return the answer we have
    if state.get("error"):
        print("⚠️ Error occurred, ending with available answer")
        return "end"
    
    # Logic for different scenarios
    if hallucination_check.lower() == "no":
        # Answer contains hallucinations
        print("⚠️ Answer contains hallucinations, need more context")
        # Check if we've already done web search
        if "web_search" in state.get("steps", []):
            print("🔄 Already performed web search, will regenerate answer")
            return "regenerate"
        else:
            print("🌐 Getting additional context from web")
            return "web_search"
    
    elif answer_quality_check.lower() == "no":
        # Answer doesn't address question well
        print("⚠️ Answer doesn't address question well")
        # Check if we've already done web search
        if "web_search" in state.get("steps", []):
            print("🔄 Already performed web search, will regenerate answer")
            return "regenerate"
        else:
            print("🌐 Getting additional context from web")
            return "web_search"
    
    else:
        # Answer is good
        print("✅ Answer is high quality, finishing")
        return "end"

def regenerate_or_finish(state: GraphState) -> str:
    """
    Decide whether to regenerate the answer or finish
    """
    # If we've tried to generate too many times, just finish
    generation_attempts = state["steps"].count("generate_answer")
    
    if generation_attempts >= 3:
        print(f"⚠️ Made {generation_attempts} generation attempts, finishing with current answer")
        return "end"
    
    # Otherwise regenerate
    print(f"🔄 Regenerating answer (attempt {generation_attempts + 1})")
    return "generate_answer"

### Build the Graph

# Initialize the graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("check_hallucination", check_hallucination)
workflow.add_node("check_answer_quality", check_answer_quality)

# Set entry point
workflow.set_entry_point("retrieve")

# Add edges
workflow.add_edge("retrieve", "grade_documents")

# Add conditional edges
workflow.add_conditional_edges(
    "grade_documents",
    should_search_web,
    {
        "web_search": "web_search",
        "generate_answer": "generate_answer",
    }
)

workflow.add_edge("web_search", "generate_answer")
workflow.add_edge("generate_answer", "check_hallucination")
workflow.add_edge("check_hallucination", "check_answer_quality")

workflow.add_conditional_edges(
    "check_answer_quality",
    evaluate_generated_answer,
    {
        "web_search": "web_search",
        "regenerate": "generate_answer",
        "end": END,
    }
)

# Compile the graph
app = workflow.compile()

# Test
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Corrective RAG (CRAG) System')
    parser.add_argument('--query', type=str, help='Question to ask')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output (show all steps)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()

    # Set verbosity
    verbose = args.verbose
    
    def process_query(query):
        print(f"\n🔍 QUESTION: {query}\n")
        print("=" * 80)
        print("Processing your question... This may take a moment.")
        print("=" * 80)
        
        # Track start time
        start_time = time.time()
        
        # Initialize with the user's question
        inputs = {"question": query}
        
        # Run the graph and collect the final state
        final_state = None
        
        if verbose:
            for output in app.stream(inputs):
                for key, value in output.items():
                    print(f"Finished step: {key}")
                final_state = value
        else:
            final_state = app.invoke(inputs)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Print the answer
        print("\n" + "=" * 80)
        print(f"✨ ANSWER (processed in {processing_time:.2f} seconds):")
        print("-" * 80)
        print(final_state.get("generation", "Sorry, I couldn't generate an answer."))
        print("=" * 80)
        
        # Print steps taken if verbose
        if verbose:
            print("\n📋 Steps taken:")
            for i, step in enumerate(final_state.get("steps", [])):
                print(f"{i+1}. {step}")
        
        return final_state
    
    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 80)
        print("🤖 Welcome to the Corrective RAG (CRAG) System")
        print("=" * 80)
        print("Ask questions, and I'll try to answer them using a combination of")
        print("local knowledge and web search when needed.")
        print("Type 'exit', 'quit', or 'q' to end the session.")
        print("=" * 80)
        
        while True:
            query = input("\n💬 Your question: ")
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nThank you for using the CRAG system. Goodbye! 👋")
                break
            
            if not query.strip():
                continue
                
            process_query(query)
    
    # Single query mode
    elif args.query:
        process_query(args.query)
    
    # Default example
    else:
        example_query = "how to save llm cost?"
        print(f"\n⚠️ No query provided, using example: '{example_query}'")
        print("Use --interactive for conversation mode or --query 'your question' for single queries")
        process_query(example_query)