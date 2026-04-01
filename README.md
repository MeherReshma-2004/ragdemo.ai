# ragdemo.ai
Retrieval Augmented Generation (RAG) is a technique that enhances the capabilties of LLM's by combining information retreival with text generation. Instead of relying on pre-trained knowledge. RAG fetch relevant data from external source and use it to generate more accurate responses.

streamlit
python-dotenv
PyPDF2
google-generativeai

langchain # core framework
langchain-huggingface # connect huggingface algorithms / models to perform embedding
faiss-cpu # Fast vector database to store the embedded data
langchain-community # extra integration 
langchain-text-splitters # split large texts into smaller chunks
sentence-transformers # call an pretrain embedding model to convert text into vectors
langchain-core # document, chains etc..

#### in simple words

How RAG works?
text -> split text -> convert vector -> store in database -> search similar content -> send to LLM -> get answers for questions

'all-MiniLM-L6-v2' --> simple hugging face embedding model which splits the text and converts the text into vectors