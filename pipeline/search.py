import os
from dotenv import load_dotenv
from pipeline.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    import os

    def __init__(self, persist_dir: str = None, embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        # Si persist_dir n'est pas fourni, utiliser un chemin relatif au fichier actuel
        if persist_dir is None:
            # Obtenir le dossier où se trouve CE fichier
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            # Remonter d'un niveau et aller dans faiss_store
            persist_dir = os.path.join(current_file_dir, "..", "faiss_store")
            # Normaliser le chemin (résoudre les ..)
            persist_dir = os.path.normpath(persist_dir)
        
        print(f"[DEBUG] Looking for vector store in: {os.path.abspath(persist_dir)}")
        
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        
        print(f"[DEBUG] Checking for:\n  - {faiss_path}\n  - {meta_path}")
        print(f"[DEBUG] Files exist: faiss={os.path.exists(faiss_path)}, meta={os.path.exists(meta_path)}")
        
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print("**[INFO] Building vector store as it does not exist...")
            from pipeline.data_loader import load_all_documents
            
            # Même correction pour le chemin data
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
            data_dir = os.path.normpath(data_dir)
            print(f"[DEBUG] Loading documents from: {os.path.abspath(data_dir)}")
            
            docs = load_all_documents(data_dir)
            self.vectorstore.build_from_documents(docs)
        else:
            print("[INFO] Loading existing vector store...")
            self.vectorstore.load()
        
        groq_api_key = os.getenv("API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)