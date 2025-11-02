# to be called only if new files are added to the data directory
from pipeline.data_loader import load_all_documents
from pipeline.vectorstore import FaissVectorStore

if __name__ == "__main__":
    docs = load_all_documents("../data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)