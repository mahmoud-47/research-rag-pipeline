from pathlib import Path
from typing import List, Any, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import logging

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, 
    Docx2txtLoader, JSONLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedDocumentLoader:
    """
    Optimized document loader with parallel processing and lazy loading support.
    """
    
    LOADER_MAP = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.docx': Docx2txtLoader,
        '.json': JSONLoader,
    }
    
    def __init__(self, data_dir: str, max_workers: int = 4):
        """
        Initialize loader.
        
        Args:
            data_dir: Directory containing documents
            max_workers: Number of parallel workers for loading
        """
        self.data_path = Path(data_dir).resolve()
        self.max_workers = max_workers
        logger.info(f"Initialized loader for: {self.data_path}")
    
    def _collect_files(self) -> List[Path]:
        """Collect all supported files in one directory traversal."""
        supported_extensions = set(self.LOADER_MAP.keys())
        files = []
        
        for item in self.data_path.rglob('*'):
            if item.is_file() and item.suffix.lower() in supported_extensions:
                files.append(item)
        
        logger.info(f"Found {len(files)} supported files")
        return files
    
    def _load_single_file(self, file_path: Path) -> List[Any]:
        """Load a single file using appropriate loader."""
        extension = file_path.suffix.lower()
        loader_class = self.LOADER_MAP.get(extension)
        
        if not loader_class:
            logger.warning(f"No loader for {extension}: {file_path}")
            return []
        
        try:
            loader = loader_class(str(file_path))
            docs = loader.load()
            logger.debug(f"Loaded {len(docs)} docs from {file_path.name}")
            return docs
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []
    
    def load_all_parallel(self) -> List[Any]:
        """
        Load all documents in parallel.
        
        Returns:
            List of loaded documents
        """
        files = self._collect_files()
        documents = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._load_single_file, f): f 
                for f in files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    docs = future.result()
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} total documents")
        return documents
    
    def load_lazy(self) -> Generator[Any, None, None]:
        """
        Lazily load documents one at a time (memory efficient).
        
        Yields:
            Individual documents
        """
        files = self._collect_files()
        
        for file_path in files:
            docs = self._load_single_file(file_path)
            for doc in docs:
                yield doc
    
    def load_by_type(self, file_types: List[str]) -> List[Any]:
        """
        Load only specific file types.
        
        Args:
            file_types: List of extensions (e.g., ['.pdf', '.txt'])
        
        Returns:
            List of loaded documents
        """
        files = self._collect_files()
        filtered_files = [f for f in files if f.suffix.lower() in file_types]
        logger.info(f"Loading {len(filtered_files)} files of types {file_types}")
        
        documents = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._load_single_file, f) for f in filtered_files]
            for future in as_completed(futures):
                documents.extend(future.result())
        
        return documents


def load_all_documents(data_dir: str, max_workers: int = 4) -> List[Any]:
    """
    Convenience function matching original API.
    
    Args:
        data_dir: Directory containing documents
        max_workers: Number of parallel workers
    
    Returns:
        List of loaded documents
    """
    loader = OptimizedDocumentLoader(data_dir, max_workers=max_workers)
    return loader.load_all_parallel()


# Example usage
if __name__ == "__main__":
    # Method 1: Simple parallel loading (drop-in replacement)
    docs = load_all_documents("../data", max_workers=8)
    print(f"Loaded {len(docs)} documents")
    
    # Method 2: Using class for more control
    loader = OptimizedDocumentLoader("../data", max_workers=8)
    
    # Parallel loading
    all_docs = loader.load_all_parallel()
    print(f"Parallel load: {len(all_docs)} documents")
    
    # Lazy loading (memory efficient for large datasets)
    print("\nLazy loading first 5 docs:")
    for i, doc in enumerate(loader.load_lazy()):
        if i >= 5:
            break
        print(f"  - {doc.metadata.get('source', 'unknown')}")
    
    # Load only specific types
    pdf_docs = loader.load_by_type(['.pdf', '.txt'])
    print(f"\nLoaded {len(pdf_docs)} PDF/TXT documents")