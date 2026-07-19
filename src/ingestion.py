import os
from typing import List
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.file import PyMuPDFReader

def load_documents_from_directory(data_dir: str) -> List[Document]:
    """
    Loads PDF documents from a specified directory.
    
    Args:
        data_dir (str): The path to the folder containing PDF files.
        
    Returns:
        List[Document]: A list of LlamaIndex Document objects representing the loaded data.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")

    print(f"Loading documents from {data_dir}...")
    
    # SimpleDirectoryReader is a utility from LlamaIndex designed to easily ingest local files.
    # By default, for PDFs, it reads each page as a separate Document object.
    # It also automatically extracts useful metadata such as file_name, file_path, and page_label.
    # We explicitly use PyMuPDFReader to prevent extracting raw PDF binary garbage.
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        required_exts=[".pdf"], # We only want to load PDFs for now
        file_extractor={".pdf": PyMuPDFReader()},
        recursive=False
    )
    
    # Load the data into LlamaIndex Document objects
    documents = reader.load_data()
    
    print(f"Successfully loaded {len(documents)} document objects (pages).")
    return documents

if __name__ == "__main__":
    # Example usage for testing the module independently
    # Create the data directory if it doesn't exist
    test_dir = "../data"
    os.makedirs(test_dir, exist_ok=True)
    
    # We would need to put a sample PDF in the data folder to see results.
    # For the sake of the script running without errors when empty, we use a try-except.
    try:
        docs = load_documents_from_directory(test_dir)
        
        # Display the first document to inspect its properties and metadata
        if docs:
            print("\n--- Example Document Object ---")
            sample_doc = docs[0]
            print(f"Document ID: {sample_doc.doc_id}")
            print(f"Metadata: {sample_doc.metadata}")
            # Truncating text for display purposes
            print(f"Content Snippet: {sample_doc.text[:200]}...") 
        else:
            print(f"No PDFs found in {test_dir}. Please add a PDF and run again to see the objects.")
    except Exception as e:
        print(f"Error loading documents: {e}")
