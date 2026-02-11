import pytest
from sci_fi_explorer.app import SciFiExplorer
from langchain_core.documents import Document
from unittest.mock import patch, MagicMock


# Shared ScifiExplorer object and sample for the test units
@pytest.fixture
def explorer(tmp_path,monkeypatch):
    
    monkeypatch.setattr(
        "sci_fi_explorer.app.OpenAIEmbeddings",
        MagicMock()  # Replaces OpenAIEmbeddings with a mock object
    )
    monkeypatch.setattr(
        "sci_fi_explorer.app.ChatOpenAI",
        MagicMock() # Replaces ChatOpenAI with a mock object
    )

     # Temporary text data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    (data_dir / "dummy.txt").write_text(
        "This is a test sci-fi story.",
        encoding="utf-8",
    )
    (data_dir / "story1.txt").write_text(
        "This is story one.",
        encoding="utf-8",
    )
    (data_dir / "story2.txt").write_text(
        "This is story two.",
        encoding="utf-8",
    )

    # Temporary Chroma persistence directory
    db_dir = tmp_path / "chroma_db"
    db_dir.mkdir()

    return SciFiExplorer(
        filepath= str(data_dir),   # Folder containing .txt files
        persist_dir=str(db_dir),  # Folder for Chroma DB
        collection_name="test_collection",
    )


def test_build_retriever(explorer):

    mock_vs = MagicMock()
    explorer.vectorstore = mock_vs # Inject mock into the explorer, Mock vectorstore instead of building a real Chroma DB

    explorer.build_retriever() # Build retriever using the mocked vectorstore
    mock_vs.as_retriever.assert_called_once() # Check that retriever method was called
    assert explorer.retriever is not None  # Check retriever is assigned

 
def test_retrieve_docs(explorer):

    mock_docs = [
        Document(page_content="Doc one."),
        Document(page_content="Doc two.")
    ]
    explorer.retriever = MagicMock() # Replace the retriever with a mock
    explorer.retriever.invoke.return_value = mock_docs # declaring to the the mock what to return when invoke() is called.
    query = "What is sci-fi?"
    result = explorer.retrieve_docs(query)

  # Checking  the function returns a dictionary from the retrieved files
    assert result["question"] == query
    assert "Doc one." in result["context"]
    assert "Doc two." in result["context"]
    assert isinstance(result,dict)

def test_text_to_document(explorer, tmp_path):

    # Create a real temporary file
    test_file = tmp_path / "test_text.txt"
    test_file.write_text("hello world", encoding="utf-8")

    # Set filepath to point to this single file
    explorer.filepath = str(test_file)

    document = explorer.text_to_document()

    assert "hello world" in document[0].page_content 
    assert all(isinstance(doc, Document) for doc in document) # Checking the loaded txt is converted to document
    assert len(document) == 1 # Checking  we got 1 document after loading text file
                   
def test_texts_to_documents(explorer):

    docs = explorer.texts_to_documents()

    # Check all documents are loaded
    assert len(docs) == 3  # only .txt files ( three dummy txt shared files from fixture)
    assert all(isinstance(doc, Document) for doc in docs)

    # Check content
    contents = [doc.page_content for doc in docs]
    assert "This is story one." in contents
    assert "This is story two." in contents


def test_chunk_docs(explorer):
    
    text = "these are test strings for chunking_function" * 100
    docs = [Document(page_content=text)]
    chunks = explorer.chunk_document(docs)
    assert all(isinstance(chunk, Document) for chunk in chunks) # Checks that each chunk is a Document 
    assert len(chunks) > 1 # Checks if document was splitted in multiple Documents(chunks)
    assert all(len(c.page_content) <= 250 for c in chunks) # Checks if all the chunks  have the correct size
   

