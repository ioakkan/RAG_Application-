import pytest
from sci_fi_explorer.app import SciFiExplorer
from langchain_core.documents import Document
from unittest.mock import patch, MagicMock


#Shared ScifiExplorer object for all the test units
@pytest.fixture
def explorer(tmp_path,monkeypatch ):
    
    monkeypatch.setattr(
        "sci_fi_explorer.app.OpenAIEmbeddings",
        MagicMock()
    )
    monkeypatch.setattr(
        "sci_fi_explorer.app.ChatOpenAI",
        MagicMock()
    )
    return SciFiExplorer(
        filepath= None,
        persist_dir=tmp_path,
        collection_name="test",
    )


def test_build_retriever(explorer):
    mock_vs = MagicMock()
    explorer.vectorstore = mock_vs

    explorer.build_retriever()

    mock_vs.as_retriever.assert_called_once()
    assert explorer.retriever is not None 

 
def test_retrieve_docs(explorer):
    mock_docs = [
        Document(page_content="Doc one."),
        Document(page_content="Doc two.")
    ]

    explorer.retriever = MagicMock()
    explorer.retriever.invoke.return_value = mock_docs

    query = "What is sci-fi?"
    result = explorer.retrieve_docs(query)

  # checking  the function returns a dictionary from the retrieved files
    assert result["question"] == query
    assert "Doc one." in result["context"]
    assert "Doc two." in result["context"]
 

def test_text_to_document(explorer, tmp_path):
    # creating a real temporary file
    test_file = tmp_path / "test_text.txt"
    test_file.write_text("This is Sci-Fi world", encoding="utf-8")
    
    explorer.filepath =  str(test_file)
    document = explorer.text_to_document()
    assert "This is Sci-Fi world" in document[0].page_content 
    assert all(isinstance(doc, Document) for doc in document) #cheicking the loaded txt is converted to doc
    assert len(document) == 1 # checking  we got 1 document after loading text file
                   


def test_chunk_docs(explorer):
    
    text = "these are test strings for chunking_function" * 100
    docs = [Document(page_content=text)]
    chunks = explorer.chunk_document(docs)
    assert all(isinstance(chunk, Document) for chunk in chunks) #checks the chunks are doc objects
    assert len(chunks) > 1 #checks if doc was splitted in multiple docs
    assert all(len(c.page_content) <= 250 for c in chunks) #checks if all the chunks  have the correct size
   

