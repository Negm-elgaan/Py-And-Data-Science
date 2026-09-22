# Import library
from langchain_community.document_loaders import PyPDFLoader as PyPDFL

# Create a document loader for rag_paper.pdf
loader = PyPDFL(file_path = "rag_paper.pdf")

# Load the document
data = loader.load()
print(data[0])
#####################
from langchain_community.document_loaders import UnstructuredHTMLLoader as USHTMLL
# Create a document loader for unstructured HTML
loader = USHTMLL(file_path = "datacamp-blog.html")

# Load the document
data = loader.load()

# Print the first document's content
print(data[0].page_content)

# Print the first document's metadata
print(data[0].metadata)