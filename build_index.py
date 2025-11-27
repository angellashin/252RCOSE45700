import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data"
INDEX_DIR = "index"

def load_documents(): #pdf 불러오기
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(DATA_DIR, filename)
            print(f"1.Loading{path}")
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()

            for d in pdf_docs:
                d.metadata["source"] = filename
            docs.extend(pdf_docs)
    return docs

def main(): 
    os.makedirs(INDEX_DIR, exist_ok=True)
    #load documents
    print("load documents")
    docs = load_documents()
    #문서 파싱
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 200,
        separators=["\n\n", "\n" , " ", ""], 
    )

    chunks = splitter.split_documents(docs)
    print(f"len(chunks)개로 분할")

    #KoSimCSE 계산
    embeddings = HuggingFaceEmbeddings(
        model_name = "jhgan/ko-sroberta-multitask"
    )
     #chroma 벡터스토어
    db = Chroma.from_documents(
         documents = chunks,
         embedding = embeddings,
         persist_directory=INDEX_DIR,
     )
    db.persist()

if __name__ == "__main__":
    main()