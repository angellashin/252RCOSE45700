import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # Chroma 대신 FAISS 사용

load_dotenv()

def ingest_data():
    print("PDF 데이터 로딩 중...")
    
    loader = PyPDFDirectoryLoader("./data")
    docs = loader.load()
    
    if not docs:
        print("폴더에 PDF 파일이 없습니다.")
        return

    print(f"   - 총 {len(docs)} 페이지 로드 완료")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    print(f"   - 총 {len(splits)}개 청크로 분할")

    print("FAISS 벡터 인덱스 생성 중")
    
    # FAISS로 벡터 저장소 생성
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings()
    )
    
    # 로컬에 저장 (폴더명: faiss_index)
    vectorstore.save_local("faiss_index")
    print("(./faiss_index 폴더 생성됨)")

if __name__ == "__main__":
    ingest_data()