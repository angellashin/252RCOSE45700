# main.py
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from openai import OpenAI

# ========================
# 0. 환경 설정
# ========================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다. .env 파일에 OPENAI_API_KEY=... 를 추가하세요.")

client = OpenAI(api_key=OPENAI_API_KEY)

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

# LangChain 임베딩 + 벡터스토어 전역 변수
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore: Optional[Chroma] = None
retriever = None

# 세션별 대화 히스토리: session_id -> [ {role, content}, ... ]
CHAT_HISTORY: Dict[str, List[Dict[str, str]]] = {}

app = FastAPI(title="KU RAG Chatbot")


# ========================
# 1. LangChain RAG 인덱스 빌드
# ========================

def build_vectorstore_if_needed():
    """
    - data/ 폴더의 PDF들을 LangChain으로 로드
    - 텍스트 청크로 나눈 뒤
    - OpenAIEmbeddings + Chroma 벡터스토어 생성
    - 이미 chroma_db가 있으면 재사용
    """
    global vectorstore, retriever

    if retriever is not None and vectorstore is not None:
        # 이미 초기화된 경우
        return

    if not DATA_DIR.exists():
        raise RuntimeError("data 폴더를 찾을 수 없습니다. 강의계획서 PDF들을 data/에 넣어주세요.")

    # 이미 저장된 벡터스토어가 있으면 그거 재사용
    if CHROMA_DIR.exists():
        print("📂 기존 Chroma 벡터스토어 로드 중...")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        print("✅ 기존 벡터스토어 로드 완료.")
        return

    # 없으면 새로 빌드
    print("📂 PDF 로드 시작 (LangChain PyPDFDirectoryLoader)...")
    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    documents = loader.load()
    print(f"✅ 문서 로드 완료. 문서 수: {len(documents)}")

    print("✂️ 텍스트 청크 분할 (LangChain RecursiveCharacterTextSplitter)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = splitter.split_documents(documents)
    print(f"✅ 청크 분할 완료. 청크 수: {len(splits)}")

    print("🧠 Chroma 벡터스토어 생성 (OpenAIEmbeddings 사용)...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("✅ 벡터스토어 생성 및 저장 완료.")


def get_history_text(session_id: str, max_turns: int = 5) -> str:
    history = CHAT_HISTORY.get(session_id, [])
    history = history[-max_turns * 2 :]  # 최근 max_turns 쌍만 유지
    CHAT_HISTORY[session_id] = history

    lines: List[str] = []
    for msg in history:
        role = "사용자" if msg["role"] == "user" else "AI"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def add_to_history(session_id: str, role: str, content: str):
    if session_id not in CHAT_HISTORY:
        CHAT_HISTORY[session_id] = []
    CHAT_HISTORY[session_id].append({"role": role, "content": content})


# ========================
# 2. RAG 질의 (비-스트리밍)
# ========================

def rag_answer(question: str, session_id: str) -> Dict[str, Any]:
    """
    LangChain:
      - Chroma retriever로 관련 문서 검색
      - 멀티턴 히스토리 포함해서 OpenAI LLM 호출
      - 답변 + 출처 목록 반환
    """
    build_vectorstore_if_needed()

    # 관련 문서 검색 (LangChain retriever)
    docs = retriever.get_relevant_documents(question)

    context_parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        # PyPDFDirectoryLoader의 source는 전체 경로일 수 있음 → 파일명만 추출
        filename = os.path.basename(src) if src else "unknown"
        context_parts.append(f"[출처: {filename}, p.{page}]\n{d.page_content}\n")
        sources.append({"source": filename, "page": page})

    context_text = "\n\n".join(context_parts)
    history_text = get_history_text(session_id)

    system_prompt = (
        "당신은 대학 강의 계획서 관련 질문에 답하는 한국어 AI 어시스턴트입니다. "
        "반드시 제공된 문서(context)와 대화 히스토리만을 근거로 답변하세요. "
        "모르면 모른다고 말하세요."
    )

    user_content = f"""
이전 대화:
{history_text if history_text else '(이전 대화 없음)'}

사용자 질문:
{question}

관련 문서(context):
{context_text}

위 정보를 참고하여 자연스럽고 친절한 한국어로 답변해 주세요.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
    )
    answer_text = resp.choices[0].message.content

    add_to_history(session_id, "user", question)
    add_to_history(session_id, "assistant", answer_text)

    # 출처 텍스트 만들기
    sources_text_lines: List[str] = []
    seen = set()
    for s in sources:
        key = (s["source"], s["page"])
        if key in seen:
            continue
        seen.add(key)
        sources_text_lines.append(f"- {s['source']} (p.{s['page']})")
    sources_text = "\n".join(sources_text_lines)

    return {
        "answer": answer_text,
        "sources_text": sources_text,
    }


# ========================
# 3. RAG 스트리밍 버전
# ========================

def rag_stream_answer(question: str, session_id: str):
    """
    - LangChain retriever로 관련 문서 검색
    - OpenAI ChatCompletion(stream=True)로 토큰 단위 스트리밍
    - 마지막에 [참고 문서] 섹션 함께 전송
    """
    build_vectorstore_if_needed()

    docs = retriever.get_relevant_documents(question)

    context_parts: List[str] = []
    sources: List[Dict[str, Any]] = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        filename = os.path.basename(src) if src else "unknown"
        context_parts.append(f"[출처: {filename}, p.{page}]\n{d.page_content}\n")
        sources.append({"source": filename, "page": page})

    context_text = "\n\n".join(context_parts)
    history_text = get_history_text(session_id)

    system_prompt = (
        "당신은 대학 강의 계획서 관련 질문에 답하는 한국어 AI 어시스턴트입니다. "
        "반드시 제공된 문서(context)와 대화 히스토리만을 근거로 답변하세요. "
        "모르면 모른다고 말하세요."
    )

    user_content = f"""
이전 대화:
{history_text if history_text else '(이전 대화 없음)'}

사용자 질문:
{question}

관련 문서(context):
{context_text}

위 정보를 참고하여 자연스럽고 친절한 한국어로 답변해 주세요.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    full_answer = ""

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
        stream=True,
    )

    # 본문 토큰 스트리밍
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_answer += delta
        yield delta

    # 히스토리에 저장
    add_to_history(session_id, "user", question)
    add_to_history(session_id, "assistant", full_answer)

    # 출처 텍스트
    sources_text_lines: List[str] = []
    seen = set()
    for s in sources:
        key = (s["source"], s["page"])
        if key in seen:
            continue
        seen.add(key)
        sources_text_lines.append(f"- {s['source']} (p.{s['page']})")
    sources_text = "\n".join(sources_text_lines)

    footer = "\n\n[참고 문서]\n" + sources_text
    yield footer


# ========================
# 4. FastAPI: 정적 파일 (UI) + API 엔드포인트
# ========================

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "RAG 챗봇이 실행 중입니다. /static/index.html을 만들어 UI를 추가하세요."}


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources_text: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = rag_answer(req.question, req.session_id)
        return ChatResponse(
            answer=result["answer"],
            sources_text=result["sources_text"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    try:
        generator = rag_stream_answer(req.question, req.session_id)
        return StreamingResponse(generator, media_type="text/plain; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
