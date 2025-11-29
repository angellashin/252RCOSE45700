from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS # Chroma 대신 FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


#FAISS 벡터 스토어 로드

vectorstore = FAISS.load_local(
    "faiss_index", 
    OpenAIEmbeddings(), 
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# RAG 체인 구성
contextualize_q_system_prompt = """
이전 대화 내용과 최신 사용자 질문이 주어졌을 때, 
이 문맥을 참고하여 질문을 이해할 수 있는 독립적인 질문으로 수정하세요.
질문에 바로 답하지 말고, 수정된 질문만 반환하세요.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """
당신은 대학 강의계획서에 기반하여 질문에 답변하는 AI 조교입니다.
아래의 [Context]에 있는 내용만 사용하여 답변하세요.
답변의 끝에는 반드시 참조한 문서의 파일명(Source)을 명시해야 합니다.
모르는 내용은 지어내지 말고, 정보가 없다고 정중히 말하세요.

[Context]
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 메모리 및 채팅
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async def generate():
        response_generator = conversational_rag_chain.astream(
            {"input": request.message},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        sources = set()
        
        async for chunk in response_generator:
            if "answer" in chunk:
                yield chunk["answer"]
            
            if "context" in chunk:
                for doc in chunk["context"]:
                    src = doc.metadata.get("source", "알 수 없는 출처")
                    src_filename = os.path.basename(src)
                    page = doc.metadata.get("page", "")
                    source_info = f"{src_filename} (p.{page + 1})" if page != "" else src_filename
                    sources.add(source_info)
        
        if sources:
            yield "\n\n---\n**[참조 문서]**\n"
            for src in sources:
                yield f"- {src}\n"

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/")
def read_root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")