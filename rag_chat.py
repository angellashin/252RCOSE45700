import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from openai import OpenAI

# 🔹 0) 환경변수 로드 (여기까진 가벼움)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")

client = OpenAI(api_key=OPENAI_API_KEY)

# 전역 대화 히스토리 (질문, 답변)
chat_history: List[Tuple[str, str]] = []


def init_rag():
    """
    RAG에서 무거운 초기화 (임베딩, Chroma 로드)를 여기서만 수행.
    """
    print("1) 한국어 임베딩(KoSimCSE) 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask"
    )
    print("   ✅ 임베딩 로드 완료")

    print("2) Chroma 벡터스토어(index 폴더) 로드 중...")
    db = Chroma(
        persist_directory="index",
        embedding_function=embeddings,
    )
    print("   ✅ Chroma 로드 완료")

    retriever = db.as_retriever(search_kwargs={"k": 3})
    print("3) Retriever 준비 완료")

    return retriever


def build_messages(question: str, context_text: str):
    messages = [
        {
            "role": "system",
            "content": (
                "너는 대학 강의계획서를 기반으로 답변하는 한국어 AI 튜터야. "
                "반드시 제공된 문서 정보만 사용해서 답변하고, "
                "문서에 없는 내용은 모른다고 말해야 해."
            ),
        }
    ]

    # 과거 대화 히스토리 반영
    for user_msg, bot_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    # 이번 질문 + 컨텍스트
    messages.append(
        {
            "role": "user",
            "content": (
                "다음은 강의계획서에서 검색된 관련 내용이야:\n"
                f"{context_text}\n\n"
                f"위 내용을 참고해서, 다음 질문에 한국어로 자세히 답변해줘:\n{question}"
            ),
        }
    )

    return messages


def ask(retriever, question: str) -> str:
    print("   🔎 관련 문서 검색 중...")
    docs = retriever.get_relevant_documents(question)

    if not docs:
        answer = "관련된 문서를 찾지 못했어요. 질문을 조금 더 구체적으로 해 줄 수 있을까요?"
        chat_history.append((question, answer))
        return answer + "\n\n📚 출처:\n(검색 결과 없음)"

    context_texts = []
    sources = []
    for d in docs:
        context_texts.append(d.page_content)
        src = d.metadata.get("source")
        if src and src not in sources:
            sources.append(src)

    context_str = "\n\n---\n\n".join(context_texts)

    print("   🤖 OpenAI에 요청 보내는 중...")
    messages = build_messages(question, context_str)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()

    chat_history.append((question, answer))

    source_lines = "\n".join(f"- {s}" for s in sources)
    return f"{answer}\n\n📚 출처:\n{source_lines}"


if __name__ == "__main__":
    print("💬 RAG 챗봇 초기화 시작...")

    # ✅ 무거운 초기화는 여기에서만!
    retriever = init_rag()

    print("\n✅ RAG 초기화 완료!")
    print("💬 RAG 챗봇 시작! 'quit' 또는 'exit' 입력 시 종료")

    while True:
        q = input("\n질문: ")
        if q.lower() in ["quit", "exit"]:
            print("👋 종료합니다.")
            break

        try:
            response = ask(retriever, q)
            print("\n" + response)
        except Exception as e:
            print(f"\n⚠️ 에러 발생: {e}")
