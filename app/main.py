import os
import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# ⭐️ Embedding 설정
# USE_BGE_EMBEDDING = True 로 설정시 HuggingFace BAAI/bge-m3 임베딩 사용 (2.7GB 다운로드 시간 걸릴 수 있습니다)
# USE_BGE_EMBEDDING = False 로 설정시 OpenAIEmbeddings 사용 (OPENAI_API_KEY 입력 필요. 과금)
USE_BGE_EMBEDDING = True

#if not USE_BGE_EMBEDDING:
    # OPENAI API KEY 입력
    # Embedding 을 무료 한글 임베딩으로 대체하면 필요 없음!
    # os.environ["OPENAI_API_KEY"] = ""

# ⭐️ LangServe 모델 설정(EndPoint)
# 1) REMOTE 접속: 본인의 REMOTE LANGSERVE 주소 입력
# (예시)
# LANGSERVE_ENDPOINT = "https://poodle-deep-marmot.ngrok-free.app/llm/"
LANGSERVE_ENDPOINT = "http://localhost:8000/chat/"

# 로컬 디렉토리 설정
LOCAL_DOCS_DIR = "C:\\Users\\slek9\\Desktop\\aa"

# 2) LocalHost 접속: 끝에 붙는 N4XyA 는 각자 다르니
# http://localhost:8000/llm/playground 에서 python SDK 에서 확인!
# LANGSERVE_ENDPOINT = "http://localhost:8000/llm/c/N4XyA"

# 필수 디렉토리 생성 @Mineru
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# # 프롬프트를 자유롭게 수정해 보세요!
# RAG_PROMPT_TEMPLATE = """당신은 전문 변호사 입니다. 검색된 다음 문맥을 이해해서 질문에 답하세요. 답을 모른다면 모른다고 답변하세요.
# Question: {question}
# Context: {context}
# Answer:"""


RAG_PROMPT_TEMPLATE = """You are a competent artificial intelligence legal assistant. Answer questions in the following format. Answer in your language and say you don't know if you don't know.

Question: {question}

Context: {context}

Case Summary:
[Summarize the key points of the case based on the question and context]

Relevant Statutory Provisions:
[statutory_provisions]

Analysis:
[Provide an objective analysis of the case based on the given information and relevant statutes]

Conclusion:
[Summarize the key points of the analysis]

IMPORTANT: I am a legal assistant AI, not a legal professional. This information is for informational purposes only. You should always consult with a qualified legal professional before proceeding with any legal matter or making any legal decisions.
"""

st.set_page_config(page_title="LawGPT", page_icon="🏛️")
st.title("⚖️ LawGPT")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?"),
    ]

with st.sidebar:
    st.link_button("🎁🙏 후원하기", "https://toss.me/lawgpt")

def print_history():
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)


def add_history(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource(show_spinner="Embedding file...")
def embed_files_from_directory(directory):
    """지정된 디렉토리에서 파일들을 읽어와 임베딩합니다."""
    retriever = None
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_extension = os.path.splitext(filepath)[1].lower()
            if file_extension in [".pdf", ".txt", ".docx"]:
                retriever = embed_file(filepath)
    return retriever

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file_path):
    """단일 파일을 임베딩합니다."""
    cache_dir = LocalFileStore(
        f"./.cache/embeddings/{os.path.basename(file_path)}"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=text_splitter)

    if USE_BGE_EMBEDDING:
        model_name = "BAAI/bge-m3"
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    else:
        embeddings = OpenAIEmbeddings()

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever





# with st.sidebar:
#     file = st.file_uploader(
#         "파일 업로드",
#         type=["pdf", "txt", "docx"],
#     )
#
# if file:
#     retriever = embed_file(file)

retriever = embed_files_from_directory(LOCAL_DOCS_DIR)

print_history()


if user_input := st.chat_input(placeholder="임대차 계약 만료 후 보증금 반환 문제 어떻게 해결하나요?"):
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        # ngrok remote 주소 설정
        ollama = RemoteRunnable(LANGSERVE_ENDPOINT)
        # LM Studio 모델 설정
        # ollama = ChatOpenAI(
        #     base_url="http://localhost:1234/v1",
        #     api_key="lm-studio",
        #     model="teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf",
        #     streaming=True,
        #     callbacks=[StreamingStdOutCallbackHandler()],  # 스트리밍 콜백 추가
        # )
        chat_container = st.empty()
        if retriever is not None:
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

            # 체인을 생성합니다.
            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | ollama
                | StrOutputParser()
            )
            # 문서에 대한 질의를 입력하고, 답변을 출력합니다.
            answer = rag_chain.stream(user_input)  # 문서에 대한 질의
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))

        else:
            prompt = ChatPromptTemplate.from_template(
                "다음의 질문에 간결하게 답변해 주세요:\n{input}"
            )

            # 체인을 생성합니다.
            chain = prompt | ollama | StrOutputParser()

            answer = chain.stream(user_input)  # 문서에 대한 질의
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))

