from fastapi import FastAPI  # FastAPI 프레임워크를 사용하여 API를 만듭니다.
from fastapi.responses import RedirectResponse  # 특정 URL로 리다이렉트하는 응답을 생성합니다.
from fastapi.middleware.cors import CORSMiddleware  # Cross-Origin Resource Sharing (CORS)를 처리하기 위한 미들웨어입니다.
from typing import List, Union  # 타입 힌트를 위해 List와 Union 타입을 사용합니다.

# langserve 관련 임포트
from langserve.pydantic_v1 import BaseModel, Field  # Pydantic을 사용하여 데이터 모델을 정의합니다.
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # LangChain에서 메시지 타입을 가져옵니다.
from langserve import add_routes  # FastAPI 앱에 LangServe 라우트를 추가합니다.

# 사용할 체인들을 임포트합니다.
from chain import chain  # 기본 체인
from chat import chain as chat_chain  # 대화형 체인
from translator import chain as EN_TO_KO_chain  # 번역 체인
from llm import llm as model  # LLM (Large Language Model)


app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/chat/playground")


add_routes(app, chain, path="/prompt")


class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )


add_routes(
    app,
    chat_chain.with_types(input_type=InputChat),
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

add_routes(app, EN_TO_KO_chain, path="/translate")

add_routes(app, model, path="/llm")



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)