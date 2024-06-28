from langchain_community.chat_models import ChatOllama  # LangChain에서 제공하는 다양한 챗 모델 중 Ollama 모델을 사용하기 위해 임포트합니다.
from langchain_core.output_parsers import StrOutputParser  # LLM의 출력을 문자열로 변환하는 Output Parser입니다.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 챗 모델에 사용할 프롬프트를 정의하고 관리하는 클래스입니다.

# LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
llm = ChatOllama(model="EEVE-Korean-10.8B:latest")

# Prompt 설정
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a useful AGI, your name is 'Jarvis'. You must answer in Korean.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# LangChain 표현식 언어 체인 구문을 사용합니다.
chain = prompt | llm | StrOutputParser()



