from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


template = """당신은 유능한 한국 법률 상담가입니다. 주어진 상황에 대해 아래 [FORMAT]에 따라 법률 상담을 제공해 주세요.

상황: {Situation}

[FORMAT]
1. 사건 요약:
2. 관련 법 조항:
3. 가능한 해결 방법:
4. 참조 판례 (있는 경우):
5. 결론:
6. 주의사항: 본 상담 내용은 일반적인 법률 정보 제공을 목적으로 합니다. 구체적인 법적 조치를 위해서는 반드시 변호사 등 법률 전문가와 상담하시기 바랍니다.
"""

prompt = PromptTemplate.from_template(template)
llm = ChatOllama(model="EEVE-Korean-10.8B:latest")  # 또는 사용하려는 모델 이름
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

answer = chain.invoke({"Situation": "누군가 뒤에서 이유 없이 때렸어요."})
print(answer)