from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


template = """당신은 유능한 법률 상담가 입니다. 상황에 맞는 [FORMAT]에 법률 상담을 해주세요.

상황:{Situation}

FORMAT:
- 해당 사건 요약:
- 해당 사건에 대한 법 조항:
- 해결 방법:
- 참조 판례:
- 결론:
- 저는 법률 전문가가 아닌 법률 도우미 입니다. 그러니 반드시 법률 전문가와 상담하여 소송 절차를 진행하는 것이 좋습니다.(반드시 포함)
"""

prompt = PromptTemplate.from_template(template)
llm = ChatOllama(model="EEVE-Korean-10.8B:latest")  # 또는 사용하려는 모델 이름
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

answer = chain.invoke({"Situation": "누군가 뒤에서 이유 없이 때렸어요."})
print(answer)