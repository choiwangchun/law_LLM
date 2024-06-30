from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="EEVE-Korean-10.8B:latest")  # 또는 사용하려는 모델 이름

answer = llm.stream("대한민국의 아름다운 관광지 10곳과 주소를 알려주세요!")

for token in answer:
    print(token.content, end="", flush=True)