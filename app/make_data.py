from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-pro")

for i in range(30):
    input = llm.invoke("일상 생활에서 발생할 수 있는 법적 사례를 만들어 내서 1문장의 질문을 만들어 보세요. 예를 들어, 뒤에서 오던 차가 내 차를 박았다면 어떻게 해야 할까요? 누군가 나를 모욕하면 어떻게 대응해야 할까요? 모르는 사람과 싸움이 났다면 어떻게 해야 할까요?")
    output = llm.invoke(input.content + "라는 질문에 대해서 다음과 같은 양식으로 법률 상담을 해주세요. \n해당 사건 요약: \n해당 사건에 대한 법 조항: [참조 조문] \n해결방법: \n참조 판례: [참조 판례] \n결론: \n저는 법률 전문가가 아닌 법률 도우미 입니다. 그러니 반드시 법률 전문가와 상담하여 소송 절차를 진행하는 것이 좋습니다.(반드시 포함)")

    # 엑셀 파일 이름
    excel_file = 'legal_questions.xlsx'

    # 새로운 데이터
    data = {'input': [input.content],
            'output': [output.content]}

    # 엑셀 파일이 존재하는지 확인
    if os.path.exists(excel_file):
        # 기존 파일이 있다면 데이터를 읽어옴
        df = pd.read_excel(excel_file)
        # 새 데이터를 추가
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    else:
        # 파일이 없다면 새로운 DataFrame 생성
        df = pd.DataFrame(data)

    # 데이터를 엑셀 파일로 저장
    df.to_excel(excel_file, index=False)

print(f"Data has been saved to {excel_file}")