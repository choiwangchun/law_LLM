import pandas as pd
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from tqdm import trange

url = "http://www.law.go.kr/DRF/lawSearch.do?OC=choi.tensor&target=law&type=XML"
response = urlopen(url).read()
xtree = ET.fromstring(response)

totalCnt = int(xtree.find('totalCnt').text)

rows = []
page = 1

for i in trange(int(totalCnt / 20)):

    for node in xtree:
        try:
            공포번호 = node.find('공포번호').text

            rows.append({'공포번호': 공포번호,
                         })

        except Exception as e:
            continue

    page += 1
    url = "http://www.law.go.kr/DRF/lawSearch.do?OC=ngho1202&target=law&type=XML&page={}".format(page)
    response = urlopen(url).read()
    xtree = ET.fromstring(response)

cases = pd.DataFrame(rows)
cases.to_csv('./check.csv', index=False)

