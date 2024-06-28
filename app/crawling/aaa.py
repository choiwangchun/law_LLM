import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
import pyautogui

# CSV 파일 경로 설정
csv_file_path = "name.csv"


# 다운로드 폴더 설정 (선택 사항)
download_dir = "C:\\Users\\slek9\\PycharmProjects\\law_LLM\\app\\law_RAG_data\\pdf"

if download_dir:
    prefs = {"download.default_directory": download_dir}
    driver = webdriver.Chrome()
i = 4054

with open(csv_file_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row_number, row in enumerate(reader):
        if row_number < 4053:
            continue  # Skip rows before the 33rd
        law_serial_number = row['법령일련번호']

        url = f"https://www.law.go.kr/DRF/lawService.do?OC=choi.tensor&target=law&MST={law_serial_number}&type=HTML"
        driver.get(url)
        time.sleep(1)
        try:
            # 1. Wait for the frame to load
            WebDriverWait(driver, 10).until(
                EC.frame_to_be_available_and_switch_to_it((By.ID, 'lawService'))
            )
            time.sleep(0.5)

            # 2. Click the first save button ("저장")
            first_save_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, 'bdySaveBtn'))
            )
            first_save_button.click()
            time.sleep(0.5)

            # 3. Select the PDF radio button
            pdf_radio_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, 'FileSavePdf1'))
            )
            pdf_radio_button.click()
            time.sleep(0.5)

            # 4. Find and click the save button image using pyautogui
            button_x = 811  # Replace with the X-coordinate of the button
            button_y = 690  # Replace with the Y-coordinate of the button

            pyautogui.click(button_x, button_y)
            print(f"{law_serial_number} 다운로드 시작 {i}")
            i += 1
            time.sleep(2)

#5320
        except Exception as e:
                print(f"{law_serial_number} 다운로드 실패: {e}")


driver.quit()