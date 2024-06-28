import os
import csv
from tqdm import tqdm


def compare_files(csv_path, folder_path):
    # Read the CSV file
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_numbers = [row[0] for row in csv_reader]  # Assuming the number is in the first column

    # Get the list of files in the folder
    folder_files = os.listdir(folder_path)

    missing_files = []
    empty_files = []

    # Use tqdm to show progress
    for number in tqdm(csv_numbers, desc="Checking files"):
        file_found = False
        for file in folder_files:
            if number in file:
                file_found = True
                file_path = os.path.join(folder_path, file)
                if os.path.getsize(file_path) == 0:
                    empty_files.append(number)
                break

        if not file_found:
            missing_files.append(number)

    print(f"\nTotal files in CSV: {len(csv_numbers)}")
    print(f"Total files in folder: {len(folder_files)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Empty files: {len(empty_files)}")

    print("\nMissing file numbers:")
    print(", ".join(missing_files))

    print("\nEmpty file numbers:")
    print(", ".join(empty_files))


# Usage
input_file_path = 'C:\\Users\\slek9\\PycharmProjects\\law_LLM\\app\\law_RAG_data\\check.csv'
folder_path = 'C:\\Users\\slek9\\PycharmProjects\\law_LLM\\app\\law_RAG_data\\pdf'
compare_files(input_file_path, folder_path)