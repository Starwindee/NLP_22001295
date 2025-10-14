import os

def load_raw_text_data(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text

def load_raw_dataset_json(file_path: str) -> list:
    import json

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data