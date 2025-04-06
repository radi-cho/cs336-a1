from typing import Iterable

def read_text_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        content = file.read()
    return content

def read_text_lines(file_path: str) -> Iterable[str]:
    with open(file_path, 'r') as file:
        for line in file:
            yield line
