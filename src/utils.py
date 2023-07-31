import streamlit as st
import pickle
import numpy as np
import openai
from retrying import retry
import threading
import json
from typing import Union

import os

import requests

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Retry parameters
retry_kwargs = {
    "stop_max_attempt_number": 5,
    "wait_exponential_multiplier": 1000,
    "wait_exponential_max": 10000,
}


# Initialize Google Drive API
creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"])
drive_service = build('drive', 'v3', credentials=creds)


@retry(**retry_kwargs)
def download_file_from_gdrive(file_id: str) -> io.BytesIO:
    """Download a file from Google Drive.
    Args:
        file_id: The ID of the file on Google Drive.
    Returns:
        The downloaded file as a BytesIO object.
    """
    request = drive_service.files().get_media(fileId=file_id)
    downloaded = io.BytesIO()
    downloader = MediaIoBaseDownload(downloaded, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    downloaded.seek(0)
    return downloaded


def load_source_file(file_path: str, gdrive_file_id: str) -> Union[str, io.BytesIO]:
    """Load a source file either from a local path or from Google Drive.
    Args:
        file_path: The local path to the file.
        gdrive_file_id: The ID of the file on Google Drive.
    Returns:
        The local file path if the file exists locally,
        otherwise the downloaded file as a BytesIO object.
    """
    if os.path.exists(file_path):
        return file_path
    else:
        return download_file_from_gdrive(gdrive_file_id)


@retry(**retry_kwargs)
def generate_text_embedding(text: str, model="text-embedding-ada-002") -> np.ndarray:
    """Generate a text embedding using OpenAI.
    Args:
        text: The input text.
        model: The model to use for generating the embedding.
    Returns:
        The generated embedding as a numpy array.
    """
    text = text.replace("\n", " ")
    return openai.Embedding.create(
        input=[text], model=model
        )["data"][0]["embedding"]


def load_tag_vector(tag_vec_pickle: str) -> dict:
    """Load the tag vector from a pickle file.
    Returns:
        The loaded tag vector as a dictionary.
    """
    with open(tag_vec_pickle, "rb") as f:
        return pickle.load(f)


def create_query_vector(query_tags: list, tag_vector: dict) -> np.ndarray:
    """Create a query vector from a list of tags.
    Args:
        query_tags: The list of tags.
        tag_vector: The tag vector dictionary.
    Returns:
        The created query vector as a numpy array.
    """
    query_vector = [tag_vector[tag] for tag in query_tags]
    return sum(np.array(query_vector)) / len(query_vector)


def calculate_score(
    title_vec: np.ndarray,
    abstract_vec: np.ndarray,
    query_vector: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Calculate the score for a query vector.
    Args:
        title_vec: The title vector.
        abstract_vec: The abstract vector.
        query_vector: The query vector.
        alpha: The weight for the title score.
    Returns:
        The calculated score as a numpy array.
    """
    title_score = title_vec @ query_vector
    abstract_score = abstract_vec @ query_vector
    return alpha * title_score + (1 - alpha) * abstract_score


def chat_completion_request(
    messages: list, result: list, model="gpt-3.5-turbo-0613", functions=None
):
    """Send a chat completion request to OpenAI.
    Args:
        messages: The list of messages to send.
        result: The list to append the response to.
        model: The model to use for the chat completion.
        functions: The list of functions to use for the chat completion.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        result.append(response)
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")


# Refactored Code Continued

def generate_summary(title: str, abstract: str) -> str:
    """Generate a summary for a title and abstract using ChatGPT.
    Args:
        title: The title.
        abstract: The abstract.
    Returns:
        The generated summary as a string.
    """
    prompt = """
    以下の論文について何がすごいのか、次の項目を日本語で出力してください。

    (1)既存研究では何ができなかったのか。
    (2)どのようなアプローチでそれを解決しようとしたか
    (3)結果、何が達成できたのか


    タイトル: {title}
    アブストラクト: {abstract}
    日本語で出力してください。
    """.format(title=title, abstract=abstract)

    functions = [
        {
            "name": "format_output",
            "description": "アブストラクトのサマリー",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_of_existing_research": {
                        "type": "string",
                        "description": "既存研究では何ができなかったのか",
                    },
                    "how_to_solve": {
                        "type": "string",
                        "description": "どのようなアプローチでそれを解決しようとしたか",
                    },
                    "what_they_achieved": {
                        "type": "string",
                        "description": "結果、何が達成できたのか",
                    },
                },
                "required": [
                    "problem_of_existing_research",
                    "how_to_solve",
                    "what_they_achieved",
                ],
            },
        }
    ]

    messages = [{"role": "user", "content": prompt}]
    result: list = []
    thread = threading.Thread(
        target=chat_completion_request,
        args=(messages, result, "gpt-3.5-turbo-0613", functions)
    )
    thread.start()
    thread.join()

    if len(result) == 0:
        return "ChatGPTの結果取得に失敗しました...😢"

    res = result[0]
    func_result = res.json()["choices"][0]["message"]["function_call"]["arguments"]
    output = json.loads(func_result)
    a1 = output["problem_of_existing_research"]
    a2 = output["how_to_solve"]
    a3 = output["what_they_achieved"]
    gen_text = f"""以下の項目についてChatGPTが回答します。
    <ol>
        <li><b>既存研究では何ができなかったのか</b></li>
        <li style="list-style:none;">{a1}</li>
        <li><b>どのようなアプローチでそれを解決しようとしたか</b></li>
        <li style="list-style:none;">{a2}</li>
        <li><b>結果、何が達成できたのか</b></li>
        <li style="list-style:none;">{a3}</li>
    </ol>"""
    return gen_text
