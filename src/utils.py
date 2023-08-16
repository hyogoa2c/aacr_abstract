import io
import json
import os
import pickle
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
import openai
import requests
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from retrying import retry

# Retry parameters
retry_kwargs = {
    "stop_max_attempt_number": 5,
    "wait_exponential_multiplier": 1000,
    "wait_exponential_max": 10000,
}


class ApiKeyManager:
    def __init__(self, secrets: Dict[str, str]):
        self.secrets = secrets

    def get_openai_key(self):
        return self.secrets["OPENAI_API_KEY"]

    def get_gcp_service_account(self):
        return self.secrets["gcp_service_account"]


class AbstractApiHandler(ABC):
    @abstractmethod
    def chat_completion_request(
        self, messages: list, result: list, model: str, functions: list
    ):
        pass

    @abstractmethod
    def generate_text_embedding(self, text: str, model: str):
        pass


class OpenAiHandler(AbstractApiHandler):
    def __init__(self, api_key_manager: ApiKeyManager):
        self.api_key_manager = api_key_manager

    def chat_completion_request(
        self, messages: list, result: list, model: str, functions: list
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key_manager.get_openai_key(),
        }
        json_data = {"model": model, "messages": messages, "functions": functions}
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
            raise

    @retry(**retry_kwargs)
    def generate_text_embedding(self, text: str, model: str):
        text = text.replace("\n", " ")
        try:
            response = openai.Embedding.create(
                input=[text], model=model
            )
            return response["data"][0]["embedding"]  # type: ignore
        except Exception as e:
            print("Unable to generate text embedding")
            print(f"Exception: {e}")
            raise


class AbstractFileHandler(ABC):
    @abstractmethod
    def download_file(self, file_id: str) -> io.BytesIO:
        pass

    @abstractmethod
    def load_source_file(
        self, file_path: str, gdrive_file_id: str
    ) -> Union[str, io.BytesIO]:
        pass

    @abstractmethod
    def load_pickle_file(self, file_path: str) -> Dict[str, str]:
        pass


class GoogleDriveHandler(AbstractFileHandler):
    def __init__(self, api_key_manager: ApiKeyManager):
        self.api_key_manager = api_key_manager
        self.creds = Credentials.from_service_account_info(
            self.api_key_manager.get_gcp_service_account()
            )
        self.drive_service = build('drive', 'v3', credentials=self.creds)

    @retry(**retry_kwargs)
    def download_file(self, file_id: str) -> io.BytesIO:
        request = self.drive_service.files().get_media(fileId=file_id)
        downloaded = io.BytesIO()
        downloader = MediaIoBaseDownload(downloaded, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        downloaded.seek(0)
        return downloaded

    def load_source_file(
        self, file_path: str, gdrive_file_id: str
    ) -> Union[str, io.BytesIO]:
        if os.path.exists(file_path):
            return file_path
        else:
            return self.download_file(gdrive_file_id)

    def load_pickle_file(self, file_path: str) -> Dict[str, str]:
        with open(file_path, "rb") as f:
            return pickle.load(f)


class AbstractScoreCalculator(ABC):
    @abstractmethod
    def create_query_vector(
        self, query_tags: List[str], tag_vector: Dict[str, str]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def calculate_score(
        self, title_vec: np.ndarray,
        abstract_vec: np.ndarray,
        query_vector: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        pass


class ScoreCalculator(AbstractScoreCalculator):
    def create_query_vector(
        self, query_tags: List[str],
        tag_vector: Dict[str, str]
    ) -> np.ndarray:
        query_vector = [tag_vector[tag] for tag in query_tags]
        return sum(np.array(query_vector)) / len(query_vector)  # type: ignore

    def calculate_score(
        self, title_vec: np.ndarray,
        abstract_vec: np.ndarray,
        query_vector: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        title_score = title_vec @ query_vector
        abstract_score = abstract_vec @ query_vector
        return alpha * title_score + (1 - alpha) * abstract_score


class AbstractSummaryGenerator(ABC):
    @abstractmethod
    def generate_summary(self, placeholder, title: str, abstract: str) -> str:
        pass


class SummaryGenerator(AbstractSummaryGenerator):
    def __init__(self, api_handler: AbstractApiHandler):
        self.api_handler = api_handler

    def generate_summary(self, placeholder, title: str, abstract: str) -> str:
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

        placeholder.markdown("ChatGPTが考え中です...😕", unsafe_allow_html=True)

        messages = [{"role": "user", "content": prompt}]
        result: list = []
        thread = threading.Thread(
            target=self.api_handler.chat_completion_request,
            args=(messages, result, "gpt-3.5-turbo-0613", functions)
        )
        thread.start()
        i = 0
        faces = ["😕", "😆", "😴", "😊", "😱", "😎", "😏"]
        while thread.is_alive():
            i += 1
            face = faces[i % len(faces)]
            placeholder.markdown(
                f"ChatGPTが考え中です...{face}", unsafe_allow_html=True
                )
            time.sleep(0.5)
        thread.join()

        if len(result) == 0:
            placeholder.markdown(
                "ChatGPTの結果取得に失敗しました...😢", unsafe_allow_html=True
                )
            return ""

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
        render_text = f"""<div style="border: 1px rgb(128, 132, 149) solid;
                            padding: 20px;">{gen_text}</div>"""
        placeholder.markdown(render_text, unsafe_allow_html=True)
        return gen_text
