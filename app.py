import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import urllib.parse
from ast import literal_eval

from src.utils import (
    ApiKeyManager,
    GoogleDriveHandler,
    OpenAiHandler,
    ScoreCalculator,
    SummaryGenerator
    )

secrets = dict(st.secrets)

api_key_manager = ApiKeyManager(secrets)
file_handler = GoogleDriveHandler(api_key_manager)
openai_handler = OpenAiHandler(api_key_manager)
abstract_score = ScoreCalculator()
abstract_summary = SummaryGenerator(openai_handler)

# File IDs
abstract_vec_id = secrets['gcp_service_account']['abstract_vec_id']
title_vec_id = secrets['gcp_service_account']['title_vec_id']
metas_id = secrets['gcp_service_account']['metas_id']

# File paths
tag_vec_pickle = "./resources/tag_vector.pickle"
metas_csv_file = "./data/vector/store/aacr_metas.csv"
title_vec_npy_file = "./data/vector/store/title_vec.npy"
abstract_vec_npy_file = "./data/vector/store/abstract_vec.npy"

# Retry parameters
retry_kwargs = {
    "stop_max_attempt_number": 5,
    "wait_exponential_multiplier": 1000,
    "wait_exponential_max": 10000,
}

embedding_model = "text-embedding-ada-002"

# Load source files
metas_csv_source = file_handler.load_source_file(metas_csv_file, metas_id)
title_vec_source = file_handler.load_source_file(title_vec_npy_file, title_vec_id)
abstract_vec_source = file_handler.load_source_file(
    abstract_vec_npy_file, abstract_vec_id
    )


def search_for_rows(
    tag_query_vector: np.ndarray, text_query_vector: np.ndarray, k: int, alpha: float
) -> pd.DataFrame:
    """Search for rows based on a tag query vector and a text query vector.
    Args:
        tag_query_vector: The tag query vector.
        text_query_vector: The text query vector.
        k: The number of rows to return.
        alpha: The weight for the title score.
    Returns:
        The top k rows as a DataFrame.
    """
    meta_df = pd.read_csv(
        metas_csv_source,
        converters={'authors': literal_eval, 'affiliations': literal_eval},
        encoding="utf-8"
        )
    title_vec = np.load(title_vec_source)
    abstract_vec = np.load(abstract_vec_source)

    if tag_query_vector is not None and text_query_vector is not None:
        query_vector = (tag_query_vector + text_query_vector) / 2.0
    elif tag_query_vector is not None:
        query_vector = tag_query_vector
    elif text_query_vector is not None:
        query_vector = text_query_vector
    else:
        raise ValueError("Both query vectors are None")

    score = abstract_score.calculate_score(title_vec, abstract_vec, query_vector, alpha)
    top_k_indices = np.argsort(-score)[:k]
    return meta_df.iloc[top_k_indices]


def display_search_results(results: pd.DataFrame):
    """Display the search results.
    Args:
        results: The search results as a DataFrame.
    """
    results.fillna("", inplace=True)

    if "summary_clicked" not in st.session_state:
        st.session_state.summary_clicked = [False] * len(results)

    if "summary" not in st.session_state:
        st.session_state.summary = [""] * len(results)

    for i, (_, row) in enumerate(results.iterrows()):
        id = row["id"]
        title = row["title"]
        abstract = row["abstract"][0:300]
        if len(row["authors"]) == 1:
            authors = row["authors"]
        elif len(row["authors"]) == 2:
            authors = row["authors"][0] + ' and ' + row["authors"][1]
        else:
            authors = row["authors"][0] + ' and ' + row["authors"][1] + ' et al.'
        if len(row["affiliations"]) == 1:
            affiliations = row["affiliations"][0]
        else:
            affiliations = row["affiliations"][0] + " and " + str(
                len(row["affiliations"]) - 1
                ) + " others"
        st.markdown(f"#### {id}: **{title}**")
        st.markdown(f"**Abstract**\: {abstract} ...")
        st.markdown(f"Author(s)\: {authors}")
        st.markdown(f"Affiliation(s)\: {affiliations}")

        link = f"[この研究と似た論文を探す](/?q={urllib.parse.quote(title)})"
        st.markdown(link, unsafe_allow_html=True)

        if st.button("この研究の何がすごいのかChatGPTに聞く", key=f"summary_{i}"):
            st.session_state.summary_clicked[i] = True

        if st.session_state.summary_clicked[i]:
            if len(st.session_state.summary[i]) == 0:
                placeholder = st.empty()
                gen_text = abstract_summary.generate_summary(
                    placeholder, row["title"], row["abstract"]
                    )
                st.session_state.summary[i] = gen_text
            else:
                print("summary exists")
                st.markdown(
                    st.session_state.summary[i], unsafe_allow_html=True
                    )

        st.markdown("---")


def main():
    st.set_page_config(page_title="LLMによるAACR演題検索システム")
    image = Image.open("banner.png")

    st.image(
        image,
        caption="April 14-19, 2023, Orange County Convention Center, Orlando, Florida",
        use_column_width=True,
    )

    st.title("AACR 2023, 文書埋め込みを用いた論文検索")
    st.caption(
        "検索キーワードをOpenAI APIを使ってベクトル化し、"
        "AACR 2023のAbstractから関連する論文を検索することができます。"
        "また、論文の内容をChatGPTに要約してもらうことができます。"
    )

    if "search_clicked" not in st.session_state:
        st.session_state.search_clicked = False

    def clear_session_state(key):
        if key in st.session_state:
            st.session_state.pop(key)

    def clear_session():
        st.session_state.search_clicked = False
        clear_session_state("summary_clicked")
        clear_session_state("summary")

    tag_vector = file_handler.load_pickle_file(tag_vec_pickle)

    query_text = st.text_input("テキストで検索", "")
    query_tags = st.multiselect(
        "タグで検索",
        list(tag_vector.keys()),
        []
    )

    target_options = ["タイトルから検索", "タイトルとアブストラクトから検索", "アブストラクトから検索"]
    target = st.radio("検索条件", target_options, on_change=clear_session)
    ratio = target_options.index(target) / 2.0  # type: ignore

    num_results = st.selectbox(
        "表示件数:", (20, 50, 100, 200), index=0, on_change=clear_session
    )

    st.write("検索するには「検索」ボタンをクリックしてください。")

    if st.button("検索"):
        st.session_state.search_clicked = True
        clear_session_state("summary_clicked")
        clear_session_state("summary")

    if st.session_state.search_clicked:
        with st.spinner("検索中..."):
            tag_query_vector = None
            if len(query_tags) > 0:
                tag_query_vector = abstract_score.create_query_vector(
                    query_tags, tag_vector
                    )

            text_query_vector = None
            if len(query_text) > 0:
                text_query_vector = openai_handler.generate_text_embedding(
                    query_text, embedding_model
                    )

            results = search_for_rows(
                tag_query_vector, text_query_vector,  # type: ignore
                k=num_results, alpha=ratio  # type: ignore
                )

        if len(results) > 0:
            st.success("検索が完了しました。以下に結果を表示します。")
            display_search_results(results)
        else:
            st.warning("該当する論文が見つかりませんでした。")

    st.button("クリア", on_click=clear_session)


if __name__ == "__main__":
    main()
