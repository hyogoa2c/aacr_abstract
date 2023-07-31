import streamlit as st
import pandas as pd
import numpy as np
import openai
from PIL import Image
import urllib
from ast import literal_eval

from src.utils import (
    generate_text_embedding,
    load_source_file,
    load_tag_vector,
    calculate_score,
    create_query_vector,
    generate_summary
    )


# File IDs
abstract_vec_id = '1-GuSdkDGI2u8JAXibKsU4G1KCPWMK7rV'
title_vec_id = '1-70bNFFhVrmJKp86i0BzehoeEb8dplHP'
metas_id = '1-8aTZHij2eu7xF-Dil4PNLcCftX_peqD'

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

# Load source files
metas_csv_source = load_source_file(metas_csv_file, metas_id)
title_vec_source = load_source_file(title_vec_npy_file, title_vec_id)
abstract_vec_source = load_source_file(abstract_vec_npy_file, abstract_vec_id)


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

    score = calculate_score(title_vec, abstract_vec, query_vector, alpha)
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
        abstract = row["abstract"]
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
        st.markdown(f"{abstract}")
        st.markdown(f"Author(s)\: {authors}")
        st.markdown(f"Affiliation(s)\: {affiliations}")

        link = f"[この研究と似た論文を探す](/?q={urllib.parse.quote(title)})"
        st.markdown(link, unsafe_allow_html=True)

        if st.button("この研究の何がすごいのかChatGPTに聞く", key=f"summary_{i}"):
            st.session_state.summary_clicked[i] = True

        if st.session_state.summary_clicked[i]:
            if len(st.session_state.summary[i]) == 0:
                st.session_state.summary[i] = generate_summary(
                    row["title"], row["abstract"]
                    )
            st.markdown(st.session_state.summary[i], unsafe_allow_html=True)

        st.markdown("---")


# Refactored Code Continued

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

    openai.api_key = st.secrets["OPENAI_API_KEY"]

    if "search_clicked" not in st.session_state:
        st.session_state.search_clicked = False

    def clear_session():
        st.session_state.search_clicked = False
        if "summary_clicked" in st.session_state:
            st.session_state.pop("summary_clicked")
        if "summary" in st.session_state:
            st.session_state.pop("summary")

    tag_vector = load_tag_vector(tag_vec_pickle)

    query_text = st.text_input("テキストで検索", "")
    query_tags = st.multiselect(
        "タグで検索",
        list(tag_vector.keys()),
        []
    )
    alpha = st.slider(
        "タイトルとアブストラクトの重み（0: アブストラクトのみ, 1: タイトルのみ）",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
    )
    k = st.number_input(
        "表示する論文数",
        min_value=1,
        max_value=20,
        value=5,
    )
    st.write("検索するには「検索」ボタンをクリックしてください。")

    if st.button("検索"):
        st.session_state.search_clicked = True
        if "summary_clicked" in st.session_state:
            st.session_state.pop("summary_clicked")
        if "summary" in st.session_state:
            st.session_state.pop("summary")

    if st.session_state.search_clicked:
        with st.spinner("検索中..."):
            tag_query_vector = None
            if len(query_tags) > 0:
                tag_query_vector = create_query_vector(query_tags, tag_vector)

            text_query_vector = None
            if len(query_text) > 0:
                text_query_vector = generate_text_embedding(query_text)

            results = search_for_rows(tag_query_vector, text_query_vector, k, alpha)

        if len(results) > 0:
            st.success("検索が完了しました。以下に結果を表示します。")
            display_search_results(results)
        else:
            st.warning("該当する論文が見つかりませんでした。")

    st.button("クリア", on_click=clear_session)


if __name__ == "__main__":
    main()
