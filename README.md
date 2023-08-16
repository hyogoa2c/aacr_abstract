# AACR 2023 Annual Meeting Abstract Search

## Motivation

cvpr論文検索サイトのコードを参考にAACRの論文を検索するサイトを作成してみる。

ChatGPTにPDFを読み込んでいろいろやるチャレンジの一環

## 起動方法

```bash
poetry shell
streamlit run app.py
```

## セットアップ

`./.streamlit/secrets.toml`に以下を設定する

```toml
OPENAI_API_KEY = "YOUR_API_KEY"

[gcp_service_account]
type = "service_account"
project_id = "aacr-streamlit"
private_key_id = "YOUR_PRIVATE_KEY_ID"
private_key = "-----BEGIN PRIVATE KEY-----
private_key
-----END PRIVATE KEY-----\n"
client_email = "CLIENT_EMAIL"
client_id = "CLIENT_ID"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/streamlit-access%40aacr-streamlit.iam.gserviceaccount.com"
```

## TODO

- [x] /?qで検索語のウインドウが開かないバグの修正＞ChatGPTでも確認＋Streamlitのマニュアルを確認＋<https://github.com/kotarotanahashi/cvpr/>の動きを参考に
- [ ] アブストラクトの全文確認のためのリンクをhttps://aacrjournals.org/cancerres/search-results?q={urllib.parse.quote(title)}で実行する様に変更する。
- [ ] abstract省略表示の後に、残りのワードカウントを追加表示させる。
