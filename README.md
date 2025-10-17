# MunHelper

Context-based Wikipedia lookup. <br>
Searches selected Wikipedia pages and answers queries using a Llama-Index vector store and Ollama for generation.

## Features

- OpenAPI-compatible API using FastAPI
- Website using Streamlit
- Semantic search over selected Wikipedia pages
- Ingestion script to build a Llama-Index vector store
- Integration with Ollama for generation
- Docker Containerization
- Auto Update (coming soon)

## Quickstart (bare metal)

- Ensure Python 3.10+ is installed and you have Ollama running locally (see Ollama docs).
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Configure `.env` as needed (a sample is provided).

- Ingest the Wikipedia pages and build the index:

```bash
python scripts/ingest_wikipedia.py
```

- Start the API:

```bash
python app.py
```

API is served at `http://localhost:5000/docs`.

## Usage

1. Populate the index by running the ingestion script (see above). This will create files under `data/wikipedia_pages/` and persist a Llama-Index under `data/index/`.
2. Use the API endpoints:
   - `GET /search/wiki?query=...` — run a semantic search and return raw retriever output.
   - `GET /chat/wiki?query=...` — run retrieval and stream Ollama-generated answer using the retrieved context.
   - `GET /manage/reindex-wikipedia` — re-run ingestion and rebuild the index (proof-of-concept; blocking).

3. If you want different Wikipedia pages, edit `data/wikipedia_pages.txt` and call `/manage/reindex-wikipedia`.

## Other Installation Methods

> [!NOTE]
> These methods may leave residue if you decide to uninstall.
> I recommend using the Docker method for a cleaner installation.

<details>

<summary>
Bare Metal installation instructions
</summary>

## Installation - Bare Metal

- Clone the repository
  - ```shell
    git clone https://github.com/Mahasvan/Munhelper
    ```
- Install the dependencies
  - ```shell
    pip install -r requirements.txt
    ```
- Ingest Wikipedia pages and build the Llama-Index
  - ```shell
  python scripts/ingest_wikipedia.py
  ```

- Install Ollama and pull preferred model (if using Ollama images locally)
  - ```shell
  ollama pull llama3.2:1b
  ```
- Set up environment variables accordingly (refer `app.py`)
- Start the API
  - ```shell
    python app.py
    ```
- Access the API at `http://localhost:5000/docs` (or whatever port you configured)
- Setting up the frontend
  - Open another terminal window, and `cd` into the `frontend` folder
  - Follow the instructions given [here](https://github.com/Mahasvan/MunHelper-frontend/).
- Make sure to read the [Usage](#usage) section.

</details>

<details>
<summary>Docker build images from scratch</summary>

## Run with Docker (build images from scratch)

- Follow all steps in the [Docker Instructions](#installation-with-docker) until the last step.
- Start the containers using `docker-compose-build` instead of `docker-compose`
  - ```shell
     docker-compose -f docker-compose-build.yml build --no-cache
     docker-compose -f docker-compose-build.yml up -d
    ```
- Make sure to read the [Usage](#usage) section.
</details>