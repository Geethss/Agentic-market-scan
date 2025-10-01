
# Agentic Market Scan

**Agentic Market Scan** is a multi-agent workflow for **automated market intelligence**.  
It uses **LangGraph** for orchestration and integrates **Gemini/OpenAI LLMs**, **Tavily Search**, **Chroma**, and **Streamlit** to transform unstructured web snippets into structured, evidence-backed vendor comparison reports with trust &amp; recency scoring, evidence-backed citations, and exportable artifacts (CSV, JSON, Markdown, HTML).

## Features

- **Multi-agent workflow (LangGraph)** — nodes for `decompose → scout → ingest → extract → score → write`
- **Automated web research** — Tavily API + full-text fetch for premium users
- **Evidence-backed vendor matrix** — CSV, JSON, Markdown, and styled HTML report generation
- **LLM flexibility** — supports both **Gemini** (default) and **OpenAI GPT** (for pro/premium)
- **Semantic memory** — deduplicated snippets stored in **Chroma**
- **Trust & recency scoring** — vendors ranked by evidence quantity, recency, domain diversity, and coverage
- **Production-ready** — rate limiting, Celery-compatible orchestration, Dockerized infra, and Streamlit frontend

---

## Agents Overview

The system is built from modular **agents**, each responsible for one step in the pipeline:

1. **`decompose.py`**  
   - Breaks down a user’s query into structured **fields** (e.g. pricing, features, compliance)  
   - Generates **focused subqueries** for web search  

2. **`scout.py`**  
   - Runs web searches via **Tavily API**  
   - Collects relevant snippets, URLs, and metadata  
   - Optionally fetches full text for premium-tier users  

3. **`ingest.py` + `dedupe_rank.py`**  
   - Chunks, embeds, and deduplicates snippets  
   - Uses **Chroma** for semantic memory storage  
   - Removes redundant or near-duplicate chunks to keep retrieval clean  

4. **`extract.py`**  
   - Uses **Gemini/OpenAI LLMs** to extract structured vendor data from snippets  
   - Normalizes into fields: name, pricing, features, integrations, compliance, etc.  
   - Ensures citations by linking claims back to snippet IDs  

5. **`score.py`**  
   - Assigns each vendor a **trust score** based on:  
     - Evidence count  
     - Recency of sources  
     - Domain diversity  
     - Coverage of key fields  
   - Produces normalized ranking (0–100)  

6. **`write.py`**  
   - Exports artifacts for end users:  
     - `matrix.csv` — vendor comparison  
     - `brief.md` — Markdown summary with citations  
     - `vendors.json` — structured vendor data  
     - `report.html` — styled interactive report (with optional PDF export)  

---

## Quickstart

1. **Clone & install**
   ```bash
   git clone https://github.com/yourname/agentic-market-scan.git
   cd agentic-market-scan
   pip install -r requirements.txt

2. **Set environment variables**

   Copy .env.example → .env and fill:
     ```bash
     GEMINI_API_KEY=...
     OPENAI_API_KEY=...
     TAVILY_API_KEY=...
   
2. **Run Streamlit frontend**
   ```bash
   streamlit run frontend/app.py

## Outputs

Each run produces artifacts in runs/<job_id>/:

     - `matrix.csv` — vendor comparison  
     - `brief.md` — Markdown summary with citations  
     - `vendors.json` — structured vendor data  
     - `report.html` — styled interactive report (with optional PDF export) 

Along with the entire search process undertaken for debugging.

## Tech Stack

* **LangGraph** — agent orchestration
* **Gemini / OpenAI GPT** — LLM reasoning
* **Tavily** — web search & snippets
* **Chroma** — semantic memory & embeddings
* **Streamlit** — demo frontend

## Future work

* Async Celery pipeline execution
* Job history browser in Streamlit
* PDF/PowerPoint report export
* More robust structured extraction (regex + critic loop)
* Enterprise-ready auth and quotas

## Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open an [issue](../../issues) or submit a pull request.  

1. Fork the repo  
2. Create your feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit your changes (`git commit -m 'Add some amazing feature'`)  
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

## 🧑‍💻 Author

Built with ❤️ to showcase multi-agent workflows, LangGraph, and production-ready GenAI apps for enterprise market research.

Developed by **Geethanjali**  



