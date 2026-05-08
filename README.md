# ✍️ Blog Writer Agent using LangGraph

An intelligent, multi-agent AI application that automatically researches, plans, and writes high-quality technical blog posts. Built with **LangGraph**, **LangChain**, and **Streamlit**, this tool simulates a team of specialized AI agents working together to produce comprehensive and grounded markdown articles.

---

## 🌟 Key Features

- **Multi-Agent Architecture**: Leverages LangGraph to coordinate specialized agents (Router, Researcher, Orchestrator, Writer, Reducer).
- **Dynamic Web Research**: Automatically determines if web research is needed using [Tavily Search](https://tavily.com/) to fetch up-to-date information and cite sources.
- **Parallel Processing**: Uses LangGraph's fan-out capabilities to write multiple blog sections concurrently, drastically reducing generation time.
- **Intelligent Planning**: Creates highly actionable, developer-focused outlines with specific goals, target word counts, and required code snippets.
- **Streamlit UI**: A clean, intuitive web interface to input topics and view the generated blog posts in real-time.

## 🏗️ System Architecture

The agent workflow follows a directed state graph:

1. **Router**: Analyzes the topic and decides if the agent needs external research (open-book vs. closed-book).
2. **Researcher**: If research is required, queries Tavily, extracts evidence, and deduplicates sources.
3. **Orchestrator**: Acts as a senior technical writer. Creates a detailed `Plan` with 5–9 structured `Tasks` (sections).
4. **Worker (Fan-out)**: Multiple worker agents execute the tasks in parallel. They follow strict grounding rules, cite sources, and write Markdown content.
5. **Reducer**: Merges all generated sections into a cohesive, final Markdown document and saves it locally.

## 🛠️ Tech Stack

- **Framework:** [LangGraph](https://python.langchain.com/docs/langgraph/) & [LangChain](https://python.langchain.com/)
- **LLM:** OpenAI (`gpt-4.1-mini`)
- **Search API:** Tavily
- **Frontend:** Streamlit
- **Data Validation:** Pydantic

## 🚀 Getting Started

### Prerequisites

You will need API keys for:
- [OpenAI](https://platform.openai.com/)
- [Tavily Search](https://tavily.com/)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SyedNadimMehdi30/blog_post_generator_langraph.git
   cd blog_post_generator_langraph
   ```

2. **Set up a Conda environment**:
   ```bash
   conda create -n llmdemo python=3.12 -y 
   conda activate llmdemo
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## 💻 Usage

Start the Streamlit application:

```bash
streamlit run app.py
```

1. Open your browser to the URL provided in the terminal (usually `http://localhost:8501`).
2. Enter your desired blog topic (e.g., *"State of Multimodal LLMs in 2026"*).
3. Click **Generate Blog**.
4. The system will plan, research, and write the blog post, displaying the final Markdown in the UI and saving it as a `.md` file in your working directory.

## 📂 Repository Structure

- `app.py`: Main LangGraph agent definition and Streamlit UI code.
- `requirements.txt`: Required Python dependencies.
- `1.Blog_agent_easy.ipynb`: Introductory notebook for a simple blog agent.
- `2.Blog_agent_prompted.ipynb`: Advanced prompt engineering demonstration.
- `3.Blog_agent_research_tool.ipynb`: Integrating research tools.
- `langGraph_easy_demo.ipynb`: A minimal LangGraph demo.