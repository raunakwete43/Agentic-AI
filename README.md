## AI Agents 

This repository contains code and examples from the AI Agents course for each session. 

### Project Structure

- `main.py` – Entry point (prints a hello message)
- `session_{i}/langchain/` – LangChain-based LLM examples
- `session_{i}/openai/` – OpenAI API usage examples

### Requirements

- Python 3.12+
- Setup Environment:
	```fish
	uv sync
	```
	or use `pyproject.toml` with your preferred tool.

### Running Examples

Navigate to a session folder and run scripts, e.g.:

```fish
uv run session_1/openai/structured_outputs.py
```

---
For .env files,
1. Rename `.env.example` file to `.env`
```bash
mv .env.example
```
2. Replace `<gemini_api_key>` with your actual GEMINI API KEY.


---
For learning and experimentation only.

