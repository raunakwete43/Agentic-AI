## AI Agents – Session 1

This repository contains code and examples from the first session of an online AI course. It demonstrates basic usage of language models with LangChain and OpenAI APIs.

### Project Structure

- `main.py` – Entry point (prints a hello message)
- `session_1/langchain/` – LangChain-based LLM examples
- `session_1/openai/` – OpenAI API usage examples

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
For learning and experimentation only.

---
For .env files, replace `<gemini-api-key>` with your actual Gemini API key.
