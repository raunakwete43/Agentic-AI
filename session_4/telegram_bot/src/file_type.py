from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from rich.markdown import Markdown
from rich.console import Console
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from pydantic import BaseModel, Field
from typing import Literal
from rich.pretty import pprint
import logging
from langchain_core.rate_limiters import InMemoryRateLimiter
import asyncio

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL", "gemini-2.5-flash")
BASE_URL = os.getenv("BASE_URL")

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1, check_every_n_seconds=5, max_bucket_size=5
)

llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,  # type: ignore
    base_url=BASE_URL,
    temperature=0.1,
)

console = Console()


class Document(BaseModel):
    doc_type: Literal["experiment", "assignment"]
    subject: Literal["Deep Learning", "Blockchain", "NLP", "Cybersecurity"]
    # subject: str
    number: int = Field(..., gt=0)


async def get_file_info(file_path: str) -> Document:
    logger.debug(f"Processing file: {file_path}")
    loader = PyMuPDF4LLMLoader(file_path)
    data = await loader.aload()
    logger.info(f"Loaded {len(data)} pages from the document.")
    try:
        response = await llm.with_structured_output(Document).ainvoke(
            f"""Extract the document type, subject and number based on the file content,
            
        # Content:
        {data[0].page_content}
        """
        )
        logger.debug("LLM response received.")
    except Exception as e:
        logger.error(f"Error during LLM invocation: {e}")
        raise

    return Document.model_validate(response)


async def main():
    files = os.listdir("test_dir")
    file_paths = [
        os.path.join("test_dir", file) for file in files if file.endswith(".pdf")
    ]

    for file_path in file_paths:
        doc_info = await get_file_info(file_path)
        print(f"File: {file_path}")
        pprint(doc_info)
        print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
