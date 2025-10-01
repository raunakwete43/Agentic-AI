import os
import re
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from pydantic import BaseModel, Field
import logging
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL = "gemini-2.5-flash"
BASE_URL = os.getenv("BASE_URL")

llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,  # type: ignore
    base_url=BASE_URL,
    temperature=0.1,
    max_retries=3,
    timeout=60,
)

Q_PROMPT = """
You are an expert academic content analyst. Extract ALL questions from the document with complete accuracy.

**RULES:**
1. Extract EVERY question including sub-questions (1.a, 1.b, etc.)
2. For marks: Use exact number mentioned.
3. Diagram: True ONLY if explicit words like "diagram", "draw", "sketch", "illustrate" appear
4. Question text: Remove question numbers and marks from the text
"""

A_PROMPT_TEMPLATE = """
You are a {subject} professor creating perfect exam answers.

**QUESTION:** {question_text}
**MARKS:** {marks}
**DIAGRAM REQUIRED:** {requires_diagram}
**REQUIRED POINT COUNT:** {point_count}

**STRICT REQUIREMENTS:**
- Answer must have EXACTLY {point_count} bullet points
- Each bullet point must start with hyphen and space ("- ")
- No markdown, no HTML, no numbered lists
- Scale depth appropriately for {marks} marks
- {diagram_requirement_text}

**THINKING PROCESS:**
1. Analyze question requirements for {marks} marks
2. Structure exactly {point_count} comprehensive points
3. Progress from basic to advanced concepts
4. Ensure academic rigor and practical examples
5. {diagram_thinking}

Now generate the answer following ALL requirements strictly.
"""


class Question(BaseModel):
    question: str = Field(..., description="The complete question text")
    marks: int = Field(..., gt=0, le=20, description="Marks allocated")
    requires_diagram: bool = Field(False, description="Whether diagram is required")
    question_number: str = Field(..., description="Original question number")


class Questions(BaseModel):
    questions: List[Question] = Field(..., description="List of all questions")


class Answer(BaseModel):
    answer: str = Field(..., description="Answer in bullet point format")
    diagram_description: str | None = Field(None, description="Diagram description")
    point_count: int = Field(..., description="Number of points in answer")


def calculate_point_count(marks: int) -> int:
    if marks <= 2:
        return 4
    elif marks <= 5:
        return marks * 2
    elif marks <= 10:
        return marks * 2
    else:
        return min(25, marks * 2)


async def load_and_process_document(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info(f"Loading document: {file_path}")
        loader = PyMuPDF4LLMLoader(file_path)
        data = await loader.aload()

        if not data:
            raise ValueError("No content extracted from document")

        doc_content = "\n".join([page.page_content for page in data])
        doc_content = re.sub(r"\n\s*\n", "\n\n", doc_content)

        logger.info(f"Document loaded successfully. Pages: {len(data)}")
        return doc_content

    except Exception as e:
        logger.error(f"Error loading document: {e}")
        raise


async def extract_questions(doc_content: str) -> Questions:
    try:
        logger.info("Extracting questions from document...")

        response = await llm.with_structured_output(Questions).ainvoke(
            [
                SystemMessage(content=Q_PROMPT),
                HumanMessage(
                    content=f"""Extract all questions from the document provided\n# Document Content:\n{doc_content}"""
                ),
            ]
        )


        questions = Questions.model_validate(response)
        logger.info(f"Successfully extracted {len(questions.questions)} questions")
        return questions

    except Exception as e:
        logger.error(f"Error extracting questions: {e}")
        raise


async def generate_answer(question: Question, subject: str = "Deep Learning") -> Answer:
    try:
        point_count = calculate_point_count(question.marks)

        diagram_requirement_text = (
            "Include a detailed diagram description after the bullet points"
            if question.requires_diagram
            else "No diagram required - focus entirely on textual explanation"
        )

        diagram_thinking = (
            "Create comprehensive diagram description with components and relationships"
            if question.requires_diagram
            else "Focus on conceptual clarity without visual aids"
        )

        prompt = A_PROMPT_TEMPLATE.format(
            subject=subject,
            question_text=question.question,
            marks=question.marks,
            requires_diagram=question.requires_diagram,
            point_count=point_count,
            diagram_requirement_text=diagram_requirement_text,
            diagram_thinking=diagram_thinking,
        )

        logger.info(
            f"Generating answer for {question.question_number} ({question.marks} marks)"
        )

        response = await llm.with_structured_output(Answer).ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content="Generate the answer following ALL format requirements exactly."
                ),
            ]
        )

        answer = Answer.model_validate(response)
        logger.info(f"Answer generated successfully: {answer.point_count} points")
        return answer

    except Exception as e:
        logger.error(f"Error generating answer for {question.question_number}: {e}")
        raise


def create_pdf_report(
    questions: List[Question],
    answers: List[Answer],
    output_path: str,
    subject: str = "Deep Learning",
):
    """Create a PDF report with questions and answers in tabular format"""

    # Create custom styles
    styles = getSampleStyleSheet()

    # Title style
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=16,
        spaceAfter=30,
        alignment=1,  # Center aligned
    )

    # Question style
    question_style = ParagraphStyle(
        "QuestionStyle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.darkblue,
        spaceAfter=6,
        leftIndent=0,
    )

    # Answer style
    answer_style = ParagraphStyle(
        "AnswerStyle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.darkgreen,
        spaceAfter=12,
        leftIndent=10,
    )

    # Diagram style
    diagram_style = ParagraphStyle(
        "DiagramStyle",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.purple,
        backColor=colors.lightgrey,
        spaceAfter=12,
        leftIndent=10,
        borderPadding=5,
    )

    # Create the PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    story = []

    # Add title
    title = Paragraph(f"{subject} - Questions and Answers", title_style)
    story.append(title)
    story.append(Spacer(1, 20))

    # Create table data
    table_data = []

    # Table headers
    headers = [
        Paragraph("<b>Question No</b>", question_style),
        Paragraph("<b>Question and Answer</b>", question_style),
    ]
    table_data.append(headers)

    # Add questions and answers to table
    for i, (question, answer) in enumerate(zip(questions, answers)):
        # Question number cell
        qn_cell = Paragraph(
            f"<b>{question.question_number}</b><br/>"
            f"Marks: {question.marks}<br/>"
            f"Diagram: {question.requires_diagram}",
            question_style,
        )

        # Question and answer cell
        content_parts = []

        # Add question
        content_parts.append(
            Paragraph(f"<b>Question:</b> {question.question}", question_style)
        )
        content_parts.append(Spacer(1, 10))

        # Add answer
        content_parts.append(Paragraph("<b>Answer:</b>", question_style))

        # Process answer bullet points
        for line in answer.answer.split("\n"):
            if line.strip().startswith("- "):
                clean_line = line.strip()[2:]  # Remove the "- " prefix
                answer_paragraphs = []
                # Split long lines into multiple paragraphs if needed
                while len(clean_line) > 100:
                    # Find a good breaking point
                    break_point = clean_line[:100].rfind(" ")
                    if break_point == -1:
                        break_point = 100
                    answer_paragraphs.append(
                        Paragraph(f"• {clean_line[:break_point]}", answer_style)
                    )
                    clean_line = clean_line[break_point:].strip()
                answer_paragraphs.append(Paragraph(f"• {clean_line}", answer_style))

                for para in answer_paragraphs:
                    content_parts.append(para)

        # Add diagram description if present
        if answer.diagram_description and question.requires_diagram:
            content_parts.append(Spacer(1, 10))
            content_parts.append(
                Paragraph("<b>Diagram Description:</b>", question_style)
            )
            diagram_lines = answer.diagram_description.split("\n")
            for line in diagram_lines:
                if line.strip():
                    content_parts.append(Paragraph(line.strip(), diagram_style))

        # Create a simple flowable for the content
        from reportlab.platypus.flowables import KeepInFrame

        content_frame = KeepInFrame(400, 600, content_parts)

        # Add row to table
        table_data.append([qn_cell, content_frame])

    # Create table
    table = Table(table_data, colWidths=[1.5 * inch, 5 * inch])

    # Style the table
    table_style = TableStyle(
        [
            # Header
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            # Table body
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            # Alternate row colors
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]
    )

    table.setStyle(table_style)
    story.append(table)

    # Build PDF
    doc.build(story)
    logger.info(f"PDF report saved to: {output_path}")


async def generate_assignment_answers(
    file_path: str, subject: str, output_dir: str
) -> str:
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        doc_content = await load_and_process_document(file_path)
        questions_data = await extract_questions(doc_content)
        answers = []

        logger.info(
            f"Generating answers for {len(questions_data.questions)} questions..."
        )

        for idx, question in enumerate(questions_data.questions, start=1):
            logger.info(
                f"Processing question {idx}/{len(questions_data.questions)}: {question.question_number}"
            )
            answer = await generate_answer(question, subject)
            answers.append(answer)

        # Create PDF output file
        original_filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(original_filename)[0]
        output_pdf_path = os.path.join(output_dir, f"{name_without_ext}_answers.pdf")

        # Generate PDF report
        create_pdf_report(questions_data.questions, answers, output_pdf_path, subject)

        return output_pdf_path  # Return the path to the generated PDF file

    except Exception as e:
        logger.error(f"Error generating answers: {e}")
        raise
