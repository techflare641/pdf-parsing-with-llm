import asyncio
import os
from functools import partial
from typing import Optional

from models.openai import complete_and_parse_message
from pipelines.doc_parsings import process_pdf, ParsedPDF
from utils import NoopLogHelper
from models.openai import complete_message_openai

from pydantic import BaseModel, Field

class SummaryResult(BaseModel):
  summary: str = Field(..., description="Summary of the company")

log = NoopLogHelper()

async def test_pdf_parsing() -> Optional[ParsedPDF]:
  pitchdeck = await process_pdf(
    pdf_title="MixPanel Mgmt Press.pdf",
    inpath=os.path.join(
      os.path.dirname(__file__), "docs/test.pdf"
    ),
    log=log,
    cancel_requested=asyncio.Event(),
  )
  return pitchdeck

async def test_generate_summary(pdf: ParsedPDF) -> Optional[dict]:
  text = ""
  for page in pdf.pages:
    text += page.content.extracted_text

  system_prompt = """Summarize the key points from a company pitch deck, focusing on its industry and economic sector. Identify core activities, unique selling propositions, and the company's positioning. Present the summary in a cohesive paragraph (150-250 words), ensuring it is informative yet concise while highlighting the importance of the industry and sector. Make sure to that you adhere to the output format.
    # Examples

    **Example Input:**

    "A company specializing in sustainable energy solutions tackles climate change with innovative solar panel technology. Their mission is to provide affordable, renewable energy to underserved regions, concentrated in the renewable energy industry and operating primarily within the economic sector of green technology."

    **Example Output:**

    {
        "summary": "This company stands out in the renewable energy industry by addressing climate change through advanced solar technology. As a key player in the green technology economic sector, it focuses on delivering affordable renewable energy to underserved regions, underscoring its mission with strategic insights into the sustainable energy market."
    }

    # Notes

    - Focus specifically on summarizing aspects related to the industry and economic sector.
    - Ensure the tone remains objective, emphasizing the company's positioning within its sector."""

  summary_result = await complete_and_parse_message(
    cancel_requested=asyncio.Event(),
    always_try_once=False,
    log=log,
    model="gpt-4o-2024-08-06",
    system=system_prompt,
    user=text,
    completer=partial(
        complete_message_openai,
        structured_output=True,
        output_model=SummaryResult,
    ),
    parser=SummaryResult.model_validate_json,
    max_parser_retries=1,
  )
  if summary_result is None:
    log.error("No summary result")
    return None

  _, parsed_summary = summary_result
  print(f"Summary: {parsed_summary.summary}")
  return parsed_summary

async def test_pdf_parsing_with_summary() -> Optional[dict]:
  pitchdeck = await test_pdf_parsing()
  if pitchdeck:
    summary = await test_generate_summary(pitchdeck)
    return summary
  else:
    log.error("Failed to parse PDF")
    return None

if __name__ == "__main__":
  asyncio.run(test_pdf_parsing_with_summary())