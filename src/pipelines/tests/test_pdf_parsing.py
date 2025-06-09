import asyncio
import os

from pipelines.doc_parsings import process_pdf
from utils import NoopLogHelper

log = NoopLogHelper()

async def test_pdf_parsing() -> None:
  pitchdeck = await process_pdf(
    pdf_title="MixPanel Mgmt Press.pdf",
    inpath=os.path.join(
      os.path.dirname(__file__), "docs/test.pdf"
    ),
    log=log,
    cancel_requested=asyncio.Event(),
  )
  
  print(pitchdeck)

if __name__ == "__main__":
  asyncio.run(test_pdf_parsing())