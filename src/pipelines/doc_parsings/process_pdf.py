# Standard library imports
import asyncio
import base64
from collections import deque
from contextlib import contextmanager
from functools import partial
from typing import Iterator, List, Literal, Optional

# Third-party imports
import aiohttp
import pymupdf  # type: ignore
from lonelypsc.util.errors import combine_multiple_exceptions
from pydantic import BaseModel, Field

from models.openai.complete_and_parse_message import (
    complete_and_parse_message,
)
from models.openai.completer import complete_message_openai

# Local imports
from pipelines.doc_parsings.prompts import doc_extraction_prompt
from pipelines.models.doc_processing.doc_processing_data_model import (
    DocOCRResult,
)
from utils.retry import retry
from utils.log_exceptions import log_exceptions
from utils.log_helper import LogHelper


class PreparedOCRPage(BaseModel):

    page_number: int = Field(description="The page number within the PDF document")
    page_png_base64: str = Field(
        description="The base64-encoded PNG image data of the page"
    )


class ParsedPDFPage(BaseModel):

    page_number: int = Field(description="The page number within the PDF document")
    content: DocOCRResult = Field(description="The content extracted from the PDF page")


class ParsedPDF(BaseModel):

    pdf_title: str = Field(
        description="The title of the PDF document, usually the file name"
    )
    pages: List[ParsedPDFPage] = Field(
        description="List of pages in the PDF document, each containing extracted content"
    )


@contextmanager
def open_pymupdf_from_bytes(
    *, stream: bytes, filetype: Literal["pdf"] = "pdf"
) -> Iterator[pymupdf.Document]:
    """Context manager for safely handling PyMuPDF documents.

    Args:
        stream: The PDF bytes
        filetype: The file type (currently only "pdf" is supported)

    Yields:
        PyMuPDF document object
    """
    pdf_doc = pymupdf.open(stream=stream, filetype=filetype)
    try:
        yield pdf_doc
    finally:
        pdf_doc.close()


@contextmanager
def open_pymupdf_from_file(
    filename: str,
    /,
    *,
    filetype: Literal["pdf"] = "pdf",
) -> Iterator[pymupdf.Document]:
    """Context manager for safely handling PyMuPDF documents.

    Args:
        filename: The path to the PDF file
        filetype: The file type (currently only "pdf" is supported)

    Yields:
        PyMuPDF document object
    """
    pdf_doc = pymupdf.open(filename, filetype=filetype)
    try:
        yield pdf_doc
    finally:
        pdf_doc.close()


def convert_pdf_page_to_image_sync(path: str, page_num: int) -> PreparedOCRPage:
    """
    Converts a single PDF page to a base64-encoded PNG image (synchronous).

    Args:
        path (str): the path to the PDF file
        page_num (int): The zero-based index of the page to convert

    Returns:
        str: Base64-encoded PNG image data
    """
    with open_pymupdf_from_file(path, filetype="pdf") as pdf_document:
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(150 / 72, 150 / 72))  # 300 DPI
        img_bytes = pix.tobytes("png")
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    return PreparedOCRPage(page_number=page_num, page_png_base64=base64_image)


async def convert_pdf_page_to_image(path: str, page_num: int) -> PreparedOCRPage:
    """
    Async wrapper to load the PDF document at the given path and return the
    given page as a base64-encoded PNG image.

    Args:
        path (str): the path to the PDF file
        page_num: The zero-based index of the page to convert

    Returns:
        str: Base64-encoded PNG image data
    """
    return await asyncio.to_thread(convert_pdf_page_to_image_sync, path, page_num)


class DocOCRFailedException(Exception):
    """Custom exception for OCR failures due to a missing or invalid response."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def doc_ocr_result_parser(text: str) -> DocOCRResult:
    """Parse the JSON text response into a DocOCRResult object.

    Args:
        text (str): JSON text response from the model

    Returns:
        DocOCRResult: Parsed result

    Raises:
        ValueError: If the JSON cannot be parsed into a DocOCRResult
    """
    open_brace = text.find("{")
    close_brace = text.rfind("}")
    if open_brace == -1 or close_brace == -1:
        raise ValueError("no JSON object found")
    output = text[open_brace : close_brace + 1]
    response = DocOCRResult.model_validate_json(output)
    return response


async def perform_pdf_page_recognition(
    log: LogHelper,
    cancel_requested: asyncio.Event,
    /,
    *,
    prepared: PreparedOCRPage,
) -> ParsedPDFPage:
    """Passes the given base64-encoded PNG image to the OpenAI API for OCR.

    Args:
        log (LogHelper): the logger to use
        cancel_requested (asyncio.Event): Event to check for cancellation
        prepared (PreparedOCRPage): The prepared OCR page containing the
            page number and base64-encoded PNG image data

    Raises:
        DocOCRFailedException: If the OpenAI API call returns an invalid response
    """
    system_prompt = f"""
                         {doc_extraction_prompt}
                         Output in JSON format that conforms to this schema:
                         {DocOCRResult.model_json_schema()}
                         """

    # Call the new completer with image and parse the result
    result = await complete_and_parse_message(
        cancel_requested=cancel_requested,
        always_try_once=True,
        log=log,
        model="gpt-4o-2024-08-06",  # Use the vision model
        system=system_prompt,
        user="Analyze this document image and extract the requested information.",
        image_base64=prepared.page_png_base64,
        completer=partial(
            complete_message_openai,
            structured_output=True,
            output_model=DocOCRResult,
        ),
        parser=doc_ocr_result_parser,
    )

    if result is None:
        raise DocOCRFailedException("OpenAI API call was cancelled")

    raw_response, parsed_content = result

    if parsed_content is None:
        raise DocOCRFailedException(
            f"Failed to parse API response into DocOCRResult: {raw_response}"
        )

    # Log usage info - Note: We don't have direct access to token usage here
    log.info(f"OCR completed for page {prepared.page_number}")

    return ParsedPDFPage(page_number=prepared.page_number, content=parsed_content)


async def retrying_perform_pdf_page_recognition(
    *,
    cancel_requested: asyncio.Event,
    always_try_once: bool,
    prepared: PreparedOCRPage,
    log: LogHelper,
) -> Optional[ParsedPDFPage]:
    """Standard retry wrapper around performing OCR on a PDF page.

    Args:
        cancel_requested (asyncio.Event): Checked regularly; if set, stops
            early and returns None
        always_try_once (bool): if False, cancel_requested is checked before
            the first attempt. if True, only on subsequent attempts
        prepared (PreparedOCRPage): The prepared OCR page containing the
            page number and base64-encoded PNG image data
        log (LogHelper): the logger to use
    """

    @log_exceptions(
        log=log,
        message=lambda: f"{prepared.page_number=} ({len(prepared.page_png_base64)=})",
    )
    async def _attempt() -> ParsedPDFPage:
        return await perform_pdf_page_recognition(
            log,
            cancel_requested,
            prepared=prepared,
        )

    return await retry(
        _attempt,
        on=(DocOCRFailedException, aiohttp.ClientError),
        cancel_requested=cancel_requested,
        always_try_once=always_try_once,
        max_attempts=5,
    )


def _get_number_of_pages_sync(inpath: str) -> int:
    with open_pymupdf_from_file(inpath, filetype="pdf") as pdf_doc:
        return len(pdf_doc)


async def process_pdf(
    *,
    pdf_title: str,
    inpath: str,
    log: LogHelper,
    cancel_requested: asyncio.Event,
    max_cpu_concurrency: int = 2,
    max_network_concurrency: int = 5,
    max_waiting_images: int = 2,
) -> Optional[ParsedPDF]:
    """Processes all pages of a PDF document and returns the results.

    Args:
        pdf_title (str): The title of the PDF document, usually the original
            file name
        inpath (str): The path to the PDF file locally
        log (LogHelper): Logger instance
        cancel_requested (asyncio.Event): Checked regularly; if set, stops
            early and returns None
        max_cpu_concurrency (int): Maximum number of CPU-bound tasks to run
            concurrently in different threads (default: 2, as instances are
            currently small)
        max_network_concurrency (int): Maximum number of network-bound tasks
            to run concurrently in different threads (default: 5)
        max_waiting_images (int): Maximum number of images that have been loaded
            and shifted to the main thread before slowing down cpu-bound tasks

    Returns:
        ParsedPDF: if OCR was successful
        None: if interrupted by `cancel_requested`

    Raises:
        Exception: if all retries are exhausted on any of the pages
    """
    log.info(f"Processing PDF file @ {inpath} using title {pdf_title}")
    if cancel_requested.is_set():
        log.info("Cancellation requested, stopping PDF processing")
        return None

    try:
        num_pages = await asyncio.to_thread(_get_number_of_pages_sync, inpath)
    except Exception as e:
        log.error("Failed to open PDF document and determine number of pages")
        log.exception(e)
        raise

    log.info(f"Number of pages in PDF: {num_pages}")
    if num_pages == 0:
        log.error("PDF document is empty")
        return ParsedPDF(pdf_title=pdf_title, pages=[])

    if cancel_requested.is_set():
        log.info("Cancellation requested, stopping PDF processing")
        return None

    next_to_start = 0
    preparing_for_ocr: deque[asyncio.Task[PreparedOCRPage]] = deque()
    ready_for_ocr: deque[PreparedOCRPage] = deque()
    working_on_ocr: deque[asyncio.Task[Optional[ParsedPDFPage]]] = deque()
    done_pages_unsorted: List[ParsedPDFPage] = []
    errors: List[BaseException] = []

    while True:
        # prepare a new page for ocr
        if (
            not cancel_requested.is_set()
            and not errors
            and len(preparing_for_ocr) + len(ready_for_ocr) < max_waiting_images
            and len(preparing_for_ocr) < max_cpu_concurrency
            and next_to_start < num_pages
        ):
            log.info(f"Starting to prepare page {next_to_start} for OCR")
            preparing_for_ocr.append(
                asyncio.create_task(convert_pdf_page_to_image(inpath, next_to_start))
            )
            next_to_start += 1
            continue

        # start ocr on a new page
        if (
            not cancel_requested.is_set()
            and not errors
            and ready_for_ocr
            and len(working_on_ocr) < max_network_concurrency
        ):
            next_ready = ready_for_ocr.popleft()
            log.info(f"Starting OCR on page {next_ready.page_number}")
            working_on_ocr.append(
                asyncio.create_task(
                    retrying_perform_pdf_page_recognition(
                        cancel_requested=cancel_requested,
                        always_try_once=False,
                        prepared=next_ready,
                        log=log,
                    )
                )
            )
            continue

        # sweep preparing for ocr
        found_done = False
        for i in range(len(preparing_for_ocr)):
            prep_item = preparing_for_ocr.popleft()
            if not prep_item.done():
                preparing_for_ocr.append(prep_item)
                continue

            found_done = True
            if (exc := prep_item.exception()) is not None:
                errors.append(exc)
                log.error(f"Error preparing page {i} for OCR:")
                log.exception(exc)
                continue

            ready_for_ocr.append(prep_item.result())
            log.info(f"Page {i} prepared for OCR")

        if found_done:
            continue

        # sweep working on ocr
        found_done = False
        for i in range(len(working_on_ocr)):
            ocr_item = working_on_ocr.popleft()
            if not ocr_item.done():
                working_on_ocr.append(ocr_item)
                continue

            found_done = True
            if (exc := ocr_item.exception()) is not None:
                errors.append(exc)
                log.error("Error performing OCR on page:")
                log.exception(exc)
                continue
            ocr_result = ocr_item.result()
            if ocr_result is None:
                log.info(
                    f"OCR page was interrupted by cancel_requested before finishing; {cancel_requested.is_set()=}"
                )
                continue
            done_pages_unsorted.append(ocr_result)
            log.info(f"Page {ocr_result.page_number} OCR completed")
        if found_done:
            continue

        # check if nothing is happening
        if not working_on_ocr and not preparing_for_ocr:
            log.info(
                f"all tasks completed; {bool(errors)=}, {cancel_requested.is_set()=}"
            )
            break

        # wait until a task completes; cancel_requested is irrelevant as we do
        # not want to cancel the tasks that are already running; cancel_requested
        # is used to stop spawning new tasks
        await asyncio.wait(
            [
                *preparing_for_ocr,
                *working_on_ocr,
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

    log.info(
        f"Finished processing PDF file {pdf_title} @ {inpath}; {bool(errors)=}, {cancel_requested.is_set()=}, {len(done_pages_unsorted)=}"
    )

    if errors:
        log.error("Errors occurred during PDF processing:")
        for error in errors:
            log.exception(error)
        raise combine_multiple_exceptions(
            "Errors occurred during PDF processing", errors
        )

    if len(done_pages_unsorted) < num_pages:
        log.warning(
            f"Only {len(done_pages_unsorted)} pages were processed out of {num_pages} total pages before cancellation"
        )
        return None

    # Sort the pages by their page number
    done_pages_sorted = sorted(done_pages_unsorted, key=lambda page: page.page_number)
    log.success(f"Successfully processed {len(done_pages_sorted)} pages in {pdf_title}")
    return ParsedPDF(
        pdf_title=pdf_title,
        pages=done_pages_sorted,
    )
