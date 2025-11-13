from typing import Annotated

from marker.builders import BaseBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.registry import get_block_class


class DocumentBuilder(BaseBuilder):
    """
    Constructs a Document given a PdfProvider, LayoutBuilder, and OcrBuilder.
    """
    lowres_image_dpi: Annotated[
        int,
        "DPI setting for low-resolution page images used for Layout and Line Detection.",
    ] = 96
    highres_image_dpi: Annotated[
        int,
        "DPI setting for high-resolution page images used for OCR.",
    ] = 192
    disable_ocr: Annotated[
        bool,
        "Disable OCR processing.",
    ] = False

    def __call__(
        self,
        provider: PdfProvider,
        layout_builder: LayoutBuilder,
        line_builder: LineBuilder,
        ocr_builder: OcrBuilder,
        progress_callback=None,
    ):
        total_pages = len(provider.page_range)
        if progress_callback:
            progress_callback(
                {
                    "stage": "document_preparation",
                    "processed_pages": 0,
                    "total_pages": total_pages,
                }
            )

        document = self.build_document(provider)

        if progress_callback:
            progress_callback(
                {
                    "stage": "layout_detection",
                    "processed_pages": 0,
                    "total_pages": total_pages,
                }
            )
        layout_builder(document, provider)

        if progress_callback:
            progress_callback(
                {
                    "stage": "line_extraction",
                    "processed_pages": 0,
                    "total_pages": total_pages,
                }
            )
        line_builder(document, provider)

        if not self.disable_ocr:
            if progress_callback:
                progress_callback(
                    {
                        "stage": "ocr_processing",
                        "processed_pages": 0,
                        "total_pages": total_pages,
                    }
                )
            ocr_builder(document, provider)

        if progress_callback:
            progress_callback(
                {
                    "stage": "document_ready",
                    "processed_pages": total_pages,
                    "total_pages": total_pages,
                }
            )

        return document

    def build_document(self, provider: PdfProvider):
        PageGroupClass: PageGroup = get_block_class(BlockTypes.Page)
        lowres_images = provider.get_images(provider.page_range, self.lowres_image_dpi)
        highres_images = provider.get_images(provider.page_range, self.highres_image_dpi)
        initial_pages = [
            PageGroupClass(
                page_id=p,
                lowres_image=lowres_images[i],
                highres_image=highres_images[i],
                polygon=provider.get_page_bbox(p),
                refs=provider.get_page_refs(p)
            ) for i, p in enumerate(provider.page_range)
        ]
        DocumentClass: Document = get_block_class(BlockTypes.Document)
        return DocumentClass(filepath=provider.filepath, pages=initial_pages)
