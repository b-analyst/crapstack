from crapstack.haystack.nodes.file_converter.base import BaseConverter

from crapstack.haystack.utils.import_utils import safe_import

from crapstack.haystack.nodes.file_converter.csv import CsvTextConverter
from crapstack.haystack.nodes.file_converter.docx import DocxToTextConverter
from crapstack.haystack.nodes.file_converter.json import JsonConverter
from crapstack.haystack.nodes.file_converter.tika import TikaConverter, TikaXHTMLParser
from crapstack.haystack.nodes.file_converter.txt import TextConverter
from crapstack.haystack.nodes.file_converter.azure import AzureConverter
from crapstack.haystack.nodes.file_converter.parsr import ParsrConverter


MarkdownConverter = safe_import(
    "crapstack.haystack.nodes.file_converter.markdown", "MarkdownConverter", "preprocessing"
)  # Has optional dependencies
ImageToTextConverter = safe_import(
    "crapstack.haystack.nodes.file_converter.image", "ImageToTextConverter", "ocr"
)  # Has optional dependencies
PDFToTextOCRConverter = safe_import(
    "crapstack.haystack.nodes.file_converter.pdf_ocr", "PDFToTextOCRConverter", "ocr"
)  # Has optional dependencies
PDFToTextConverter = safe_import(
    "crapstack.haystack.nodes.file_converter.pdf", "PDFToTextConverter", "pdf"
)  # Has optional dependencies
