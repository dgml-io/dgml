"""Ready-made document-to-PDF converters for DGML.

These are referenced from a workspace ``conversion`` config by dotted path,
exactly like a user's own converter — there is no privileged "built-in" path::

    {
      "conversion": {
        "docx": {"provider": "translators_pdf.libreoffice:LibreOfficeConverter"},
        "xlsx": {"provider": "translators_pdf.xlsx:XlsxIslandsConverter"}
      }
    }

Each converter subclasses :class:`dgml_core.conversion.DocConverter`. SDK/binary
imports are lazy (in ``__init__``), so importing this package never requires
LibreOffice, Aspose, or the ``xlsx`` extra to be present.
"""

from __future__ import annotations

from .aspose import AsposeCellsConverter, AsposeWordsConverter
from .command import CommandConverter
from .libreoffice import LibreOfficeConverter
from .xlsx import XlsxIslandsConverter

__all__ = [
    "AsposeCellsConverter",
    "AsposeWordsConverter",
    "CommandConverter",
    "LibreOfficeConverter",
    "XlsxIslandsConverter",
]
