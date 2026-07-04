# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
