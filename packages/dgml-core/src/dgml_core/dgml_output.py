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

"""The DGML the pipeline produced for a file in a DocSet.

Each (DocSet, file) pair has one core DGML file,
``docsets/<did>/files/<fid>/<stem>.dgml.xml``. ``docset generate`` writes the
grounded document tree there, and ``extraction extract`` adds a
``dg:extraction`` element to it (or writes a standalone one when no tree exists
yet). DGML is always per DocSet: its tags come from the DocSet's schema, so a
file has no DGML of its own outside a DocSet.

:func:`read_dgml` returns that file's stored bytes and blob key: the full DGML,
tree and extraction alike. :func:`dgml_core.extraction.get_extracted_values`
builds on it to return just the extracted values as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import layout
from .docsets import DocSetStore
from .errors import DgmlNotFound, FileNotFound
from .files import FileStore
from .storage import Workspace


@dataclass(frozen=True)
class DgmlDocument:
    """A pair's core DGML file as stored."""

    key: str
    """Blob key, ``docsets/<did>/files/<fid>/<stem>.dgml.xml``."""
    xml: bytes
    """The stored bytes, unmodified."""


def read_dgml(workspace: Workspace, docset_id: str, file_id: str) -> DgmlDocument:
    """Return the core DGML file for ``(docset_id, file_id)``.

    Raises:
        :class:`InvalidArgument` — empty ids.
        :class:`DocSetNotFound` — unknown docset.
        :class:`FileNotFound` — unknown file, or a file not assigned to the docset.
        :class:`DgmlNotFound` — assigned, but no DGML written yet.
    """
    docsets = DocSetStore(workspace)
    docsets.get(docset_id)  # existence check only: raises DocSetNotFound
    record = FileStore(workspace).get(file_id)
    if not docsets.is_assigned(docset_id, file_id):
        raise FileNotFound(f"file '{file_id}' is not assigned to docset '{docset_id}'")

    stem = Path(record.original_filename).stem
    key = layout.dgml_xml_key(docset_id, file_id, stem)
    try:
        xml = workspace.blobs.get_blob(key)
    except FileNotFoundError:
        raise DgmlNotFound(
            f"no DGML for file '{file_id}' in docset '{docset_id}'; "
            "run 'dgml docset generate' or 'dgml extraction extract' first"
        ) from None
    return DgmlDocument(key=key, xml=xml)
