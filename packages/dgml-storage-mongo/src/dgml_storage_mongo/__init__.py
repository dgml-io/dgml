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

"""Sample DGML storage backends on MongoDB.

- :class:`~dgml_storage_mongo.store.MongoDocStore` — documents, one collection
  per :class:`~dgml_core.layout.Collection` member.
- :class:`~dgml_storage_mongo.gridfs_store.MongoGridFSBlobStore` — blobs in a
  GridFS bucket.
- :class:`~dgml_storage_mongo.combined.MongoGridFSStore` — both roles on one
  database.
"""

from ._client import MONGO_URI_ENV
from .combined import MongoGridFSStore
from .gridfs_store import CHUNK_BYTES, DEFAULT_BUCKET, MongoGridFSBlobStore
from .store import MongoDocStore

__all__ = [
    "CHUNK_BYTES",
    "DEFAULT_BUCKET",
    "MONGO_URI_ENV",
    "MongoDocStore",
    "MongoGridFSBlobStore",
    "MongoGridFSStore",
]
