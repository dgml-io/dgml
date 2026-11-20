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

"""Custom extractors (:mod:`dgml_core.extractor`): config loading, output
validation, the GROUNDED run through dgml's tail, fallback and its logging,
telemetry, batch mode and job replay."""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import threading
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from dgml_core import Provider, layout, llm
from dgml_core.batch import FakeBackend, active_session, provider_of, register_backend
from dgml_core.batch import registry as batch_registry
from dgml_core.batch.jobs import (
    BatchJobStore,
    credential_ref,
    resolve_credential,
    start_session,
)
from dgml_core.configuration import Configuration, Identity, Model, Models, ProviderSpec, Storage
from dgml_core.configuration import Extractor as ExtractorSection
from dgml_core.docsets import DocSetStore
from dgml_core.errors import (
    AuthError,
    BatchPending,
    BatchUnavailable,
    CorruptMetadata,
    ExtractorConfigInvalid,
    ExtractorFailed,
    ExtractorOutputInvalid,
    InvalidArgument,
    ValuesExtractionFailed,
)
from dgml_core.extraction import auto_extract, extract_file, extraction_config
from dgml_core.extraction_schema import parse_rnc
from dgml_core.extraction_xml import dgml_xml_to_values
from dgml_core.extractor import (
    ExtractionRequest,
    ExtractorConfig,
    ExtractorOutput,
    ValueExtractor,
    load_extractor_config,
    make_extractor,
    output_from_json,
    output_to_json,
    validate_extractor_output,
    vocabulary_problems,
)
from dgml_core.grounded import (
    GroundedConfig,
    _pdf_bytes,
    extract_values,
    extract_values_batch_pairs,
    extract_values_many,
    load_grounded_config,
    no_batch_block,
)
from dgml_core.models_config import ConfigSection
from dgml_core.storage import Workspace
from dgml_core.usage import read_events

from .conftest import write_config
from .test_grounded import (
    DEFAULT_SCHEMA_MODEL,
    DEFAULT_VALUES_MODEL,
    _png_header,
    _seed_docset_with_schema,
    _seed_file,
    _seed_page_text,
    _tool_call_response,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

#: The seeded page prints "Hello" [100,210,182,242] and "world" [190,210,290,242].
HELLO_WORLD_BOX = [100, 210, 290, 242]


def good_output(**extra: Any) -> ExtractorOutput:
    """A tree for the `title` schema whose one box reads as its text."""
    return ExtractorOutput(
        values={
            "title": {
                "text": "Hello world",
                "locations": [{"page_number": 1, "bounding_box": list(HELLO_WORLD_BOX)}],
            }
        },
        **extra,
    )


class FakeExtractor(ValueExtractor):
    """Answers from a class-level table (by file id); records every call."""

    name: ClassVar[str] = "fake"
    config_fields: ClassVar[frozenset[str]] = frozenset({"mode"})

    outputs: ClassVar[dict[str, ExtractorOutput | BaseException]] = {}
    calls: ClassVar[list[str]] = []
    many_calls: ClassVar[list[list[str]]] = []
    constructed: ClassVar[int] = 0
    requests: ClassVar[list[ExtractionRequest]] = []

    @classmethod
    def parse_config(cls, config: ExtractorConfig) -> ExtractorConfig:
        cls._check_no_extra_fields(config.options)
        mode = config.options.get("mode", "ok")
        if mode not in ("ok", "bad"):
            raise ExtractorConfigInvalid(f"fake extractor: 'mode' must be ok or bad (got {mode!r})")
        return config

    def __init__(self, config: ExtractorConfig) -> None:
        type(self).constructed += 1
        self.config = config

    def extract(self, request: ExtractionRequest) -> ExtractorOutput:
        type(self).calls.append(request.file_id)
        type(self).requests.append(request)
        out = type(self).outputs.get(request.file_id, good_output())
        if isinstance(out, BaseException):
            raise out
        return out

    def extract_many(
        self, requests: Sequence[ExtractionRequest], *, max_concurrency: int
    ) -> list[ExtractorOutput | Exception]:
        type(self).many_calls.append([r.file_id for r in requests])
        return super().extract_many(requests, max_concurrency=max_concurrency)


class SelfVerifyingExtractor(FakeExtractor):
    """An extractor that checks its own boxes and returns normalized text."""

    name: ClassVar[str] = "selfverify"
    verifies_boxes: ClassVar[bool] = True


class NotAnExtractor:
    pass


class HalfExtractor(ValueExtractor):
    """Misses ``extract``: abstract, so a config error."""

    name: ClassVar[str] = "half"

    @classmethod
    def parse_config(cls, config: ExtractorConfig) -> ExtractorConfig:
        return config

    def __init__(self, config: ExtractorConfig) -> None:
        pass


class ForgetfulExtractor(FakeExtractor):
    name: ClassVar[str] = "forgetful"

    @classmethod
    def parse_config(cls, config: ExtractorConfig) -> ExtractorConfig:
        return None  # type: ignore[return-value]  # validates, forgets to return


class CrashingInit(FakeExtractor):
    name: ClassVar[str] = "crashing"

    def __init__(self, config: ExtractorConfig) -> None:
        raise RuntimeError("no SDK")


class KeyedExtractor(FakeExtractor):
    """Asks for the Anthropic key; records the config each hook saw."""

    name: ClassVar[str] = "keyed"
    api_key_providers: ClassVar[frozenset[Provider]] = frozenset({Provider.ANTHROPIC})

    parsed: ClassVar[list[ExtractorConfig]] = []
    configs: ClassVar[list[ExtractorConfig]] = []

    @classmethod
    def parse_config(cls, config: ExtractorConfig) -> ExtractorConfig:
        cls.parsed.append(config)
        return super().parse_config(config)

    def __init__(self, config: ExtractorConfig) -> None:
        super().__init__(config)
        type(self).configs.append(config)


def install(monkeypatch: pytest.MonkeyPatch, cls: type) -> str:
    """Publish *cls* on this module so the resolver imports it by dotted path."""
    monkeypatch.setattr(sys.modules[__name__], cls.__name__, cls, raising=False)
    return f"{__name__}:{cls.__name__}"


@pytest.fixture(autouse=True)
def _reset_fakes() -> Iterator[None]:
    for cls in (
        FakeExtractor,
        SelfVerifyingExtractor,
        ForgetfulExtractor,
        CrashingInit,
        KeyedExtractor,
    ):
        cls.outputs = {}
        cls.calls = []
        cls.many_calls = []
        cls.constructed = 0
        cls.requests = []
    KeyedExtractor.parsed = []
    KeyedExtractor.configs = []
    yield


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> str:
    return install(monkeypatch, FakeExtractor)


def _config(provider: str, **extractor: Any) -> GroundedConfig:
    return GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL,
        values_model=DEFAULT_VALUES_MODEL,
        extractor=ExtractorConfig(provider=provider, **extractor),
    )


def _seed(workspace: Workspace, fid: str = "f1aaaaaaaaaa") -> str:
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)
    workspace.blobs.put_blob(layout.file_page_image_key(fid, 1), _png_header(1000, 1000))
    ds_id, _ = _seed_docset_with_schema(workspace, fid)
    return ds_id


def _phase1(values: dict[str, Any]) -> Any:
    return _tool_call_response("submit_values", {"values": values})


_PHASE1_VALUES = {"title": {"text": "Hello world", "locations": [{"page_number": 1}]}}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_no_section_means_no_extractor(workspace: Workspace) -> None:
    write_config(workspace, {"grounded": {"values_model": "m", "schema_model": "s"}})
    assert load_extractor_config(workspace) is None
    assert load_grounded_config(workspace).extractor is None
    write_config(
        workspace, {"grounded": {"values_model": "m", "schema_model": "s"}, "extractor": {}}
    )
    assert load_extractor_config(workspace) is None


def test_load_config_resolves_and_validates_eagerly(workspace: Workspace, fake: str) -> None:
    write_config(
        workspace,
        {
            "grounded": {"values_model": "m", "schema_model": "s"},
            "extractor": {"provider": fake, "fallback": False, "max_concurrency": 2, "mode": "ok"},
        },
    )
    config = load_extractor_config(workspace)
    assert config == ExtractorConfig(
        provider=fake, fallback=False, max_concurrency=2, options={"mode": "ok"}
    )
    assert load_grounded_config(workspace).extractor == config
    assert "mode" in repr(config)


@pytest.mark.parametrize(
    ("section", "match"),
    [
        ({"provider": ""}, "extractor.provider"),
        ({"provider": 3}, "extractor.provider"),
        ({"provider": "nodots"}, "dotted path"),
        ({"provider": "no.such.module:Thing"}, "could not import"),
        ({"provider": f"{__name__}:NoSuchClass"}, "has no attribute"),
        ({"provider": f"{__name__}:NotAnExtractor"}, "not a ValueExtractor subclass"),
        ({"provider": f"{__name__}:HalfExtractor"}, "abstract class"),
        ({"provider": f"{__name__}:ForgetfulExtractor"}, "must return an ExtractorConfig"),
        ({"provider": f"{__name__}:FakeExtractor", "typo": 1}, "unknown fields"),
        ({"provider": f"{__name__}:FakeExtractor", "mode": "weird"}, "'mode' must be"),
        ({"provider": f"{__name__}:FakeExtractor", "fallback": "yes"}, "extractor.fallback"),
        ({"provider": f"{__name__}:FakeExtractor", "max_concurrency": 0}, "max_concurrency"),
        ({"provider": f"{__name__}:FakeExtractor", "max_concurrency": True}, "max_concurrency"),
    ],
)
def test_load_config_rejects(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, section: dict[str, Any], match: str
) -> None:
    for cls in (NotAnExtractor, HalfExtractor, ForgetfulExtractor, FakeExtractor):
        install(monkeypatch, cls)
    write_config(
        workspace, {"grounded": {"values_model": "m", "schema_model": "s"}, "extractor": section}
    )
    with pytest.raises(ExtractorConfigInvalid, match=match):
        load_extractor_config(workspace)
    # A config error reaches every extraction caller through the grounded loader.
    with pytest.raises(ExtractorConfigInvalid):
        load_grounded_config(workspace)


def test_callers_that_never_extract_skip_a_broken_section(workspace: Workspace) -> None:
    """``extractor=False`` (schema generation, a batch job's key lookup) does
    not read ``[extractor]``, so a broken one cannot fail it."""
    write_config(
        workspace,
        {
            "grounded": {"values_model": "m", "schema_model": "s", "schema_api_key": "sk-schema"},
            "extractor": {"provider": "no.such.module:Nope"},
        },
    )
    with pytest.raises(ExtractorConfigInvalid):
        load_grounded_config(workspace)
    assert load_grounded_config(workspace, extractor=False).extractor is None
    ref = credential_ref("grounded", "schema", None)
    assert resolve_credential(workspace, ref) == "sk-schema"


def test_section_must_be_a_table(workspace: Workspace) -> None:
    workspace.config_path.write_text('extractor = "x"\n', encoding="utf-8")
    with pytest.raises((ExtractorConfigInvalid, CorruptMetadata)):
        load_extractor_config(workspace)


def test_make_extractor_constructs_once_per_call(fake: str) -> None:
    ext = make_extractor(ExtractorConfig(provider=fake))
    assert isinstance(ext, FakeExtractor) and FakeExtractor.constructed == 1


def test_configuration_build_round_trip(fake: str, tmp_path: Path) -> None:
    cfg = Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="n", organization="o"),
        storage=Storage.combined(ProviderSpec("dgml_core.storage_local:LocalStore")),
        models=Models(advanced=Model("adv"), expert=Model("exp")),
        extractor=ExtractorSection(provider=fake, fallback=False, options={"mode": "ok"}),
    )
    assert cfg.sections[ConfigSection.EXTRACTOR] == {
        "provider": fake,
        "fallback": False,
        "mode": "ok",
    }
    ws = Workspace.open(configuration=cfg, root=tmp_path)
    assert load_extractor_config(ws) == ExtractorConfig(
        provider=fake, fallback=False, options={"mode": "ok"}
    )
    assert load_grounded_config(ws).extractor is not None


def test_library_caller_configures_the_extractor_in_memory(fake: str, tmp_path: Path) -> None:
    """A workspace opened from a Configuration (no config.toml) runs the
    extractor through the public entry point, with the section's fallback
    setting, and the per-call override still applies."""
    cfg = Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="n", organization="o"),
        storage=Storage.combined(ProviderSpec("dgml_core.storage_local:LocalStore")),
        models=Models(advanced=Model(DEFAULT_VALUES_MODEL), expert=Model(DEFAULT_SCHEMA_MODEL)),
        extractor=ExtractorSection(provider=fake, fallback=False),
    )
    ws = Workspace.open(configuration=cfg, root=tmp_path)
    assert ws.config_path.exists() is False
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(ws, fid)

    with patch("litellm.completion", side_effect=AssertionError("no model call expected")):
        result = extract_file(ws, ds_id, fid)
    assert result.extractor == fake and FakeExtractor.calls == [fid]

    # fallback = false from the section: the extractor's error is raised ...
    FakeExtractor.outputs[fid] = RuntimeError("boom")
    with pytest.raises(ExtractorFailed, match="boom"):
        extract_file(ws, ds_id, fid)
    # ... and the per-call override turns the fallback back on.
    with patch("litellm.completion", return_value=_phase1(_PHASE1_VALUES)):
        fell_back = extract_file(ws, ds_id, fid, extractor_fallback=True)
    assert fell_back.extractor is None


def test_auto_extract_block_says_what_ran(fake: str, tmp_path: Path) -> None:
    """The auto-extract block's ``model`` is what produced the values or ran
    before the failure, and ``extractor`` (present only with one configured)
    the provider whose values were written."""

    def open_with(extractor: ExtractorSection | None) -> Workspace:
        cfg = Configuration.build(
            identity=Identity(workspace_id="tenant-1", name="n", organization="o"),
            storage=Storage.combined(ProviderSpec("dgml_core.storage_local:LocalStore")),
            models=Models(advanced=Model(DEFAULT_VALUES_MODEL), expert=Model(DEFAULT_SCHEMA_MODEL)),
            extractor=extractor,
        )
        return Workspace.open(configuration=cfg, root=tmp_path)

    ws = open_with(ExtractorSection(provider=fake, fallback=False))
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(ws, fid)

    with patch("litellm.completion", side_effect=AssertionError("no model call expected")):
        block = auto_extract(ws, ds_id, fid)
    # The fake reports no model of its own, so its provider path stands in.
    assert block["error"] is None and block["model"] == fake and block["extractor"] == fake

    # Extractor failed, no fallback: nothing ran.
    FakeExtractor.outputs[fid] = RuntimeError("boom")
    with patch("litellm.completion", side_effect=AssertionError("no model call expected")):
        block = auto_extract(ws, ds_id, fid)
    assert block["error"].startswith("EXTRACTOR_FAILED")
    assert block["model"] is None and block["extractor"] is None

    # Extractor failed, fallback ran: the values model, whatever its outcome.
    ws = open_with(ExtractorSection(provider=fake))
    bogus = {"Bogus": {"text": "x", "locations": [{"page_number": 1}]}}
    with patch("litellm.completion", return_value=_phase1(bogus)):
        block = auto_extract(ws, ds_id, fid)
    assert block["error"] is not None
    assert block["model"] == DEFAULT_VALUES_MODEL and block["extractor"] is None
    with patch("litellm.completion", return_value=_phase1(_PHASE1_VALUES)):
        block = auto_extract(ws, ds_id, fid)
    assert block["error"] is None
    assert block["model"] == DEFAULT_VALUES_MODEL and block["extractor"] is None

    # No extractor configured: no key, and a failure still names the values model.
    ws = open_with(None)
    with patch("litellm.completion", return_value=_phase1(bogus)):
        block = auto_extract(ws, ds_id, fid)
    assert block["error"] is not None and block["model"] == DEFAULT_VALUES_MODEL
    assert "extractor" not in block


def test_logger_is_under_dgml_core(fake: str) -> None:
    assert FakeExtractor.log().name == "dgml_core.extractor.fake"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_INVOICE_RNC = """\
namespace docset = "http://www.dgml.io/acme/invoices#"

Invoice =
  element docset:Invoice {
    (text | VendorName | LineItems | Kind)*
  }

VendorName =
  element docset:VendorName {
    text
  }

LineItems =
  element docset:LineItems {
    LineItem*
  }

LineItem =
  element docset:LineItem {
    (text | Quantity | UnitPrice)*
  }

Quantity =
  element docset:Quantity {
    xsd:integer
  }

UnitPrice =
  element docset:UnitPrice {
    xsd:decimal
  }

Kind =
  element docset:Kind {
    ( "a" | "b" )
  }
"""


def _loc(page: int = 1, box: list[int] | None = None) -> dict[str, Any]:
    return {"page_number": page, "bounding_box": box if box is not None else [10, 10, 50, 20]}


def _invoice() -> dict[str, Any]:
    return {
        "Invoice": {
            "VendorName": {"text": "Acme", "locations": [_loc()]},
            "LineItems": [
                {
                    "Quantity": {"text": "2", "value": "2", "locations": [_loc()]},
                    "UnitPrice": {
                        "text": "$5",
                        "value": "5",
                        "locations": [_loc(box=[60, 10, 90, 20])],
                    },
                },
            ],
            "Kind": {"text": "A", "value": "a", "locations": [_loc()]},
        }
    }


def _validate(output: Any, values: dict[str, Any] | None = None, **kw: Any) -> None:
    if values is not None:
        output = ExtractorOutput(values=values)
    validate_extractor_output(
        output,
        vocab=parse_rnc(_INVOICE_RNC),
        page_count=kw.pop("page_count", 2),
        page_size=kw.pop("page_size", lambda _p: (100, 100)),
    )


def test_validate_accepts_a_good_tree_and_nothing_found() -> None:
    _validate(None, _invoice())
    _validate(None, {})
    _validate(None, {"Invoice": None})
    # Unprinted values with provenance, and evidence with boxes, pass.
    tree = _invoice()
    tree["Invoice"]["VendorName"] = {
        "text": "Acme",
        "method": "inferred",
        "inputs": ["LineItems[0].Quantity=2"],
        "evidence": [{"text": "Acme Co", "locations": [_loc()]}],
    }
    _validate(None, tree)


@pytest.mark.parametrize(
    ("mutate", "path"),
    [
        (lambda t: t.__setitem__("Extra", {"text": "x"}), "Extra: unknown key"),
        (lambda t: t["Invoice"].__setitem__("Nope", {"text": "x"}), "Invoice.Nope: unknown key"),
        (lambda t: t["Invoice"].__setitem__("VendorName", "Acme"), "Invoice.VendorName: a field"),
        (lambda t: t["Invoice"].__setitem__("LineItems", {}), "Invoice.LineItems: a collection"),
        (lambda t: t["Invoice"]["LineItems"].append("x"), "Invoice.LineItems[1]: an entry"),
        (lambda t: t["Invoice"]["VendorName"].__setitem__("text", 3), "Invoice.VendorName.text"),
        (lambda t: t["Invoice"]["VendorName"].__setitem__("value", 3), "Invoice.VendorName.value"),
        (lambda t: t["Invoice"]["VendorName"].__setitem__("bogus", 3), "VendorName.bogus"),
        (lambda t: t["Invoice"]["VendorName"].__setitem__("method", "guessed"), ".method"),
        (lambda t: t["Invoice"]["VendorName"].__setitem__("inputs", []), ".inputs"),
        (lambda t: t["Invoice"]["VendorName"].__setitem__("locations", "x"), ".locations: must"),
        (
            lambda t: t["Invoice"]["VendorName"]["locations"].append(3),
            "locations[1]: must be a dict",
        ),
        (
            lambda t: t["Invoice"]["VendorName"]["locations"].append({"page_number": 3}),
            "locations[1].page_number: must be an int in 1..2",
        ),
        (
            lambda t: t["Invoice"]["VendorName"]["locations"].append({"page_number": 1}),
            "locations[1].bounding_box: must be four ints",
        ),
        (
            lambda t: t["Invoice"]["VendorName"]["locations"].append(_loc(box=[10, 10, 5, 20])),
            "bounding_box: needs left < right",
        ),
        (
            lambda t: t["Invoice"]["VendorName"]["locations"].append(_loc(box=[10, 10, 50, 120])),
            "bounding_box: outside the page",
        ),
        (
            lambda t: t["Invoice"]["VendorName"].__setitem__("evidence", [{"text": "x"}]),
            "VendorName.evidence[0]: must be {text, locations}",
        ),
        (
            lambda t: t["Invoice"]["LineItems"][0]["Quantity"]["locations"].append(_loc(page=0)),
            "Invoice.LineItems[0].Quantity.locations[1].page_number",
        ),
    ],
)
def test_validate_rejects_each_rule_with_its_path(mutate: Any, path: str) -> None:
    tree = _invoice()
    mutate(tree)
    with pytest.raises(ExtractorOutputInvalid) as exc:
        _validate(None, tree)
    assert path in str(exc.value)


def test_validate_rejects_non_output_and_non_dict_and_non_json() -> None:
    with pytest.raises(ExtractorOutputInvalid, match="must return an ExtractorOutput"):
        _validate({"title": {"text": "x"}})
    with pytest.raises(ExtractorOutputInvalid, match="values must be a dict"):
        _validate(ExtractorOutput(values=[]))  # type: ignore[arg-type]
    with pytest.raises(ExtractorOutputInvalid, match="JSON-serializable"):
        _validate(
            None, {"Invoice": {"VendorName": {"text": "x", "locations": [_loc()], "value": {1}}}}
        )


def test_validate_lists_at_most_ten_problems() -> None:
    tree = {"Invoice": {f"Bad{i}": {"text": "x"} for i in range(13)}}
    with pytest.raises(ExtractorOutputInvalid) as exc:
        _validate(None, tree)
    message = str(exc.value)
    assert "13 problems" in message and "and 3 more" in message
    assert message.count("unknown key") == 10


def test_validate_skips_the_box_bounds_when_the_page_size_is_unknown() -> None:
    tree = _invoice()
    tree["Invoice"]["VendorName"]["locations"] = [_loc(box=[10, 10, 500, 500])]
    _validate(None, tree, page_size=lambda _p: None)


def test_vocabulary_problems_reports_instead_of_pruning() -> None:
    vocab = parse_rnc(_INVOICE_RNC)
    assert vocabulary_problems(_invoice(), vocab) == []
    assert vocabulary_problems({"Invoice": {"Other": {}}}, vocab) == ["Invoice.Other: unknown key"]


def test_phase1_shaped_tree_fails_as_grounded_output() -> None:
    """The seam: a phase-1 tree (pages, no boxes) is not a GROUNDED output."""
    with pytest.raises(ExtractorOutputInvalid, match="bounding_box"):
        _validate(
            None, {"Invoice": {"VendorName": {"text": "Acme", "locations": [{"page_number": 1}]}}}
        )


def test_output_json_round_trip() -> None:
    out = ExtractorOutput(
        values=_invoice(), model="m", usage={"cost_usd": 0.5}, stats={"k": [1]}, tool_calls=2
    )
    assert output_from_json(output_to_json(out)) == out


# ---------------------------------------------------------------------------
# The GROUNDED run
# ---------------------------------------------------------------------------


def test_grounded_extractor_replaces_the_phases(workspace: Workspace, fake: str) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    FakeExtractor.outputs[fid] = good_output(model="fake-model", tool_calls=3)
    with patch("litellm.completion", side_effect=AssertionError("no model call expected")):
        result = extract_values(workspace, ds_id, fid, config=_config(fake))

    assert FakeExtractor.calls == [fid]
    assert result.extractor == fake
    assert result.model == "fake-model"
    assert result.tool_calls == 3
    assert result.values["title"]["locations"] == [
        {"page_number": 1, "bounding_box": HELLO_WORLD_BOX}
    ]
    xml = workspace.blobs.get_blob(result.xml_key).decode("utf-8")
    assert f'dg:extractor="{fake}"' in xml
    vocab = parse_rnc(DocSetStore(workspace).get_schema(ds_id))
    assert dgml_xml_to_values(xml, vocab=vocab) == result.values
    stats = result.stats
    assert stats["model"] == "fake-model"
    ext = stats["extractor"]
    assert ext["provider"] == fake and ext["outcome"] == "ok" and ext["fell_back"] is False
    assert ext["boxes_verified"] == "applied" and ext["locations"] == 1
    # Plain strings in the stats, not the enum members the run holds.
    assert type(ext["outcome"]) is str and type(ext["boxes_verified"]) is str
    assert stats["matching"]["total_locations"] == 1
    assert stats["phases"]["phase3"]["page_calls"] == 0


def test_request_carries_the_docset_and_reads_content_on_each_call(
    workspace: Workspace, fake: str
) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    extract_values(workspace, ds_id, fid, config=_config(fake))
    (request,) = FakeExtractor.requests
    assert request.docset_id == ds_id
    assert request.file_id == fid and request.page_count == 1
    assert request.original_filename == "doc.pdf"
    assert "element docset:title" in request.schema_rnc
    assert "title" in json.dumps(request.schema)
    assert request.guidance is None
    assert [t.name for t in request.vocab.roots] == ["title"]
    assert request.pdf() == b"%PDF-1.4 fake\n"
    assert [w["text"] for w in request.page_words(1)["words"]] == ["Hello", "world"]
    assert request.page_words(2)["words"] == []
    image = request.page_image(1)
    assert image is not None and image[1] == _png_header(1000, 1000)
    assert request.page_size(1) == (1000, 1000)
    # Nothing is cached on the request: a changed blob is what the next call reads.
    workspace.blobs.put_blob(layout.file_source_key(fid, "doc.pdf"), b"%PDF-1.4 new\n")
    assert request.pdf() == b"%PDF-1.4 new\n"
    assert "_ws" not in repr(request)


def test_dgml_never_reads_the_pdf_when_the_extractor_succeeds(
    workspace: Workspace, fake: str
) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    with patch("dgml_core.grounded._pdf_bytes", side_effect=AssertionError("pdf read")):
        extract_values(workspace, ds_id, fid, config=_config(fake))


def test_verify_boxes_and_drop_unboxed_still_run(workspace: Workspace, fake: str) -> None:
    """A box whose words do not read as the text is rejected, and the value
    then dropped, exactly as for the model's boxes."""
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    FakeExtractor.outputs[fid] = ExtractorOutput(
        values={"title": {"text": "Goodbye", "locations": [_loc(box=HELLO_WORLD_BOX)]}}
    )
    result = extract_values(workspace, ds_id, fid, config=_config(fake))
    assert result.values == {}
    assert result.stats["matching"]["rejected_boxes"] == 1
    assert result.stats["matching"]["dropped_unboxed_values"] == 1


@pytest.mark.parametrize("batch", [False, True])
def test_self_verifying_extractor_keeps_its_boxes_and_reports_the_counts(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, batch: bool
) -> None:
    """The same under ``--batch``, where the run's one instance stays in the
    batch driver and each file's run has none."""
    provider = install(monkeypatch, SelfVerifyingExtractor)
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    # Normalized text the page does not print, on a box the extractor checked.
    SelfVerifyingExtractor.outputs[fid] = ExtractorOutput(
        values={"title": {"text": "2024-01-05", "locations": [_loc(box=HELLO_WORLD_BOX)]}}
    )
    if batch:
        _backend()
        many = extract_values_many(
            workspace, ds_id, [fid], config=_config(provider), batch=True, poll_interval_s=0
        )
        outcome = many.outcomes[fid]
        assert not isinstance(outcome, Exception)
        result = outcome
    else:
        result = extract_values(workspace, ds_id, fid, config=_config(provider))
    assert result.values["title"]["text"] == "2024-01-05"
    assert result.values["title"]["locations"] == [
        {"page_number": 1, "bounding_box": HELLO_WORLD_BOX}
    ]
    assert result.stats["extractor"]["boxes_verified"] == "reported"
    assert result.stats["matching"]["rejected_boxes"] == 1
    assert result.stats["matching"]["dropped_unboxed_values"] == 0


def test_provenance_is_normalized_like_the_models(workspace: Workspace, fake: str) -> None:
    """A boxed leaf loses a stray ``method``; an empty ``value`` goes."""
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    out = good_output()
    out.values["title"]["method"] = "inferred"
    out.values["title"]["value"] = ""
    FakeExtractor.outputs[fid] = out
    result = extract_values(workspace, ds_id, fid, config=_config(fake))
    assert "method" not in result.values["title"] and "value" not in result.values["title"]
    # The extractor's own tree is untouched: dgml worked on a copy.
    assert out.values["title"]["method"] == "inferred"


def test_telemetry_is_merged_and_carried(workspace: Workspace, fake: str) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    FakeExtractor.outputs[fid] = good_output(
        usage={"cost_usd": 0.25, "prompt_tokens": 10, "completion_tokens": 2},
        stats={"retries": 1},
    )
    result = extract_values(
        workspace, ds_id, fid, config=_config(fake), record_usage=True, write_stats=True
    )
    assert result.usage["cost_usd"] == 0.25 and result.usage["prompt_tokens"] == 10
    ext = result.stats["extractor"]
    assert ext["usage"] == {"cost_usd": 0.25, "prompt_tokens": 10, "completion_tokens": 2}
    assert ext["details"] == {"retries": 1}
    assert result.stats["phases"]["phase1"]["cost_usd"] == 0.25
    (row,) = read_events(workspace)
    assert row["model"] == fake and row["cost_usd"] == 0.25
    assert row["context"]["extractor"] == fake
    stored = workspace.docs.get_doc(layout.Collection.EXTRACTION_STATS, layout.pair_id(ds_id, fid))
    assert stored is not None and stored["extractor"]["details"] == {"retries": 1}


def test_bad_telemetry_is_dropped_with_a_warning_and_the_values_kept(
    workspace: Workspace, fake: str, caplog: pytest.LogCaptureFixture
) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    FakeExtractor.outputs[fid] = good_output(usage={"cost_usd": "lots"}, stats={"x": {1}})
    caplog.set_level(logging.WARNING, logger="dgml_core")
    result = extract_values(workspace, ds_id, fid, config=_config(fake))
    assert result.values["title"]["text"] == "Hello world"
    assert result.stats["extractor"]["usage"] is None
    assert result.stats["extractor"]["details"] is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "dropping the extractor's usage" in warnings[0]
    assert "dropping the extractor's stats" in warnings[1]


def test_unset_values_key_only_fails_a_run_that_may_fall_back(
    workspace: Workspace, fake: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The values model's key is resolved at setup only when dgml's own phases
    can run: an extractor with ``fallback = false`` never needs it."""
    monkeypatch.delenv("DGML_TEST_UNSET_KEY", raising=False)
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)

    def config(fallback: bool) -> GroundedConfig:
        return dataclasses.replace(
            _config(fake, fallback=fallback), values_api_key_env="DGML_TEST_UNSET_KEY"
        )

    with pytest.raises(AuthError, match="DGML_TEST_UNSET_KEY"):
        extract_values(workspace, ds_id, fid, config=config(fallback=True))
    assert FakeExtractor.calls == []
    result = extract_values(workspace, ds_id, fid, config=config(fallback=False))
    assert result.extractor == fake and FakeExtractor.calls == [fid]


def test_extractor_logs_go_through_the_dgml_core_logger(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)

    class Chatty(FakeExtractor):
        name: ClassVar[str] = "chatty"

        def extract(self, request: ExtractionRequest) -> ExtractorOutput:
            self.log().warning("hello from the extractor")
            logging.getLogger("my_pkg.extractors").warning("hello from a third-party logger")
            return super().extract(request)

    provider = install(monkeypatch, Chatty)
    # The CLI routes `dgml_core` only: a third-party module's own logger is not
    # under it, so capture what that logger would deliver.
    caplog.set_level(logging.INFO, logger="dgml_core")
    extract_values(workspace, ds_id, fid, config=_config(provider))
    under_dgml_core = [r for r in caplog.records if r.name.startswith("dgml_core")]
    assert any(
        r.name == "dgml_core.extractor.chatty" and r.getMessage() == "hello from the extractor"
        for r in under_dgml_core
    )
    assert not any(r.name == "my_pkg.extractors" for r in under_dgml_core)
    assert any("returned 1 locations" in r.getMessage() for r in under_dgml_core)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def test_fallback_on_raise_runs_the_phases_and_logs_once(
    workspace: Workspace, fake: str, caplog: pytest.LogCaptureFixture
) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    FakeExtractor.outputs[fid] = RuntimeError("boom")
    caplog.set_level(logging.INFO, logger="dgml_core")
    with patch("litellm.completion", return_value=_phase1(_PHASE1_VALUES)) as completion:
        result = extract_values(workspace, ds_id, fid, config=_config(fake))
        second = extract_values(workspace, ds_id, fid, config=_config(fake))
    assert completion.call_count == 2
    assert result.extractor is None and result.model == DEFAULT_VALUES_MODEL
    assert result.values["title"]["locations"] == [
        {"page_number": 1, "bounding_box": HELLO_WORLD_BOX}
    ]
    xml = workspace.blobs.get_blob(result.xml_key).decode("utf-8")
    assert "dg:extractor" not in xml
    ext = result.stats["extractor"]
    assert ext["outcome"] == "failed" and ext["fell_back"] is True
    assert ext["error"] == "RuntimeError: boom"
    assert second.stats["extractor"]["fell_back"] is True

    fallback_lines = [
        r for r in caplog.records if "extracted with dgml's own pipeline" in r.getMessage()
    ]
    assert [r.levelno for r in fallback_lines] == [logging.WARNING, logging.INFO]
    assert (
        fake in fallback_lines[0].getMessage()
        and "RuntimeError: boom" in fallback_lines[0].getMessage()
    )


def test_a_different_error_class_warns_again(
    workspace: Workspace, fake: str, caplog: pytest.LogCaptureFixture
) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    caplog.set_level(logging.WARNING, logger="dgml_core")
    with patch("litellm.completion", return_value=_phase1(_PHASE1_VALUES)):
        FakeExtractor.outputs[fid] = RuntimeError("boom")
        extract_values(workspace, ds_id, fid, config=_config(fake))
        FakeExtractor.outputs[fid] = ValueError("other")
        extract_values(workspace, ds_id, fid, config=_config(fake))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2


def test_fallback_on_invalid_output(workspace: Workspace, fake: str) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    FakeExtractor.outputs[fid] = ExtractorOutput(values={"nope": {"text": "x"}})
    with patch("litellm.completion", return_value=_phase1(_PHASE1_VALUES)):
        result = extract_values(workspace, ds_id, fid, config=_config(fake))
    assert result.extractor is None
    ext = result.stats["extractor"]
    assert ext["outcome"] == "invalid" and "nope: unknown key" in ext["error"]


def test_fallback_keeps_the_extractors_spend(workspace: Workspace, fake: str) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    FakeExtractor.outputs[fid] = ExtractorOutput(
        values={"nope": {"text": "x"}}, usage={"cost_usd": 0.5}
    )
    with patch("litellm.completion", return_value=_phase1(_PHASE1_VALUES)):
        result = extract_values(workspace, ds_id, fid, config=_config(fake))
    assert result.usage["cost_usd"] == 0.5


def test_fallback_reads_the_pdf(workspace: Workspace, fake: str) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    FakeExtractor.outputs[fid] = RuntimeError("boom")
    with (
        patch("litellm.completion", return_value=_phase1(_PHASE1_VALUES)),
        patch("dgml_core.grounded._pdf_bytes", wraps=_pdf_bytes) as reader,
    ):
        extract_values(workspace, ds_id, fid, config=_config(fake))
    assert reader.call_count == 1


def test_no_fallback_raises_with_usage_attached(workspace: Workspace, fake: str) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    FakeExtractor.outputs[fid] = RuntimeError("boom")
    with patch("litellm.completion", side_effect=AssertionError("no model call expected")):
        with pytest.raises(ExtractorFailed, match="boom") as exc:
            extract_values(workspace, ds_id, fid, config=_config(fake, fallback=False))
    assert isinstance(exc.value, ValuesExtractionFailed)
    # A foreign exception is named by type; the message names the extractor.
    assert str(exc.value) == f"extractor {fake} failed: RuntimeError: boom"
    assert exc.value.usage is not None and "duration_s" in exc.value.usage
    assert exc.value.stats is not None and exc.value.stats["extractor"]["outcome"] == "failed"
    # No model ran: the stats say so, and so does the usage row.
    assert exc.value.stats["model"] is None
    assert not workspace.blobs.blob_exists(layout.dgml_xml_key(ds_id, fid, "doc"))
    with pytest.raises(ExtractorFailed):
        extract_values(
            workspace, ds_id, fid, config=_config(fake, fallback=False), record_usage=True
        )
    (row,) = read_events(workspace)
    assert row["model"] is None and row["outcome"] == "error"
    assert row["context"]["extractor"] is None and "boom" in row["error"]

    FakeExtractor.outputs[fid] = ExtractorOutput(values={"nope": {"text": "x"}})
    with pytest.raises(ExtractorOutputInvalid, match="nope") as invalid:
        extract_values(workspace, ds_id, fid, config=_config(fake, fallback=False))
    assert str(invalid.value).startswith(f"extractor {fake} failed: extractor output is invalid")

    # An ExtractorFailed the extractor raised itself keeps its message, prefixed
    # the same way and without a type name.
    FakeExtractor.outputs[fid] = ExtractorFailed("no recording for this file")
    with pytest.raises(ExtractorFailed) as own:
        extract_values(workspace, ds_id, fid, config=_config(fake, fallback=False))
    assert str(own.value) == f"extractor {fake} failed: no recording for this file"
    assert own.value.code == "EXTRACTOR_FAILED"


def test_a_crashing_constructor_falls_back_or_fails(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = install(monkeypatch, CrashingInit)
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    with patch("litellm.completion", return_value=_phase1(_PHASE1_VALUES)):
        result = extract_values(workspace, ds_id, fid, config=_config(provider))
    assert result.extractor is None and result.stats["extractor"]["error"] == "RuntimeError: no SDK"
    # The fallback ran: the values model is what ran.
    assert result.model == DEFAULT_VALUES_MODEL and result.stats["model"] == DEFAULT_VALUES_MODEL
    with pytest.raises(ExtractorFailed, match="no SDK"):
        extract_values(workspace, ds_id, fid, config=_config(provider, fallback=False))


def test_an_interrupt_never_falls_back(workspace: Workspace, fake: str) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    FakeExtractor.outputs[fid] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        extract_values(workspace, ds_id, fid, config=_config(fake))


def test_extract_file_fallback_override(workspace: Workspace, fake: str) -> None:
    write_config(
        workspace,
        {
            "grounded": {
                "values_model": DEFAULT_VALUES_MODEL,
                "schema_model": DEFAULT_SCHEMA_MODEL,
            },
            "extractor": {"provider": fake},
        },
    )
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    FakeExtractor.outputs[fid] = RuntimeError("boom")
    with pytest.raises(ExtractorFailed):
        extract_file(workspace, ds_id, fid, extractor_fallback=False)
    assert extraction_config(workspace).extractor is not None
    assert extraction_config(workspace, extractor_fallback=False).extractor.fallback is False  # type: ignore[union-attr]
    write_config(
        workspace,
        {"grounded": {"values_model": DEFAULT_VALUES_MODEL, "schema_model": DEFAULT_SCHEMA_MODEL}},
    )
    with pytest.raises(InvalidArgument, match="no custom extractor"):
        extraction_config(workspace, extractor_fallback=False)


# ---------------------------------------------------------------------------
# Many files, synchronously
# ---------------------------------------------------------------------------


def _seed_many(workspace: Workspace, n: int) -> tuple[str, list[str]]:
    fids = [f"f{i:011d}" for i in range(n)]
    ds_id = _seed(workspace, fids[0])
    store = DocSetStore(workspace)
    for fid in fids[1:]:
        _seed_file(workspace, fid, filename=f"{fid}.pdf")
        _seed_page_text(workspace, fid, page=1)
        workspace.blobs.put_blob(layout.file_page_image_key(fid, 1), _png_header(1000, 1000))
        store.add_file(ds_id, fid)
    return ds_id, fids


def test_many_files_summarize_fallbacks_in_one_warning(
    workspace: Workspace, fake: str, caplog: pytest.LogCaptureFixture
) -> None:
    ds_id, fids = _seed_many(workspace, 7)
    for fid in fids:
        FakeExtractor.outputs[fid] = RuntimeError("boom")
    caplog.set_level(logging.INFO, logger="dgml_core")
    with patch("litellm.completion", return_value=_phase1(_PHASE1_VALUES)):
        many = extract_values_many(workspace, ds_id, fids, config=_config(fake))
    assert all(not isinstance(o, Exception) for o in many.outcomes.values())
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "RuntimeError for 7 files" in warnings[0]
    assert ", ".join(fids[:5]) in warnings[0] and "and 2 more" in warnings[0]
    assert fids[6] not in warnings[0]
    # --verbose still names every file.
    per_file = [
        r for r in caplog.records if r.levelno == logging.INFO and "falling back" in r.getMessage()
    ]
    assert len(per_file) == 7


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _registry() -> Iterator[None]:
    saved = dict(batch_registry._REGISTRY)
    try:
        yield
    finally:
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)


@pytest.fixture(autouse=True)
def _no_leaked_session() -> Iterator[None]:
    yield
    session = active_session()
    if session is not None:
        session.close(None)
    assert llm._SYNC_RECORDER is None


def _backend(responder: Any = None) -> FakeBackend:
    def answer(req: Any) -> Any:
        if responder is None:
            raise AssertionError("no model request expected")
        return responder(req.kwargs)

    backend = FakeBackend(answer)
    register_backend(provider_of(DEFAULT_VALUES_MODEL), lambda _cfg: backend)
    return backend


def test_batch_with_an_extractor_makes_no_model_request(workspace: Workspace, fake: str) -> None:
    ds_id, fids = _seed_many(workspace, 3)
    backend = _backend()
    many = extract_values_many(
        workspace, ds_id, fids, config=_config(fake), batch=True, poll_interval_s=0
    )
    assert backend.submitted == []
    assert many.batch is not None and many.batch["requests"] == 0
    assert list(many.outcomes) == fids
    for outcome in many.outcomes.values():
        assert not isinstance(outcome, Exception) and outcome.extractor == fake
    # One extract_many call over every pair, in input order; one construction.
    assert FakeExtractor.many_calls == [fids] and FakeExtractor.constructed == 1


def test_batch_fallback_pairs_get_a_phase1_unit(workspace: Workspace, fake: str) -> None:
    ds_id, fids = _seed_many(workspace, 3)
    FakeExtractor.outputs[fids[1]] = RuntimeError("boom")

    def respond(kwargs: dict[str, Any]) -> Any:
        return _phase1(_PHASE1_VALUES)

    backend = _backend(respond)
    many = extract_values_many(
        workspace, ds_id, fids, config=_config(fake), batch=True, poll_interval_s=0
    )
    assert len(backend.submitted) == 1  # phase 1 for the fallback pair only
    assert many.batch is not None and many.batch["requests"] == 1
    outcomes = many.outcomes
    assert [getattr(o, "extractor", "ERR") for o in outcomes.values()] == [fake, None, fake]


def test_batch_outcomes_apply_in_pair_order_when_the_pool_returns_out_of_order(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds_id, fids = _seed_many(workspace, 3)
    gate = threading.Barrier(3)

    class Slow(FakeExtractor):
        name: ClassVar[str] = "slow"

        def extract(self, request: ExtractionRequest) -> ExtractorOutput:
            gate.wait(timeout=5)  # all three in flight at once
            return super().extract(request)

    provider = install(monkeypatch, Slow)
    for fid in fids:
        Slow.outputs[fid] = ExtractorOutput(
            values={"title": {"text": "Hello world", "locations": [_loc(box=HELLO_WORLD_BOX)]}},
            model=f"model-{fid}",
        )
    _backend()
    many = extract_values_many(
        workspace,
        ds_id,
        fids,
        config=_config(provider, max_concurrency=3),
        batch=True,
        poll_interval_s=0,
    )
    assert [getattr(o, "model", "ERR") for o in many.outcomes.values()] == [
        f"model-{f}" for f in fids
    ]


def test_batch_honors_an_extract_many_override(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Bulk(FakeExtractor):
        name: ClassVar[str] = "bulk"

        def extract_many(
            self, requests: Sequence[ExtractionRequest], *, max_concurrency: int
        ) -> list[ExtractorOutput | Exception]:
            type(self).many_calls.append([r.file_id for r in requests])
            return [
                good_output() if i != 1 else RuntimeError("one bad") for i in range(len(requests))
            ]

    provider = install(monkeypatch, Bulk)
    ds_id, fids = _seed_many(workspace, 3)
    _backend(lambda kwargs: _phase1(_PHASE1_VALUES))
    many = extract_values_many(
        workspace, ds_id, fids, config=_config(provider), batch=True, poll_interval_s=0
    )
    assert Bulk.calls == []  # extract never called
    assert Bulk.many_calls == [fids]
    assert [getattr(o, "extractor", "ERR") for o in many.outcomes.values()] == [
        provider,
        None,
        provider,
    ]


def test_batch_extract_many_raising_fails_every_pair_or_falls_back(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(FakeExtractor):
        name: ClassVar[str] = "broken"

        def extract_many(
            self, requests: Sequence[ExtractionRequest], *, max_concurrency: int
        ) -> list[ExtractorOutput | Exception]:
            raise RuntimeError("provider down")

    provider = install(monkeypatch, Broken)
    ds_id, fids = _seed_many(workspace, 2)
    _backend()
    many = extract_values_many(
        workspace,
        ds_id,
        fids,
        config=_config(provider, fallback=False),
        batch=True,
        poll_interval_s=0,
    )
    assert all(
        isinstance(o, ExtractorFailed) and "provider down" in str(o) for o in many.outcomes.values()
    )


def test_batch_wrong_length_from_extract_many_is_a_failure(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Short(FakeExtractor):
        name: ClassVar[str] = "short"

        def extract_many(
            self, requests: Sequence[ExtractionRequest], *, max_concurrency: int
        ) -> list[ExtractorOutput | Exception]:
            return [good_output()]

    provider = install(monkeypatch, Short)
    ds_id, fids = _seed_many(workspace, 2)
    _backend()
    many = extract_values_many(
        workspace,
        ds_id,
        fids,
        config=_config(provider, fallback=False),
        batch=True,
        poll_interval_s=0,
    )
    assert all(
        isinstance(o, ExtractorFailed) and "returned 1 outcomes for 2 requests" in str(o)
        for o in many.outcomes.values()
    )


def test_batch_with_an_unset_values_key_extracts_when_the_extractor_may_not_fall_back(
    workspace: Workspace, fake: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DGML_TEST_UNSET_KEY", raising=False)
    ds_id, fids = _seed_many(workspace, 2)
    _backend()
    config = dataclasses.replace(
        _config(fake, fallback=False), values_api_key_env="DGML_TEST_UNSET_KEY"
    )
    many = extract_values_many(workspace, ds_id, fids, config=config, batch=True, poll_interval_s=0)
    # The key is never read, so the run stays a batch run: one extract_many.
    assert many.batch == no_batch_block()
    for outcome in many.outcomes.values():
        assert not isinstance(outcome, Exception) and outcome.extractor == fake
    assert FakeExtractor.many_calls == [fids]


def test_batch_with_an_unbatchable_values_model_extracts_when_the_extractor_may_not_fall_back(
    workspace: Workspace, fake: str
) -> None:
    """The values model is only checked for a batch backend when it may run."""
    ds_id, fids = _seed_many(workspace, 2)
    unbatchable = "nobatchprovider/some-model"
    may_fall_back = dataclasses.replace(_config(fake), values_model=unbatchable)
    with pytest.raises(BatchUnavailable):
        extract_values_many(
            workspace, ds_id, fids, config=may_fall_back, batch=True, poll_interval_s=0
        )

    config = dataclasses.replace(_config(fake, fallback=False), values_model=unbatchable)
    many = extract_values_many(workspace, ds_id, fids, config=config, batch=True, poll_interval_s=0)
    block = many.batch
    assert block is not None and block["provider"] is None and block["requests"] == 0
    for outcome in many.outcomes.values():
        assert not isinstance(outcome, Exception) and outcome.extractor == fake
    assert FakeExtractor.many_calls == [fids]


def test_batch_job_with_no_executor_records_and_replays_outputs(
    workspace: Workspace, fake: str
) -> None:
    """With no executor the job still records each accepted output, and the
    resumed run replays them without calling the extractor again."""
    ds_id, fids = _seed_many(workspace, 2)
    config = dataclasses.replace(
        _config(fake, fallback=False), values_model="nobatchprovider/some-model"
    )
    argv = ["extraction", "extract", ds_id, "--all", "--batch"]
    session = start_session(
        workspace, command="extraction extract", argv=argv, job_id=None, wait=True
    )
    extract_values_many(workspace, ds_id, fids, config=config, batch=True, poll_interval_s=0)
    for fid in fids:
        assert session.recorded_input(f"extractor/{ds_id}/{fid}.json") is not None
    session.close(BatchPending(session.job_id, submitted_batches=0, requests_in_flight=0))

    resumed = start_session(
        workspace, command="extraction extract", argv=argv, job_id=session.job_id, wait=True
    )
    many = extract_values_many(workspace, ds_id, fids, config=config, batch=True, poll_interval_s=0)
    resumed.close(None)
    assert all(getattr(o, "extractor", None) == fake for o in many.outcomes.values())
    assert FakeExtractor.many_calls == [fids]  # only the first run asked


@pytest.mark.parametrize("fallback", [True, False])
def test_bulk_classify_batch_checks_the_values_model_only_when_it_may_run(
    workspace: Workspace, fake: str, fallback: bool
) -> None:
    """`file add --batch` auto-extraction: an extractor that may not fall back
    needs no values-model executor, so an unbatchable values model is not
    refused, and the files are extracted with a zero `batch` block."""
    from dgml_core.auto_classification import _extract_assigned_batch, prepare_bulk_classify
    from dgml_core.classification import ClassificationConfig

    ds_id, fids = _seed_many(workspace, 2)
    _backend()
    write_config(
        workspace,
        {
            "grounded": {
                "values_model": "nobatchprovider/some-model",
                "schema_model": DEFAULT_SCHEMA_MODEL,
            },
            "extractor": {"provider": fake, "fallback": fallback},
        },
    )
    docsets = DocSetStore(workspace).list_all()
    config = ClassificationConfig(model=DEFAULT_VALUES_MODEL)
    if fallback:
        with pytest.raises(BatchUnavailable, match="nobatchprovider"):
            prepare_bulk_classify(workspace, config=config, docsets=docsets, poll_interval_s=0)
        return
    state = prepare_bulk_classify(workspace, config=config, docsets=docsets, poll_interval_s=0)
    assert state.grounded is not None and state.extractor is None
    assert state.extractor_error is None

    items: dict[tuple[str, str], dict[str, Any]] = {(ds_id, fid): {} for fid in fids}
    stats = _extract_assigned_batch(
        workspace, items, batch=state, write_stats=False, record_usage=False
    )
    assert stats is not None and stats["provider"] is None and stats["requests"] == 0
    assert stats["docset_ids"] == [ds_id]
    for block in items.values():
        assert block["extraction"]["error"] is None
        assert block["extraction"]["extractor"] == fake
    assert FakeExtractor.many_calls == [fids]


def test_batch_pairs_need_an_executor_when_the_values_model_may_run(
    workspace: Workspace, fake: str
) -> None:
    ds_id, fids = _seed_many(workspace, 1)
    with pytest.raises(ValueError, match="executor is required"):
        extract_values_batch_pairs(
            workspace, [(ds_id, fids[0])], config=_config(fake), executor=None
        )
    assert FakeExtractor.many_calls == []


def test_batch_summarizes_fallbacks(
    workspace: Workspace, fake: str, caplog: pytest.LogCaptureFixture
) -> None:
    ds_id, fids = _seed_many(workspace, 7)
    for fid in fids:
        FakeExtractor.outputs[fid] = RuntimeError("boom")
    _backend(lambda kwargs: _phase1(_PHASE1_VALUES))
    caplog.set_level(logging.WARNING, logger="dgml_core")
    extract_values_many(workspace, ds_id, fids, config=_config(fake), batch=True, poll_interval_s=0)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "for 7 files" in warnings[0] and "and 2 more" in warnings[0]


def test_job_mode_calls_the_extractor_once_across_a_pause_and_resume(
    workspace: Workspace, fake: str
) -> None:
    """A --no-wait run followed by a resume: each file reaches extract_many once
    across both runs, the recorded outputs are read on the resume, only the
    unrecorded file (whose output was invalid, so never recorded) is passed
    again, and the fallback's phase-1 request digests identically."""
    ds_id, fids = _seed_many(workspace, 3)
    FakeExtractor.outputs[fids[2]] = ExtractorOutput(values={"nope": {"text": "x"}})
    backend = FakeBackend(lambda req: _phase1(_PHASE1_VALUES), polls_until_ended=1)
    register_backend(provider_of(DEFAULT_VALUES_MODEL), lambda _cfg: backend)
    argv = ["extraction", "extract", ds_id, "--all", "--batch"]

    session = start_session(
        workspace, command="extraction extract", argv=argv, job_id=None, wait=False
    )
    with pytest.raises(BatchPending) as pending:
        extract_values_many(
            workspace, ds_id, fids, config=_config(fake), batch=True, poll_interval_s=0
        )
    session.close(pending.value)
    assert FakeExtractor.many_calls == [fids]
    manifest = BatchJobStore(workspace, session.job_id).load()
    recorded = {name for name, present in manifest.inputs.items() if present}
    assert recorded == {f"extractor/{ds_id}/{fid}.json" for fid in fids[:2]}

    resumed = start_session(
        workspace, command="extraction extract", argv=argv, job_id=session.job_id, wait=True
    )
    many = extract_values_many(
        workspace, ds_id, fids, config=_config(fake), batch=True, poll_interval_s=0
    )
    resumed.close(None)
    # The resume asked only for the unrecorded file; nothing was resubmitted.
    assert FakeExtractor.many_calls == [fids, [fids[2]]]
    assert len(backend.submitted) == 1
    assert [getattr(o, "extractor", "ERR") for o in many.outcomes.values()] == [fake, fake, None]


def test_job_mode_records_an_output_whose_telemetry_does_not_encode(
    workspace: Workspace, fake: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Stats that are not JSON fail neither the file nor the run: the record
    holds the tree without them, the resume replays it, and the drop is warned
    about once."""
    ds_id, fids = _seed_many(workspace, 2)
    FakeExtractor.outputs[fids[0]] = good_output(
        usage={"cost_usd": 0.5}, stats={"when": datetime.now(UTC)}
    )
    _backend()
    caplog.set_level(logging.WARNING, logger="dgml_core")
    argv = ["extraction", "extract", ds_id, "--all", "--batch"]

    session = start_session(
        workspace, command="extraction extract", argv=argv, job_id=None, wait=True
    )
    many = extract_values_many(
        workspace, ds_id, fids, config=_config(fake), batch=True, poll_interval_s=0
    )
    session.close(BatchPending(session.job_id, submitted_batches=0, requests_in_flight=0))
    assert not any(isinstance(o, Exception) for o in many.outcomes.values())
    record = session.recorded_input(f"extractor/{ds_id}/{fids[0]}.json")
    assert record is not None
    replayed = output_from_json(record)
    assert replayed.stats is None and replayed.usage == {"cost_usd": 0.5}

    resumed = start_session(
        workspace, command="extraction extract", argv=argv, job_id=session.job_id, wait=True
    )
    extract_values_many(workspace, ds_id, fids, config=_config(fake), batch=True, poll_interval_s=0)
    resumed.close(None)
    assert FakeExtractor.many_calls == [fids]  # the resume replayed both records
    drops = [r.getMessage() for r in caplog.records if "dropping the extractor's" in r.getMessage()]
    assert len(drops) == 1 and "stats" in drops[0]


def test_sync_run_under_a_job_records_the_output(workspace: Workspace, fake: str) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    session = start_session(workspace, command="x", argv=["x"], job_id=None, wait=True)
    extract_values(workspace, ds_id, fid, config=_config(fake))
    session.persist()
    assert session.recorded_input(f"extractor/{ds_id}/{fid}.json") is not None
    session.close(BatchPending(session.job_id, submitted_batches=0, requests_in_flight=0))
    resumed = start_session(workspace, command="x", argv=["x"], job_id=session.job_id, wait=True)
    extract_values(workspace, ds_id, fid, config=_config(fake))
    resumed.close(None)
    assert FakeExtractor.calls == [fid]  # the resume replayed the record


def _crash_on_init(self: FakeExtractor, config: ExtractorConfig) -> None:
    raise RuntimeError("credential gone")


def test_sync_resume_replays_the_record_when_the_extractor_cannot_be_constructed(
    workspace: Workspace, fake: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    config = _config(fake, fallback=False)
    session = start_session(workspace, command="x", argv=["x"], job_id=None, wait=True)
    extract_values(workspace, ds_id, fid, config=config)
    session.close(BatchPending(session.job_id, submitted_batches=0, requests_in_flight=0))

    monkeypatch.setattr(FakeExtractor, "__init__", _crash_on_init)
    resumed = start_session(workspace, command="x", argv=["x"], job_id=session.job_id, wait=True)
    result = extract_values(workspace, ds_id, fid, config=config)
    resumed.close(None)
    assert result.extractor == fake and result.stats["extractor"]["outcome"] == "ok"
    assert FakeExtractor.calls == [fid]


def test_batch_resume_replays_records_when_the_extractor_cannot_be_constructed(
    workspace: Workspace, fake: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the file with no record fails on the constructor; the recorded
    ones are replayed, with no model request (``fallback = false``)."""
    ds_id, fids = _seed_many(workspace, 3)
    FakeExtractor.outputs[fids[2]] = RuntimeError("boom")
    _backend()
    config = _config(fake, fallback=False)
    argv = ["extraction", "extract", ds_id, "--all", "--batch"]
    session = start_session(
        workspace, command="extraction extract", argv=argv, job_id=None, wait=True
    )
    extract_values_many(workspace, ds_id, fids, config=config, batch=True, poll_interval_s=0)
    session.close(BatchPending(session.job_id, submitted_batches=0, requests_in_flight=0))

    monkeypatch.setattr(FakeExtractor, "__init__", _crash_on_init)
    resumed = start_session(
        workspace, command="extraction extract", argv=argv, job_id=session.job_id, wait=True
    )
    many = extract_values_many(workspace, ds_id, fids, config=config, batch=True, poll_interval_s=0)
    resumed.close(None)
    assert [getattr(o, "extractor", None) for o in many.outcomes.values()] == [fake, fake, None]
    assert "credential gone" in str(many.outcomes[fids[2]])
    assert FakeExtractor.many_calls == [fids]  # only the first run asked


def test_sync_run_under_a_job_keeps_an_invalid_outputs_cost(
    workspace: Workspace, fake: str
) -> None:
    """A rejected output under a job is reported as invalid, with what the
    extractor spent, exactly as without one; it is not recorded."""
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    bad = good_output().values
    bad["title"]["locations"][0]["page_number"] = 9
    FakeExtractor.outputs[fid] = ExtractorOutput(values=bad, usage={"cost_usd": 0.5})
    session = start_session(workspace, command="x", argv=["x"], job_id=None, wait=True)
    with pytest.raises(ExtractorOutputInvalid) as err:
        extract_values(workspace, ds_id, fid, config=_config(fake, fallback=False))
    assert session.recorded_input(f"extractor/{ds_id}/{fid}.json") is None
    session.close(None)
    assert err.value.usage is not None and err.value.usage["cost_usd"] == 0.5


def test_sync_run_under_a_job_records_an_output_whose_telemetry_does_not_encode(
    workspace: Workspace, fake: str, caplog: pytest.LogCaptureFixture
) -> None:
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    FakeExtractor.outputs[fid] = good_output(stats={"when": datetime.now(UTC)})
    caplog.set_level(logging.WARNING, logger="dgml_core")
    session = start_session(workspace, command="x", argv=["x"], job_id=None, wait=True)
    result = extract_values(workspace, ds_id, fid, config=_config(fake, fallback=False))
    record = session.recorded_input(f"extractor/{ds_id}/{fid}.json")
    session.close(BatchPending(session.job_id, submitted_batches=0, requests_in_flight=0))
    assert result.extractor == fake and result.stats["extractor"]["details"] is None
    assert record is not None and output_from_json(record).stats is None

    resumed = start_session(workspace, command="x", argv=["x"], job_id=session.job_id, wait=True)
    extract_values(workspace, ds_id, fid, config=_config(fake, fallback=False))
    resumed.close(None)
    assert FakeExtractor.calls == [fid]  # the resume replayed the record
    drops = [r.getMessage() for r in caplog.records if "dropping the extractor's" in r.getMessage()]
    assert len(drops) == 1 and "stats" in drops[0]


def test_recorded_input_and_record_input_compose_rewind(workspace: Workspace) -> None:
    session = start_session(workspace, command="x", argv=["x"], job_id=None, wait=True)
    assert session.recorded_input("a") is None
    session.record_input("a", b"1")
    session.record_input("b", None)
    assert session.recorded_input("a") == b"1"
    assert session.recorded_input("b") is None
    assert session.rewind_input("a", lambda: b"2") == b"1"
    assert session.rewind_input("b", lambda: b"2") is None
    session.close(None)


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

_ANTHROPIC_KEY = "sk-ant-extractor-test-key"
_GOOGLE_KEY = "gk-google-extractor-test-key"


@pytest.fixture
def keyed(monkeypatch: pytest.MonkeyPatch) -> str:
    return install(monkeypatch, KeyedExtractor)


def _write_keyed_config(workspace: Workspace, provider: str, **models: Any) -> None:
    write_config(
        workspace,
        {
            "models": models,
            "grounded": {"values_model": DEFAULT_VALUES_MODEL, "schema_model": "s"},
            "extractor": {"provider": provider},
        },
    )


def test_declared_provider_gets_its_models_key(workspace: Workspace, keyed: str) -> None:
    """The class gets the key of the provider it asks for, not another one, and
    parse_config never sees a key."""
    _write_keyed_config(
        workspace, keyed, anthropic_api_key=_ANTHROPIC_KEY, google_api_key=_GOOGLE_KEY
    )
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    extract_values(workspace, ds_id, fid, config=load_grounded_config(workspace))

    (config,) = KeyedExtractor.configs
    assert config.api_keys == {Provider.ANTHROPIC: _ANTHROPIC_KEY}
    assert config.api_key(Provider.ANTHROPIC) == _ANTHROPIC_KEY
    assert config.api_key(Provider.GOOGLE) is None
    assert KeyedExtractor.parsed and all(c.api_keys == {} for c in KeyedExtractor.parsed)


def test_key_env_is_read_when_the_extractor_is_built(
    workspace: Workspace, keyed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_keyed_config(workspace, keyed, anthropic_api_key_env="MY_ANTHROPIC_KEY")
    config = load_grounded_config(workspace)  # loading reads no env var
    monkeypatch.setenv("MY_ANTHROPIC_KEY", _ANTHROPIC_KEY)
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    extract_values(workspace, ds_id, fid, config=config)
    (built,) = KeyedExtractor.configs
    assert built.api_key(Provider.ANTHROPIC) == _ANTHROPIC_KEY


def test_unset_key_env_passes_no_key_with_a_warning(
    workspace: Workspace,
    keyed: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("MY_ANTHROPIC_KEY", raising=False)
    _write_keyed_config(workspace, keyed, anthropic_api_key_env="MY_ANTHROPIC_KEY")
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    with caplog.at_level(logging.WARNING, logger="dgml_core"):
        extract_values(workspace, ds_id, fid, config=load_grounded_config(workspace))
    (built,) = KeyedExtractor.configs
    assert built.api_key(Provider.ANTHROPIC) is None
    assert "MY_ANTHROPIC_KEY, which is not set" in caplog.text


def test_no_configured_key_is_none_not_an_error(workspace: Workspace, keyed: str) -> None:
    _write_keyed_config(workspace, keyed)
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    result = extract_values(workspace, ds_id, fid, config=load_grounded_config(workspace))
    assert result.extractor == keyed
    (built,) = KeyedExtractor.configs
    assert built.api_keys == {} and built.api_key(Provider.ANTHROPIC) is None


def test_keys_stay_out_of_repr_stats_usage_and_xml(workspace: Workspace, keyed: str) -> None:
    _write_keyed_config(workspace, keyed, anthropic_api_key=_ANTHROPIC_KEY)
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(workspace, fid)
    config = load_grounded_config(workspace)
    assert _ANTHROPIC_KEY not in repr(config)
    result = extract_values(
        workspace, ds_id, fid, config=config, record_usage=True, write_stats=True
    )
    (built,) = KeyedExtractor.configs
    assert _ANTHROPIC_KEY not in repr(built) and "anthropic" in repr(built)
    assert _ANTHROPIC_KEY not in json.dumps(result.stats)
    assert _ANTHROPIC_KEY not in json.dumps(read_events(workspace))
    assert _ANTHROPIC_KEY not in workspace.blobs.get_blob(result.xml_key).decode("utf-8")


def test_in_memory_configuration_passes_the_key(keyed: str, tmp_path: Path) -> None:
    cfg = Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="n", organization="o"),
        storage=Storage.combined(ProviderSpec("dgml_core.storage_local:LocalStore")),
        models=Models(
            advanced=Model(DEFAULT_VALUES_MODEL),
            expert=Model(DEFAULT_SCHEMA_MODEL),
            anthropic_api_key=_ANTHROPIC_KEY,
        ),
        extractor=ExtractorSection(provider=keyed),
    )
    ws = Workspace.open(configuration=cfg, root=tmp_path)
    fid = "f1aaaaaaaaaa"
    ds_id = _seed(ws, fid)
    extract_file(ws, ds_id, fid)
    (built,) = KeyedExtractor.configs
    assert built.api_key(Provider.ANTHROPIC) == _ANTHROPIC_KEY


def test_batch_passes_the_key(workspace: Workspace, keyed: str) -> None:
    _write_keyed_config(workspace, keyed, anthropic_api_key=_ANTHROPIC_KEY)
    ds_id, fids = _seed_many(workspace, 2)
    _backend()
    extract_values_many(
        workspace,
        ds_id,
        fids,
        config=load_grounded_config(workspace),
        batch=True,
        poll_interval_s=0,
    )
    (built,) = KeyedExtractor.configs
    assert built.api_key(Provider.ANTHROPIC) == _ANTHROPIC_KEY


def test_make_extractor_without_credentials_keeps_the_given_keys(keyed: str) -> None:
    given = ExtractorConfig(provider=keyed, api_keys={Provider.ANTHROPIC: "sk-given"})
    ext = make_extractor(given)
    assert isinstance(ext, KeyedExtractor) and ext.config.api_keys == {
        Provider.ANTHROPIC: "sk-given"
    }
    # With credentials, the resolved keys replace what the config held.
    ext = make_extractor(given, credentials={Provider.GOOGLE: (_GOOGLE_KEY, None)})
    assert isinstance(ext, KeyedExtractor) and ext.config.api_keys == {}


@pytest.mark.parametrize(
    ("providers", "match"),
    [
        (frozenset({"mistral"}), "unknown providers \\['mistral'\\]"),
        ("anthropic", "must be a set of providers"),
        (frozenset({3}), "must be a set of providers"),
    ],
)
def test_bad_api_key_providers_fail_at_config_load(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, providers: Any, match: str
) -> None:
    class BadKeys(FakeExtractor):
        name: ClassVar[str] = "badkeys"

    BadKeys.api_key_providers = providers
    provider = install(monkeypatch, BadKeys)
    _write_keyed_config(workspace, provider)
    with pytest.raises(ExtractorConfigInvalid, match=match):
        load_extractor_config(workspace)
