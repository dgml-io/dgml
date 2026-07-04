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

"""The ``dg:style`` allow-list and fact->CSS assembly.

``dg:style`` records *observed* visual formatting as inline CSS that can be
copied verbatim into an HTML ``style`` attribute. This module is the single
source of truth for which property/value pairs are permitted (the spec §9
table) and for turning the raw style facts gathered during extraction /
grounding into a canonical, validated declaration string.

Everything here is pure and string/number-level — no PDF, OCR, or XML
knowledge. Both the deterministic path (digital/hybrid, facts from pdfminer)
and the LLM-from-image path (ocr) funnel through
:func:`build_style` / :func:`validate_style` so only allowed pairs are ever
emitted.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

# ---- Allow-list (spec §9) --------------------------------------------------

# Property -> the set of permitted values, or ``None`` meaning "any CSS named
# color" (validated against :data:`CSS_NAMED_COLORS`). This is the only place
# the permitted vocabulary is defined.
#
# Not every permitted property is *derived* the same way. The deterministic
# path (digital/hybrid) derives ``font-weight``, ``font-style``, ``font-size``,
# ``color``, and ``text-transform`` (all-caps) from the PDF glyphs. ``text-align``
# is derived only by the OCR image path (:mod:`dgml.style_llm`) — page-relative
# geometry can't reliably tell right-aligned text from a left-aligned column, so
# the deterministic path leaves it alone. The three marked "image path only"
# below are likewise never derived deterministically; they are permitted so
# authored / LLM-supplied values validate and survive (deriving them from the
# PDF would need line/rect/fill analysis) and are reserved for future work.
ALLOWED: dict[str, set[str] | None] = {
    "font-weight": {"bold", "normal"},
    "font-style": {"italic", "normal"},
    "font-size": {"0.75em", "1em", "1.25em", "1.5em", "2em"},
    "text-decoration": {"underline", "line-through", "none"},  # image path only
    "text-align": {"left", "center", "right", "justify"},  # image path only
    "text-transform": {"uppercase", "capitalize", "none"},
    "color": None,
    "background-color": None,  # image path only
    "white-space": {"pre", "normal"},  # image path only
}

# Canonical emission order — stable output regardless of fact-gathering order.
_PROPERTY_ORDER: tuple[str, ...] = (
    "font-weight",
    "font-style",
    "font-size",
    "text-decoration",
    "text-align",
    "text-transform",
    "color",
    "background-color",
    "white-space",
)

# CSS-inheriting properties among our allow-list: a child inherits these from
# its nearest styled ancestor, so restating the same value on the child is
# redundant (see :func:`dgml_core.xml_grounding._suppress_inherited_style`).
# ``text-decoration`` and ``background-color`` are intentionally absent — they
# do **not** inherit, so a descendant repeating them is meaningful, not noise.
INHERITED_PROPERTIES: frozenset[str] = frozenset(
    {
        "font-weight",
        "font-style",
        "font-size",
        "text-align",
        "text-transform",
        "color",
        "white-space",
    }
)

# Values that mean "default rendering". ``color``/``background-color`` have no
# default keyword; they are simply not set when the observed color is the
# default (near-black text / no fill).
#
# These are elided so the attribute stays sparse, but *how* depends on whether
# the property inherits. A non-inheriting default (``text-decoration: none``)
# can never be a meaningful override, so :func:`build_style` drops it outright.
# An inheriting default (``font-weight: normal`` etc.) IS meaningful when it
# overrides a non-default inherited from an ancestor, so it survives
# :func:`build_style` and is elided later — only where truly redundant — by
# :func:`dgml_core.xml_grounding._suppress_inherited_style`, whose root context
# is seeded with :data:`INHERITED_DEFAULTS` (the CSS initial values).
_DEFAULTS: dict[str, str] = {
    "font-weight": "normal",
    "font-style": "normal",
    "font-size": "1em",
    "text-decoration": "none",
    "text-align": "left",
    "text-transform": "none",
    "white-space": "normal",
}

# The inheriting subset of ``_DEFAULTS`` — the effective inherited value at the
# document root, used to seed the inheritance walk in the suppression pass.
INHERITED_DEFAULTS: dict[str, str] = {
    prop: val for prop, val in _DEFAULTS.items() if prop in INHERITED_PROPERTIES
}

# ---- Font-name heuristics --------------------------------------------------

# pdfminer font names look like ``ABCDEF+Times-Bold`` (an optional subset
# prefix + the PostScript name). Substring match over the whole name. The
# bold pattern subsumes ``semibold``/``demibold`` etc. since they contain
# ``bold``.
_BOLD_RE = re.compile(r"bold|black|heavy", re.IGNORECASE)
_ITALIC_RE = re.compile(r"italic|oblique", re.IGNORECASE)


def fontname_is_bold(name: str | None) -> bool:
    """Whether a PostScript font name denotes a bold weight."""
    return bool(name) and _BOLD_RE.search(name or "") is not None


def fontname_is_italic(name: str | None) -> bool:
    """Whether a PostScript font name denotes an italic/oblique style."""
    return bool(name) and _ITALIC_RE.search(name or "") is not None


# ---- Font size -> em bucket -------------------------------------------------

_EM_BUCKETS: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0)


def size_to_em(size: float | None, baseline: float | None) -> str | None:
    """Map an absolute font ``size`` (points) to the nearest allowed ``em``
    bucket relative to the page ``baseline`` ("normal" body size). Returns
    ``None`` when the result is the ``1em`` baseline (so it's omitted) or when
    either input is missing/non-positive."""
    if not size or not baseline or baseline <= 0:
        return None
    ratio = size / baseline
    nearest = min(_EM_BUCKETS, key=lambda b: abs(b - ratio))
    if nearest == 1.0:
        return None
    return f"{nearest:g}em"


# ---- Color -> CSS named color -----------------------------------------------

# The CSS3 extended color keywords (name -> 0-255 RGB). Used both to validate
# author-/LLM-supplied color names and to snap an observed RGB to the nearest
# keyword. Greyscale names are included so shaded/gray text maps sensibly.
CSS_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "silver": (192, 192, 192),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "white": (255, 255, 255),
    "maroon": (128, 0, 0),
    "red": (255, 0, 0),
    "purple": (128, 0, 128),
    "fuchsia": (255, 0, 255),
    "magenta": (255, 0, 255),
    "green": (0, 128, 0),
    "lime": (0, 255, 0),
    "olive": (128, 128, 0),
    "yellow": (255, 255, 0),
    "navy": (0, 0, 128),
    "blue": (0, 0, 255),
    "teal": (0, 128, 128),
    "aqua": (0, 255, 255),
    "cyan": (0, 255, 255),
    "orange": (255, 165, 0),
    "aliceblue": (240, 248, 255),
    "antiquewhite": (250, 235, 215),
    "aquamarine": (127, 255, 212),
    "azure": (240, 255, 255),
    "beige": (245, 245, 220),
    "bisque": (255, 228, 196),
    "blanchedalmond": (255, 235, 205),
    "blueviolet": (138, 43, 226),
    "brown": (165, 42, 42),
    "burlywood": (222, 184, 135),
    "cadetblue": (95, 158, 160),
    "chartreuse": (127, 255, 0),
    "chocolate": (210, 105, 30),
    "coral": (255, 127, 80),
    "cornflowerblue": (100, 149, 237),
    "cornsilk": (255, 248, 220),
    "crimson": (220, 20, 60),
    "darkblue": (0, 0, 139),
    "darkcyan": (0, 139, 139),
    "darkgoldenrod": (184, 134, 11),
    "darkgray": (169, 169, 169),
    "darkgrey": (169, 169, 169),
    "darkgreen": (0, 100, 0),
    "darkkhaki": (189, 183, 107),
    "darkmagenta": (139, 0, 139),
    "darkolivegreen": (85, 107, 47),
    "darkorange": (255, 140, 0),
    "darkorchid": (153, 50, 204),
    "darkred": (139, 0, 0),
    "darksalmon": (233, 150, 122),
    "darkseagreen": (143, 188, 143),
    "darkslateblue": (72, 61, 139),
    "darkslategray": (47, 79, 79),
    "darkslategrey": (47, 79, 79),
    "darkturquoise": (0, 206, 209),
    "darkviolet": (148, 0, 211),
    "deeppink": (255, 20, 147),
    "deepskyblue": (0, 191, 255),
    "dimgray": (105, 105, 105),
    "dimgrey": (105, 105, 105),
    "dodgerblue": (30, 144, 255),
    "firebrick": (178, 34, 34),
    "floralwhite": (255, 250, 240),
    "forestgreen": (34, 139, 34),
    "gainsboro": (220, 220, 220),
    "ghostwhite": (248, 248, 255),
    "gold": (255, 215, 0),
    "goldenrod": (218, 165, 32),
    "greenyellow": (173, 255, 47),
    "honeydew": (240, 255, 240),
    "hotpink": (255, 105, 180),
    "indianred": (205, 92, 92),
    "indigo": (75, 0, 130),
    "ivory": (255, 255, 240),
    "khaki": (240, 230, 140),
    "lavender": (230, 230, 250),
    "lavenderblush": (255, 240, 245),
    "lawngreen": (124, 252, 0),
    "lemonchiffon": (255, 250, 205),
    "lightblue": (173, 216, 230),
    "lightcoral": (240, 128, 128),
    "lightcyan": (224, 255, 255),
    "lightgoldenrodyellow": (250, 250, 210),
    "lightgray": (211, 211, 211),
    "lightgrey": (211, 211, 211),
    "lightgreen": (144, 238, 144),
    "lightpink": (255, 182, 193),
    "lightsalmon": (255, 160, 122),
    "lightseagreen": (32, 178, 170),
    "lightskyblue": (135, 206, 250),
    "lightslategray": (119, 136, 153),
    "lightslategrey": (119, 136, 153),
    "lightsteelblue": (176, 196, 222),
    "lightyellow": (255, 255, 224),
    "limegreen": (50, 205, 50),
    "linen": (250, 240, 230),
    "mediumaquamarine": (102, 205, 170),
    "mediumblue": (0, 0, 205),
    "mediumorchid": (186, 85, 211),
    "mediumpurple": (147, 112, 219),
    "mediumseagreen": (60, 179, 113),
    "mediumslateblue": (123, 104, 238),
    "mediumspringgreen": (0, 250, 154),
    "mediumturquoise": (72, 209, 204),
    "mediumvioletred": (199, 21, 133),
    "midnightblue": (25, 25, 112),
    "mintcream": (245, 255, 250),
    "mistyrose": (255, 228, 225),
    "moccasin": (255, 228, 181),
    "navajowhite": (255, 222, 173),
    "oldlace": (253, 245, 230),
    "olivedrab": (107, 142, 35),
    "orangered": (255, 69, 0),
    "orchid": (218, 112, 214),
    "palegoldenrod": (238, 232, 170),
    "palegreen": (152, 251, 152),
    "paleturquoise": (175, 238, 238),
    "palevioletred": (219, 112, 147),
    "papayawhip": (255, 239, 213),
    "peachpuff": (255, 218, 185),
    "peru": (205, 133, 63),
    "pink": (255, 192, 203),
    "plum": (221, 160, 221),
    "powderblue": (176, 224, 230),
    "rosybrown": (188, 143, 143),
    "royalblue": (65, 105, 225),
    "saddlebrown": (139, 69, 19),
    "salmon": (250, 128, 114),
    "sandybrown": (244, 164, 96),
    "seagreen": (46, 139, 87),
    "seashell": (255, 245, 238),
    "sienna": (160, 82, 45),
    "skyblue": (135, 206, 235),
    "slateblue": (106, 90, 205),
    "slategray": (112, 128, 144),
    "slategrey": (112, 128, 144),
    "snow": (255, 250, 250),
    "springgreen": (0, 255, 127),
    "steelblue": (70, 130, 180),
    "tan": (210, 180, 140),
    "thistle": (216, 191, 216),
    "tomato": (255, 99, 71),
    "turquoise": (64, 224, 208),
    "violet": (238, 130, 238),
    "wheat": (245, 222, 179),
    "whitesmoke": (245, 245, 245),
    "yellowgreen": (154, 205, 50),
    "rebeccapurple": (102, 51, 153),
}

# Channels all at/below this are treated as default black text -> color omitted.
_NEAR_BLACK = 50


def rgb_to_named(rgb: tuple[int, int, int] | None) -> str | None:
    """Snap an observed 0-255 RGB to the nearest CSS named color. Returns
    ``None`` for near-black (the default text color, which should be omitted)
    or when ``rgb`` is missing."""
    if rgb is None:
        return None
    r, g, b = rgb
    if r <= _NEAR_BLACK and g <= _NEAR_BLACK and b <= _NEAR_BLACK:
        return None
    best: str | None = None
    best_dist = float("inf")
    for name, (nr, ng, nb) in CSS_NAMED_COLORS.items():
        # Skip the duplicate British spellings as match targets so we emit the
        # canonical form; they remain valid as *input* via CSS_NAMED_COLORS.
        if name in ("grey", "darkgrey", "darkslategrey", "dimgrey", "lightgrey", "slategrey"):
            continue
        dist = (r - nr) ** 2 + (g - ng) ** 2 + (b - nb) ** 2
        if dist < best_dist:
            best_dist = dist
            best = name
    return best


def is_named_color(value: str) -> bool:
    """Whether ``value`` is a recognized CSS named color (case-insensitive)."""
    return value.lower() in CSS_NAMED_COLORS


# ---- Assembly + validation -------------------------------------------------


def build_style(declarations: Mapping[str, str | None]) -> str:
    """Assemble a canonical ``dg:style`` string from resolved declarations.

    Properties are emitted in :data:`_PROPERTY_ORDER`; values are matched
    case-insensitively (and emitted lower-cased), and empty values or any value
    outside the allow-list are dropped. ``color`` / ``background-color`` accept
    any CSS named color. Returns ``""`` when nothing survives — callers omit the
    attribute entirely in that case.

    Default values (:data:`_DEFAULTS`) are handled by inheritance: a
    *non-inheriting* default (``text-decoration: none``) is dropped here since
    it can never override anything, but an *inheriting* default
    (``font-weight: normal`` etc.) is **kept** — it may override a non-default
    value inherited from an ancestor. The redundant copies (where the inherited
    value already is the default) are elided downstream by
    :func:`dgml_core.xml_grounding._suppress_inherited_style`.
    """
    parts: list[str] = []
    for prop in _PROPERTY_ORDER:
        val = declarations.get(prop)
        if not val:
            continue
        # Lower-case the value up front: allow-list enums, default values, and
        # emitted colors are all canonical lower-case, and this parser handles
        # free-form LLM output — so "Bold"/"UPPERCASE"/"Red" must match too.
        val = val.strip().lower()
        if not val:
            continue
        if _DEFAULTS.get(prop) == val and prop not in INHERITED_PROPERTIES:
            continue
        allowed = ALLOWED[prop]
        if allowed is None:
            if not is_named_color(val):
                continue
        elif val not in allowed:
            continue
        parts.append(f"{prop}: {val}")
    return "; ".join(parts)


def validate_style(raw: str) -> str:
    """Parse a free-form ``prop: value; …`` string (e.g. from an LLM) and
    return the canonical, allow-list-filtered form via :func:`build_style`."""
    return build_style(_parse_declarations(raw))


def merge_styles(base: str | None, extra: str | None) -> str:
    """Combine two ``dg:style`` strings into one canonical value. ``base`` wins
    on any property both set — used to let the deterministic, source-derived
    style take precedence over the LLM-inferred style while still
    picking up properties only the latter supplies (e.g. font-weight on an OCR
    heading whose all-caps ``text-transform`` was already derived)."""
    decls: dict[str, str] = {}
    for source in (extra, base):  # base applied last so it overrides extra
        if source:
            decls.update(_parse_declarations(source))
    return build_style(decls)


def _parse_declarations(raw: str) -> dict[str, str]:
    """Parse ``prop: value; …`` into a lower-cased property→value dict."""
    decls: dict[str, str] = {}
    for part in raw.split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        decls[key.strip().lower()] = value.strip()
    return decls
