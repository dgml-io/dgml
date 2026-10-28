// Request-body form state: one FieldState per body field, and the value it sends.

import type { Catalog } from "./catalog";
import { kindOf, unwrapNullable } from "./openapi";
import type { BodyField } from "./openapi";

/** Form state for one body field: whether it is sent, and its value as typed. */
export interface FieldState {
  include: boolean;
  value: string;
  list: string[];
  map: [string, string][];
  files: File[];
}

export function initialField(f: BodyField, cat: Catalog | null, values: Record<string, string>): FieldState {
  const def = f.schema.default;
  const state: FieldState = { include: f.required, value: "", list: [], map: [], files: [] };
  const kind = kindOf(f.schema);
  if (kind === "bool") state.value = String(def ?? false);
  else if (kind === "number") state.value = def != null ? String(def) : "";
  else if (kind === "string") state.value = typeof def === "string" ? def : "";
  else if (kind === "json") state.value = def !== undefined ? JSON.stringify(def, null, 2) : "";
  // Sensible starting points where the data offers one.
  if (f.name === "file_ids") {
    const docset = cat?.docsets.find((d) => d.id === values.docset_id);
    state.list = docset?.file_ids.slice(0, 3) ?? [];
    state.include = state.list.length > 0;
  }
  if (f.name === "secrets") {
    state.map = Object.keys(cat?.settings?.secrets ?? {}).map((k) => [k, ""]);
  }
  // Fields with a default are sent by default, so the request shows what the server assumes.
  if (def !== undefined && def !== null && kind !== "json") state.include = true;
  return state;
}

/** The JSON value a field sends. Throws on an unparseable JSON field. */
export function fieldValue(f: BodyField, st: FieldState): unknown {
  switch (kindOf(f.schema)) {
    case "bool":
      return st.value === "true";
    case "number":
      return st.value === "" ? null : Number(st.value);
    case "string":
      return st.value === "" && unwrapNullable(f.schema).nullable ? null : st.value;
    case "list":
      return st.list;
    case "map":
      return Object.fromEntries(st.map.filter(([k]) => k.trim()));
    case "json":
      return st.value.trim() ? JSON.parse(st.value) : null;
    case "files":
      return st.files;
  }
}
