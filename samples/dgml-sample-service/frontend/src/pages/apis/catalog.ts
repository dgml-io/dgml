// What an organisation actually has (docsets, files, jobs, tables, settings, S3 keys),
// turned into suggested values for the APIs page's inputs, so a call can be tried
// without copying ids around.

import { api } from "../../api";
import type { DocSet, FileEntry, Job, Org, Settings, TableInfo } from "../../api";
import { formatDate } from "../../components/ui";
import type { Endpoint } from "./openapi";

export interface Option {
  value: string;
  label: string;
  hint?: string;
}

export interface Catalog {
  /** The organisation this catalog describes. */
  org: string;
  docsets: DocSet[];
  files: FileEntry[];
  jobs: Job[];
  tables: TableInfo[];
  settings: Settings | null;
}

/** S3 keys and folders under the workspace prefix, relative to it. */
export interface S3Choices {
  keys: string[];
  folders: string[];
}

export const NO_S3: S3Choices = { keys: [], folders: [] };

const settle = <T,>(p: Promise<T>, fallback: T) => p.catch(() => fallback);

/** Everything but S3, which is listed only for the endpoints that take a key or prefix. */
export async function loadCatalog(org: string): Promise<Catalog> {
  const [docsets, files, jobs, tables, settings] = await Promise.all([
    settle(api.docsets(org), []),
    settle(api.files(org), []),
    settle(api.jobs(org, 50), []),
    settle(api.tables(org), []),
    settle<Settings | null>(api.settings(org), null),
  ]);
  return { org, docsets, files, jobs, tables, settings };
}

/** Every key under the workspace prefix (one recursive listing), and the folders they sit in. */
export async function listS3(org: string): Promise<S3Choices> {
  const keys = (await api.s3(org, "", true)).objects.map((o) => o.key);
  const folders = new Set<string>();
  for (const key of keys) {
    const parts = key.split("/");
    for (let i = 1; i < parts.length; i++) folders.add(parts.slice(0, i).join("/"));
  }
  return { keys, folders: [...folders].sort() };
}

export const CLASSIFY_MODES = ["existing", "existing-or-new"];

const plain = (values: string[]): Option[] => values.map((v) => ({ value: v, label: v }));

/** What a suggestion is computed from. `values` holds what the form currently has, so
 * dependent choices narrow: the files of the chosen docset, the pages of the chosen file. */
export interface SuggestionContext {
  orgs: Org[];
  cat: Catalog | null;
  s3: S3Choices;
  values: Record<string, string>;
  endpoint: Endpoint;
}

/** Suggestions for a parameter or body field, by name; `null` when there are none. */
export function suggestions(name: string, location: string, ctx: SuggestionContext): Option[] | null {
  const { orgs, cat, s3, values, endpoint } = ctx;
  if (name === "org_id") return orgs.map((o) => ({ value: o.id, label: o.name, hint: o.slug }));
  if (!cat) return null;
  switch (name) {
    case "docset_id":
      return cat.docsets.map((d) => ({
        value: d.id,
        label: d.name,
        hint: `${d.file_ids.length} file(s)${d.has_schema ? " · schema" : ""}`,
      }));
    case "file_id":
    case "file_ids":
      return fileOptions(cat, values, endpoint);
    case "job_id":
      return cat.jobs.map((j) => ({
        value: j.id,
        label: j.label ?? j.kind,
        hint: `${j.status} · ${formatDate(j.created_at)}`,
      }));
    case "table":
      return cat.tables.map((t) => ({ value: t.name, label: t.name, hint: `${t.group} · ${t.rows} row(s)` }));
    case "page": {
      const n = cat.files.find((f) => f.id === values.file_id)?.page_count ?? 0;
      return Array.from({ length: n }, (_, i) => ({ value: String(i + 1), label: `Page ${i + 1}` }));
    }
    case "key":
      return location === "query" ? plain(s3.keys) : null;
    case "prefix":
      return [{ value: "", label: "(workspace root)" }, ...plain(s3.folders)];
    case "mode":
      return plain(CLASSIFY_MODES);
    case "classify":
      return plain(["none", ...CLASSIFY_MODES]);
    case "llm_family":
      return plain(cat.settings?.options.llm_families ?? []);
    case "text_mode":
      return plain(cat.settings?.options.text_modes ?? []);
    case "ocr_provider":
      return plain(cat.settings?.options.ocr_providers ?? []);
    case "s3_bucket":
    case "s3_endpoint_url":
    case "s3_region":
    case "blob_folder": {
      const current = cat.settings?.[name];
      return current ? [{ value: current, label: current, hint: "current" }] : null;
    }
    default:
      return null;
  }
}

/** The organisation's files. On an assignment route, files outside the docset come first
 * for POST (assign) and files inside it for everything else (unassign, extract, read DGML). */
function fileOptions(cat: Catalog, values: Record<string, string>, endpoint: Endpoint): Option[] {
  const docset = cat.docsets.find((d) => d.id === values.docset_id);
  const scoped = endpoint.path.includes("{docset_id}") && docset !== undefined;
  const inDocset = new Set(docset?.file_ids ?? []);
  const wantInside = !(endpoint.method === "post" && /\/files\/\{file_id\}$/.test(endpoint.path));
  const options = cat.files.map((f) => ({
    value: f.id,
    label: f.original_filename,
    hint: [
      f.page_count != null ? `${f.page_count} page(s)` : null,
      scoped ? (inDocset.has(f.id) ? "in docset" : "not in docset") : null,
    ]
      .filter(Boolean)
      .join(" · "),
    rank: scoped && inDocset.has(f.id) !== wantInside ? 1 : 0,
  }));
  return options.sort((a, b) => a.rank - b.rank).map(({ rank: _rank, ...o }) => o);
}

/** Seed values for an endpoint's path and query parameters. */
export function defaultsFor(endpoint: Endpoint, cat: Catalog): Record<string, string> {
  const values: Record<string, string> = {};
  for (const p of endpoint.op.parameters ?? []) {
    if (p.name === "org_id") values.org_id = cat.org;
    else if (p.schema.default !== undefined && p.schema.default !== null)
      values[p.name] = String(p.schema.default);
  }
  const has = (param: string) => endpoint.path.includes(`{${param}}`);
  const docset = cat.docsets[0];
  if (has("docset_id") && docset) values.docset_id = docset.id;
  if (has("file_id")) {
    const preferred = docset?.file_ids.find((id) => cat.files.some((f) => f.id === id));
    const value = preferred ?? cat.files[0]?.id;
    if (value) values.file_id = value;
  }
  if (has("page") && values.file_id) values.page = "1";
  if (has("job_id") && cat.jobs[0]) values.job_id = cat.jobs[0].id;
  if (has("table") && cat.tables[0]) values.table = cat.tables[0].name;
  return values;
}
