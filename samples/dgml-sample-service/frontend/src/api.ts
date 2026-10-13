// Typed client for the sample service's REST API. Every call is scoped to an
// organisation (= one DGML workspace) except the organisation list itself.

export interface Org {
  id: string;
  name: string;
  slug: string;
  created_at: string;
}

export interface SecretStatus {
  set: boolean;
  updated_at: string | null;
}

export interface Settings {
  configured: boolean;
  llm_family: string;
  text_mode: string;
  ocr_provider: string | null;
  s3_endpoint_url: string | null;
  s3_region: string | null;
  s3_bucket: string;
  blob_folder: string;
  updated_at: string;
  secrets: Record<string, SecretStatus>;
  storage_locked: boolean;
  storage_path: string;
  options: { llm_families: string[]; text_modes: string[]; ocr_providers: string[] };
}

export interface SettingsUpdate {
  llm_family?: string;
  text_mode?: string;
  ocr_provider?: string | null;
  s3_endpoint_url?: string | null;
  s3_region?: string | null;
  s3_bucket?: string;
  blob_folder?: string;
  secrets?: Record<string, string>;
  create_bucket?: boolean;
}

export interface DocSet {
  id: string;
  name: string;
  description: string;
  key_questions: string[];
  file_ids: string[];
  has_schema: boolean;
  has_guidance: boolean;
}

export interface FileRecord {
  id: string;
  original_path: string;
  original_filename: string;
  sha256: string;
  added_at: string;
  page_count: number | null;
  text_mode: string | null;
  page_image_dpi: number | null;
  page_image_renderer: string | null;
  pdf_converter: string | null;
}

export interface DocSetFile extends Partial<FileRecord> {
  id: string;
  missing?: boolean;
  has_dgml: boolean;
}

export interface DocSetDetail extends DocSet {
  files: DocSetFile[];
}

export interface FileEntry extends FileRecord {
  docsets: { id: string; name: string }[];
}

export interface FileDetail extends FileEntry {
  jobs: Job[];
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface Job {
  id: string;
  kind: "add_file" | "classify" | "extract" | "generate_schema";
  status: JobStatus;
  file_id: string | null;
  docset_id: string | null;
  label: string | null;
  params: Record<string, unknown>;
  result: Record<string, any> | null;
  error: string | null;
  error_code: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface Location {
  page_number: number;
  bounding_box: [number, number, number, number];
}

/** A leaf of the extracted values tree: `{text, value?, locations?}`. */
export interface ValueLeaf {
  text: string;
  value?: string;
  locations?: Location[];
  computed?: boolean;
  derived_from?: string[];
}

export type ValueNode = ValueLeaf | ValueNode[] | { [tag: string]: ValueNode };

export interface PairDgml {
  docset_id: string;
  file_id: string;
  xml_key: string | null;
  xml: string | null;
  has_extraction: boolean;
  has_document_tree: boolean;
  values: Record<string, ValueNode> | null;
}

export interface S3Listing {
  bucket: string;
  endpoint_url: string | null;
  root: string;
  prefix: string;
  folders: { name: string; prefix: string }[];
  objects: { key: string; name: string; size: number; last_modified: string }[];
  truncated: boolean;
}

export interface TableColumn {
  name: string;
  type: string;
  primary_key: boolean;
  nullable: boolean;
}

export interface TableInfo {
  name: string;
  group: string;
  rows: number;
  columns: TableColumn[];
}

export interface TablePage {
  name: string;
  group: string;
  columns: TableColumn[];
  rows: Record<string, unknown>[];
  total: number;
  limit: number;
  offset: number;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: {} };
  if (body instanceof FormData) {
    init.body = body;
  } else if (body !== undefined) {
    init.body = JSON.stringify(body);
    (init.headers as Record<string, string>)["Content-Type"] = "application/json";
  }
  const resp = await fetch(`/api${path}`, init);
  if (resp.status === 204) return undefined as T;
  const text = await resp.text();
  const data = text ? JSON.parse(text) : undefined;
  if (!resp.ok) {
    const err = data?.error;
    const detail = Array.isArray(data?.detail)
      ? data.detail.map((d: any) => `${d.loc?.slice(1).join(".")}: ${d.msg}`).join("; ")
      : undefined;
    throw new ApiError(resp.status, err?.code ?? "HTTP_ERROR", err?.message ?? detail ?? text);
  }
  return data as T;
}

const get = <T>(path: string) => request<T>("GET", path);

export const api = {
  orgs: () => get<Org[]>("/orgs"),
  createOrg: (name: string, slug: string) => request<Org>("POST", "/orgs", { name, slug }),
  deleteOrg: (org: string) => request<void>("DELETE", `/orgs/${org}`),

  settings: (org: string) => get<Settings>(`/orgs/${org}/settings`),
  saveSettings: (org: string, update: SettingsUpdate) =>
    request<Settings>("PUT", `/orgs/${org}/settings`, update),
  testStorage: (org: string, createBucket: boolean) =>
    request<{ ok: boolean; bucket: string; bucket_created: boolean }>(
      "POST",
      `/orgs/${org}/settings/test-storage?create_bucket=${createBucket}`,
    ),

  docsets: (org: string) => get<DocSet[]>(`/orgs/${org}/docsets`),
  docset: (org: string, id: string) => get<DocSetDetail>(`/orgs/${org}/docsets/${id}`),
  createDocset: (org: string, body: { name: string; description: string; key_questions: string[] }) =>
    request<DocSet>("POST", `/orgs/${org}/docsets`, body),
  updateDocset: (
    org: string,
    id: string,
    body: { name?: string; description?: string; key_questions?: string[] },
  ) => request<DocSet>("PATCH", `/orgs/${org}/docsets/${id}`, body),
  deleteDocset: (org: string, id: string) => request<void>("DELETE", `/orgs/${org}/docsets/${id}`),

  schema: (org: string, id: string) => get<{ schema: string }>(`/orgs/${org}/docsets/${id}/schema`),
  setSchema: (org: string, id: string, text: string) =>
    request<{ schema: string }>("PUT", `/orgs/${org}/docsets/${id}/schema`, { text }),
  clearSchema: (org: string, id: string) =>
    request<void>("DELETE", `/orgs/${org}/docsets/${id}/schema`),
  generateSchema: (org: string, id: string, fileIds?: string[]) =>
    request<Job>("POST", `/orgs/${org}/docsets/${id}/schema/generate`, { file_ids: fileIds }),
  guidance: (org: string, id: string) =>
    get<{ guidance: string }>(`/orgs/${org}/docsets/${id}/guidance`),
  setGuidance: (org: string, id: string, text: string) =>
    request<{ guidance: string }>("PUT", `/orgs/${org}/docsets/${id}/guidance`, { text }),
  clearGuidance: (org: string, id: string) =>
    request<void>("DELETE", `/orgs/${org}/docsets/${id}/guidance`),

  assign: (org: string, docset: string, file: string, extract: boolean) =>
    request<{ assigned: boolean; job: Job | null }>(
      "POST",
      `/orgs/${org}/docsets/${docset}/files/${file}`,
      { extract },
    ),
  unassign: (org: string, docset: string, file: string) =>
    request<void>("DELETE", `/orgs/${org}/docsets/${docset}/files/${file}`),
  extract: (org: string, docset: string, file: string) =>
    request<Job>("POST", `/orgs/${org}/docsets/${docset}/files/${file}/extract`),
  dgml: (org: string, docset: string, file: string) =>
    get<PairDgml>(`/orgs/${org}/docsets/${docset}/files/${file}/dgml`),
  dgmlDownloadUrl: (org: string, docset: string, file: string) =>
    `/api/orgs/${org}/docsets/${docset}/files/${file}/dgml.xml`,

  files: (org: string) => get<FileEntry[]>(`/orgs/${org}/files`),
  file: (org: string, id: string) => get<FileDetail>(`/orgs/${org}/files/${id}`),
  upload: (org: string, files: File[], classify: string, textMode?: string) => {
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    form.append("classify", classify);
    if (textMode) form.append("text_mode", textMode);
    return request<Job[]>("POST", `/orgs/${org}/files`, form);
  },
  deleteFile: (org: string, id: string) => request<void>("DELETE", `/orgs/${org}/files/${id}`),
  classify: (org: string, id: string, mode: string, extract: boolean) =>
    request<Job>("POST", `/orgs/${org}/files/${id}/classify`, { mode, extract }),
  pageUrl: (org: string, id: string, page: number) => `/api/orgs/${org}/files/${id}/pages/${page}`,
  sourceUrl: (org: string, id: string) => `/api/orgs/${org}/files/${id}/source`,

  s3: (org: string, prefix: string, recursive = false) =>
    get<S3Listing>(
      `/orgs/${org}/explore/s3?prefix=${encodeURIComponent(prefix)}${recursive ? "&recursive=true" : ""}`,
    ),
  s3ObjectUrl: (org: string, key: string) =>
    `/api/orgs/${org}/explore/s3/object?key=${encodeURIComponent(key)}`,
  tables: (org: string) => get<TableInfo[]>(`/orgs/${org}/explore/db`),
  table: (org: string, name: string, limit: number, offset: number) =>
    get<TablePage>(`/orgs/${org}/explore/db/${name}?limit=${limit}&offset=${offset}`),

  jobs: (org: string, limit = 30) => get<Job[]>(`/orgs/${org}/jobs?limit=${limit}`),
};
