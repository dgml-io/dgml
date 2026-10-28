// The parts of the service's OpenAPI document (/openapi.json) the APIs page reads, and
// helpers to turn it into a grouped list of endpoints.

export type Schema = {
  $ref?: string;
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  anyOf?: Schema[];
  items?: Schema;
  properties?: Record<string, Schema>;
  required?: string[];
  additionalProperties?: Schema | boolean;
  contentMediaType?: string;
  format?: string;
};

export interface Param {
  name: string;
  in: "path" | "query" | "header" | "cookie";
  required?: boolean;
  schema: Schema;
  description?: string;
}

export interface Operation {
  summary?: string;
  description?: string;
  operationId: string;
  parameters?: Param[];
  requestBody?: { required?: boolean; content: Record<string, { schema: Schema }> };
  responses: Record<string, { description?: string; content?: Record<string, unknown> }>;
}

export interface OpenApi {
  info: { title: string; version: string };
  paths: Record<string, Record<string, Operation>>;
  components?: { schemas?: Record<string, Schema> };
}

export interface Endpoint {
  id: string;
  method: string;
  path: string;
  op: Operation;
  group: string;
}

export const METHOD_ORDER = ["get", "post", "put", "patch", "delete"];

const GROUP_ORDER = [
  "Service",
  "Organisations",
  "Settings",
  "Docsets",
  "Schema & guidance",
  "Assignments & DGML",
  "Files",
  "Jobs",
  "Data explorer",
  "Other",
];

export async function fetchSpec(): Promise<OpenApi> {
  const resp = await fetch("/openapi.json");
  if (!resp.ok) throw new Error(`GET /openapi.json → ${resp.status}`);
  return resp.json();
}

/** The group an endpoint is listed under, from its path below `/api/orgs/{org_id}/`. */
function groupOf(path: string): string {
  if (path === "/api/health") return "Service";
  if (path === "/api/orgs" || path === "/api/orgs/{org_id}") return "Organisations";
  const rest = path.replace(/^\/api\/orgs\/\{org_id\}\//, "");
  if (rest.startsWith("settings")) return "Settings";
  if (rest.startsWith("docsets")) {
    if (rest.includes("/files/")) return "Assignments & DGML";
    if (rest.includes("/schema") || rest.includes("/guidance")) return "Schema & guidance";
    return "Docsets";
  }
  if (rest.startsWith("files")) return "Files";
  if (rest.startsWith("explore")) return "Data explorer";
  if (rest.startsWith("jobs")) return "Jobs";
  return "Other";
}

export function endpointsOf(spec: OpenApi): Endpoint[] {
  const out: Endpoint[] = [];
  for (const [path, ops] of Object.entries(spec.paths)) {
    for (const [method, op] of Object.entries(ops)) {
      if (!METHOD_ORDER.includes(method)) continue;
      out.push({ id: `${method} ${path}`, method, path, op, group: groupOf(path) });
    }
  }
  return out;
}

/** The endpoints matching `filter`, grouped in display order. */
export function groupEndpoints(
  endpoints: Endpoint[],
  filter: string,
): { name: string; items: Endpoint[] }[] {
  const q = filter.trim().toLowerCase();
  const shown = endpoints.filter(
    (e) =>
      !q ||
      e.path.toLowerCase().includes(q) ||
      e.method.includes(q) ||
      (e.op.summary ?? "").toLowerCase().includes(q),
  );
  const byPathThenMethod = (a: Endpoint, b: Endpoint) =>
    a.path.localeCompare(b.path) || METHOD_ORDER.indexOf(a.method) - METHOD_ORDER.indexOf(b.method);
  return GROUP_ORDER.map((name) => ({
    name,
    items: shown.filter((e) => e.group === name).sort(byPathThenMethod),
  })).filter((g) => g.items.length);
}

/** A path without the `/api` / `/api/orgs/{org_id}` prefix every endpoint shares. */
export const shortPath = (path: string) =>
  path.replace(/^\/api\/orgs\/\{org_id\}/, "").replace(/^\/api/, "") || "/";

/** Follow `$ref`s to the schema they name. */
export function resolve(spec: OpenApi, schema: Schema | undefined): Schema {
  let s = schema ?? {};
  for (let i = 0; s.$ref && i < 10; i++) {
    const name = s.$ref.split("/").pop()!;
    s = spec.components?.schemas?.[name] ?? {};
  }
  return s;
}

/** `T | null` → `T`; other unions are left alone. */
export function unwrapNullable(s: Schema): { schema: Schema; nullable: boolean } {
  if (!s.anyOf) return { schema: s, nullable: false };
  const rest = s.anyOf.filter((x) => x.type !== "null");
  const nullable = rest.length < s.anyOf.length;
  if (rest.length === 1) return { schema: { ...rest[0], title: s.title, default: s.default }, nullable };
  return { schema: s, nullable };
}

export function typeLabel(s: Schema): string {
  const { schema, nullable } = unwrapNullable(s);
  let t = schema.type ?? (schema.anyOf ? "any" : "object");
  if (t === "array") t = `${typeLabel(schema.items ?? {})}[]`;
  if (schema.contentMediaType) t = "file";
  return nullable ? `${t} | null` : t;
}

/** The kind of input a value of this schema gets. */
export type FieldKind = "bool" | "number" | "string" | "list" | "map" | "files" | "json";

export function kindOf(s: Schema): FieldKind {
  const { schema } = unwrapNullable(s);
  if (schema.type === "boolean") return "bool";
  if (schema.type === "integer" || schema.type === "number") return "number";
  if (schema.type === "string") return "string";
  if (schema.type === "array") {
    const item = unwrapNullable(schema.items ?? {}).schema;
    if (item.contentMediaType || item.format === "binary") return "files";
    if (item.type === "string") return "list";
  }
  if (schema.type === "object" && schema.additionalProperties && !schema.properties) return "map";
  return "json";
}

/** One property of a request body. */
export interface BodyField {
  name: string;
  schema: Schema;
  required: boolean;
}

export function bodyFields(spec: OpenApi, schema: Schema): BodyField[] {
  const s = resolve(spec, schema);
  const req = new Set(s.required ?? []);
  return Object.entries(s.properties ?? {}).map(([name, fs]) => ({
    name,
    schema: resolve(spec, fs),
    required: req.has(name),
  }));
}
