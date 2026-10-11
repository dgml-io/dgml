// One endpoint of the APIs page: its parameters and body, Send, and the response.

import { useEffect, useMemo, useState } from "react";
import type { Job } from "../../api";
import { ErrorNote, Spinner, useLoad } from "../../components/ui";
import { useApp } from "../../state";
import { NO_S3, listS3, defaultsFor, suggestions } from "./catalog";
import type { Catalog, SuggestionContext } from "./catalog";
import { fieldValue, initialField } from "./form";
import type { FieldState } from "./form";
import { BodyFieldInput, ParamInput } from "./inputs";
import { bodyFields, kindOf } from "./openapi";
import type { Endpoint, OpenApi } from "./openapi";
import { CurlLine, ResponseView } from "./output";
import type { ApiResponse } from "./output";

export function MethodBadge({ method }: { method: string }) {
  return <span className={`method method-${method}`}>{method.toUpperCase()}</span>;
}

export function EndpointConsole({
  spec,
  endpoint,
  catalog,
  catalogLoading,
  onTargetOrg,
}: {
  spec: OpenApi;
  endpoint: Endpoint;
  /** The page's catalog, for the organisation last passed to `onTargetOrg`. */
  catalog: Catalog | null;
  catalogLoading: boolean;
  onTargetOrg: (org: string) => void;
}) {
  const { org, orgs, bump, track, version } = useApp();
  const [values, setValues] = useState<Record<string, string>>({});
  const targetOrg = values.org_id || org?.id || "";
  useEffect(() => onTargetOrg(targetOrg), [targetOrg, onTargetOrg]);
  const cat = catalog?.org === targetOrg ? catalog : null;

  // S3 keys are listed only for the endpoints that take one.
  const params = endpoint.op.parameters ?? [];
  const wantsS3 = params.some((p) => p.in === "query" && (p.name === "key" || p.name === "prefix"));
  const s3 = useLoad(() => (wantsS3 && targetOrg ? listS3(targetOrg) : Promise.resolve(NO_S3)), [
    wantsS3,
    targetOrg,
    version,
  ]);

  // The organisation the inputs were last seeded for. Seeding waits for that
  // organisation's data, so the first choices are real ones; picking another
  // organisation re-seeds everything that belongs to it.
  const [seededFor, setSeededFor] = useState<string | null>(null);
  useEffect(() => {
    if (!cat || seededFor === cat.org) return;
    setValues(defaultsFor(endpoint, cat));
    setSeededFor(cat.org);
  }, [cat, seededFor, endpoint]);

  const pathParams = params
    .filter((p) => p.in === "path")
    .sort((a, b) => endpoint.path.indexOf(`{${a.name}}`) - endpoint.path.indexOf(`{${b.name}}`));
  const queryParams = params.filter((p) => p.in === "query");

  const contentType = endpoint.op.requestBody ? Object.keys(endpoint.op.requestBody.content)[0] : null;
  const isMultipart = contentType === "multipart/form-data";
  const fields = useMemo(
    () => (contentType ? bodyFields(spec, endpoint.op.requestBody!.content[contentType].schema) : []),
    [spec, endpoint, contentType],
  );
  const [body, setBody] = useState<Record<string, FieldState>>({});
  const [bodyMode, setBodyMode] = useState<"form" | "raw">("form");
  const [raw, setRaw] = useState("");
  useEffect(() => {
    if (seededFor === null) return;
    setBody(Object.fromEntries(fields.map((f) => [f.name, initialField(f, cat, values)])));
    // Seed per endpoint and organisation; later edits are the user's.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seededFor, fields]);

  const setValue = (name: string, value: string) =>
    setValues((prev) => {
      const next = { ...prev, [name]: value };
      // A new file invalidates the page chosen for the old one.
      if (name === "file_id" && endpoint.path.includes("{page}")) next.page = "1";
      return next;
    });

  const setField = (name: string, patch: Partial<FieldState>) =>
    setBody((prev) => ({ ...prev, [name]: { ...prev[name], ...patch, include: patch.include ?? true } }));

  const ctx: SuggestionContext = { orgs: orgs ?? [], cat, s3: s3.data ?? NO_S3, values, endpoint };

  // ---- build the request ----
  const missing = [...pathParams, ...queryParams.filter((p) => p.required)]
    .filter((p) => !values[p.name])
    .map((p) => p.name);
  let url = endpoint.path.replace(/\{(\w+)\}/g, (_, n: string) => encodeURIComponent(values[n] ?? `{${n}}`));
  const qs = new URLSearchParams();
  for (const p of queryParams) {
    const v = values[p.name];
    if (v || (p.name === "prefix" && v === "")) qs.set(p.name, v);
  }
  if (qs.toString()) url += `?${qs}`;

  let jsonBody: unknown = undefined;
  let bodyError: string | null = null;
  if (contentType && !isMultipart) {
    try {
      jsonBody =
        bodyMode === "raw"
          ? raw.trim()
            ? JSON.parse(raw)
            : undefined
          : Object.fromEntries(
              fields.filter((f) => body[f.name]?.include).map((f) => [f.name, fieldValue(f, body[f.name])]),
            );
    } catch (e) {
      bodyError = `The body is not valid JSON: ${(e as Error).message}`;
    }
  }
  const multipartFields: [string, FieldState][] = isMultipart
    ? fields.filter((f) => body[f.name]?.include).map((f) => [f.name, body[f.name]])
    : [];

  const switchBodyMode = (mode: "form" | "raw") => {
    if (mode === "raw") setRaw(JSON.stringify(jsonBody ?? {}, null, 2));
    setBodyMode(mode);
  };

  // ---- send ----
  const [sending, setSending] = useState(false);
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const [sendError, setSendError] = useState<unknown>(null);
  useEffect(() => () => void (response?.blobUrl && URL.revokeObjectURL(response.blobUrl)), [response]);

  const send = async () => {
    if (endpoint.method === "delete" && !window.confirm(`DELETE ${url}?\n\nThis really deletes it.`)) return;
    setSending(true);
    setSendError(null);
    const init: RequestInit = { method: endpoint.method.toUpperCase(), headers: {} };
    if (isMultipart) {
      const form = new FormData();
      for (const f of fields) {
        const st = body[f.name];
        if (!st?.include) continue;
        if (kindOf(f.schema) === "files") st.files.forEach((file) => form.append(f.name, file, file.name));
        else form.append(f.name, st.value);
      }
      init.body = form;
    } else if (jsonBody !== undefined) {
      init.body = JSON.stringify(jsonBody);
      (init.headers as Record<string, string>)["Content-Type"] = "application/json";
    }
    try {
      const out = await call(url, init);
      setResponse(out);
      if (out.status >= 200 && out.status < 300 && endpoint.method !== "get") {
        // Jobs the call queued show up in the Activity panel; pages (and the
        // catalog) refetch on the bump.
        const jobs = jobsIn(out.json);
        if (jobs.length && targetOrg === org?.id) track(jobs);
        bump();
      }
    } catch (e) {
      setSendError(e);
      setResponse(null);
    } finally {
      setSending(false);
    }
  };

  const okStatuses = Object.keys(endpoint.op.responses).filter((s) => s.startsWith("2"));
  const description = endpoint.op.description?.trim();

  return (
    <div className="stack">
      <div className="card stack">
        <div className="row wrap api-head">
          <MethodBadge method={endpoint.method} />
          <code className="api-path">{endpoint.path}</code>
        </div>
        <div>
          <div className="strong">{endpoint.op.summary}</div>
          {description && <p className="muted api-desc">{description}</p>}
          <div className="small muted">
            Success: {okStatuses.join(", ") || "—"} · operationId <span className="mono">{endpoint.op.operationId}</span>
          </div>
        </div>
      </div>

      {(pathParams.length > 0 || queryParams.length > 0) && (
        <div className="card stack">
          <div className="row between">
            <h3>Parameters</h3>
            {(catalogLoading || s3.loading) && <Spinner label="Loading choices…" />}
          </div>
          {[...pathParams, ...queryParams].map((p) => (
            <ParamInput
              key={`${p.in}:${p.name}`}
              name={p.name}
              location={p.in}
              required={!!p.required}
              schema={p.schema}
              value={values[p.name] ?? ""}
              options={suggestions(p.name, p.in, ctx)}
              onChange={(v) => setValue(p.name, v)}
            />
          ))}
        </div>
      )}

      {contentType && (
        <div className="card stack">
          <div className="row between">
            <h3>
              Body <span className="muted small mono">{contentType}</span>
            </h3>
            {!isMultipart && (
              <div className="tabs tabs-compact">
                <button className={`tab${bodyMode === "form" ? " tab-active" : ""}`} onClick={() => switchBodyMode("form")}>
                  Form
                </button>
                <button className={`tab${bodyMode === "raw" ? " tab-active" : ""}`} onClick={() => switchBodyMode("raw")}>
                  JSON
                </button>
              </div>
            )}
          </div>
          {bodyMode === "raw" && !isMultipart ? (
            <textarea
              className="code"
              rows={Math.min(24, Math.max(6, raw.split("\n").length + 1))}
              value={raw}
              onChange={(e) => setRaw(e.target.value)}
            />
          ) : (
            fields.map((f) =>
              body[f.name] ? (
                <BodyFieldInput
                  key={f.name}
                  field={f}
                  state={body[f.name]}
                  options={suggestions(f.name, "body", ctx)}
                  onChange={(patch) => setField(f.name, patch)}
                />
              ) : null,
            )
          )}
        </div>
      )}

      <div className="card stack">
        <div className="row wrap">
          <button
            className={`btn ${endpoint.method === "delete" ? "btn-danger" : "btn-primary"}`}
            disabled={sending || missing.length > 0 || !!bodyError}
            onClick={send}
          >
            {sending ? <Spinner /> : null} Send
          </button>
          <code className="api-url">
            {endpoint.method.toUpperCase()} {url}
          </code>
        </div>
        {missing.length > 0 && <div className="small warn-text">Choose: {missing.join(", ")}</div>}
        {bodyError && <div className="note note-error">{bodyError}</div>}
        <CurlLine method={endpoint.method} url={url} json={jsonBody} multipart={isMultipart ? multipartFields : null} />
        <ErrorNote error={sendError} />
      </div>

      {response && <ResponseView response={response} />}
    </div>
  );
}

/** Make the request and read the response for display: binary bodies as a blob URL. */
async function call(url: string, init: RequestInit): Promise<ApiResponse> {
  const started = performance.now();
  const resp = await fetch(url, init);
  const ms = Math.round(performance.now() - started);
  const ct = resp.headers.get("content-type") ?? "";
  const disposition = resp.headers.get("content-disposition") ?? "";
  const filename = /filename="?([^";]+)"?/.exec(disposition)?.[1];
  const out: ApiResponse = { status: resp.status, statusText: resp.statusText, ms, contentType: ct, size: 0, filename };
  if (ct.startsWith("image/") || ct === "application/pdf" || ct.startsWith("application/octet-stream")) {
    const blob = await resp.blob();
    out.size = blob.size;
    out.blobUrl = URL.createObjectURL(blob);
    return out;
  }
  const text = await resp.text();
  out.size = new Blob([text]).size;
  out.text = text;
  if (ct.includes("json") && text) {
    try {
      out.json = JSON.parse(text);
    } catch {
      /* show as text */
    }
  }
  return out;
}

/** The jobs in a response body: a job, a list of jobs, or `{job: …}` (assign). */
function jobsIn(json: unknown): Job[] {
  const items = Array.isArray(json) ? json : [json, (json as { job?: unknown } | undefined)?.job];
  return items.filter((j): j is Job => !!j && typeof j === "object" && "kind" in j && "status" in j);
}
