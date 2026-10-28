// Data: read-only views of what the service stores for this organisation — the S3
// objects under the workspace prefix, and the organisation's rows in every Postgres
// table (DGML's DocStore tables and the service's own).

import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import type { S3Listing, TableColumn } from "../api";
import { Empty, ErrorNote, Spinner, formatDate, useLoad } from "../components/ui";
import { useApp, useOrgId } from "../state";

export function DataPage() {
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") === "db" ? "db" : "s3";
  return (
    <div className="page page-wide">
      <div className="page-head">
        <div>
          <h1>Data</h1>
          <p className="muted">
            What this organisation's workspace has actually stored — read-only, and scoped to this
            organisation.
          </p>
        </div>
      </div>
      <div className="tabs">
        <button className={`tab${tab === "s3" ? " tab-active" : ""}`} onClick={() => setParams({ tab: "s3" })}>
          S3 bucket
        </button>
        <button className={`tab${tab === "db" ? " tab-active" : ""}`} onClick={() => setParams({ tab: "db" })}>
          Postgres tables
        </button>
      </div>
      {tab === "s3" ? <S3Browser /> : <TableBrowser />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// S3
// ---------------------------------------------------------------------------

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function S3Browser() {
  const org = useOrgId();
  const { version } = useApp();
  const [params, setParams] = useSearchParams();
  const prefix = params.get("prefix") ?? "";
  const selected = params.get("key");
  const listing = useLoad(() => api.s3(org, prefix), [org, prefix, version]);

  const go = (next: Record<string, string>) => setParams({ tab: "s3", ...next });
  const crumbs = prefix ? prefix.split("/") : [];

  return (
    <div className="explorer">
      <div className="card explorer-list">
        <div className="crumbs">
          <button className="btn-link mono" onClick={() => go({})}>
            s3://{listing.data?.bucket ?? "…"}/{listing.data?.root ?? ""}
          </button>
          {crumbs.map((part, i) => (
            <span key={i}>
              <button className="btn-link mono" onClick={() => go({ prefix: crumbs.slice(0, i + 1).join("/") })}>
                {part}
              </button>
              /
            </span>
          ))}
        </div>
        <ErrorNote error={listing.error} />
        {!listing.data ? (
          <Spinner label="Listing…" />
        ) : (
          <S3Table listing={listing.data} selected={selected} onOpen={go} />
        )}
      </div>
      <div className="card explorer-preview">
        {selected ? <ObjectPreview objectKey={selected} /> : <Empty title="Select an object">to preview it here</Empty>}
      </div>
    </div>
  );
}

function S3Table({
  listing,
  selected,
  onOpen,
}: {
  listing: S3Listing;
  selected: string | null;
  onOpen: (next: Record<string, string>) => void;
}) {
  if (!listing.folders.length && !listing.objects.length) {
    return <Empty title="Empty">{listing.prefix ? "Nothing under this folder." : "Upload a file to fill the workspace."}</Empty>;
  }
  const parent = listing.prefix.split("/").slice(0, -1).join("/");
  return (
    <table className="table">
      <thead>
        <tr>
          <th>Name</th>
          <th>Size</th>
          <th>Modified</th>
        </tr>
      </thead>
      <tbody>
        {listing.prefix && (
          <tr className="clickable" onClick={() => onOpen(parent ? { prefix: parent } : {})}>
            <td colSpan={3} className="mono">
              ↑ ..
            </td>
          </tr>
        )}
        {listing.folders.map((f) => (
          <tr key={f.prefix} className="clickable" onClick={() => onOpen({ prefix: f.prefix })}>
            <td className="mono">📁 {f.name}/</td>
            <td />
            <td />
          </tr>
        ))}
        {listing.objects.map((o) => (
          <tr
            key={o.key}
            className={`clickable${selected === o.key ? " row-selected" : ""}`}
            onClick={() => onOpen({ prefix: listing.prefix, key: o.key })}
          >
            <td className="mono">{o.name}</td>
            <td className="small">{formatSize(o.size)}</td>
            <td className="small">{formatDate(o.last_modified)}</td>
          </tr>
        ))}
        {listing.truncated && (
          <tr>
            <td colSpan={3} className="muted small">
              Showing the first 1000 entries.
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

const MAX_TEXT_PREVIEW = 2 * 1024 * 1024;

function ObjectPreview({ objectKey }: { objectKey: string }) {
  const org = useOrgId();
  const url = api.s3ObjectUrl(org, objectKey);
  const [state, setState] = useState<
    { kind: "loading" } | { kind: "image" | "pdf" | "binary"; type: string; size: number } | { kind: "text"; type: string; text: string; size: number } | { kind: "error"; error: unknown }
  >({ kind: "loading" });

  useEffect(() => {
    let live = true;
    setState({ kind: "loading" });
    fetch(url)
      .then(async (resp) => {
        if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
        const type = resp.headers.get("content-type") ?? "";
        const blob = await resp.blob();
        if (!live) return;
        if (type.startsWith("image/")) return setState({ kind: "image", type, size: blob.size });
        if (type === "application/pdf") return setState({ kind: "pdf", type, size: blob.size });
        const textual = type.startsWith("text/") || type.includes("json") || type.includes("xml");
        if (!textual || blob.size > MAX_TEXT_PREVIEW) return setState({ kind: "binary", type, size: blob.size });
        let text = await blob.text();
        if (type.includes("json")) {
          try {
            text = JSON.stringify(JSON.parse(text), null, 2);
          } catch {
            /* show as is */
          }
        }
        if (live) setState({ kind: "text", type, text, size: blob.size });
      })
      .catch((error) => live && setState({ kind: "error", error }));
    return () => {
      live = false;
    };
  }, [url]);

  return (
    <div className="stack">
      <div className="row between">
        <span className="mono small preview-key">{objectKey}</span>
        <a className="btn" href={url} target="_blank" rel="noreferrer">
          Open
        </a>
      </div>
      {"type" in state && (
        <div className="muted small">
          {state.type} · {formatSize(state.size)}
        </div>
      )}
      {state.kind === "loading" && <Spinner label="Loading…" />}
      {state.kind === "error" && <ErrorNote error={state.error} />}
      {state.kind === "image" && <img className="preview-image" src={url} alt={objectKey} />}
      {state.kind === "pdf" && <iframe className="preview-pdf" src={url} title={objectKey} />}
      {state.kind === "text" && <pre className="preview-text">{state.text}</pre>}
      {state.kind === "binary" && <div className="muted">No inline preview for this object — use Open.</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Postgres
// ---------------------------------------------------------------------------

const PAGE_SIZE = 50;

function TableBrowser() {
  const org = useOrgId();
  const { version } = useApp();
  const [params, setParams] = useSearchParams();
  const tables = useLoad(() => api.tables(org), [org, version]);
  const name = params.get("table") ?? tables.data?.[0]?.name ?? null;
  const offset = Number(params.get("offset") ?? 0) || 0;
  const page = useLoad(
    () => (name ? api.table(org, name, PAGE_SIZE, offset) : Promise.resolve(null)),
    [org, name, offset, version],
  );

  const groups = new Map<string, NonNullable<typeof tables.data>>();
  for (const t of tables.data ?? []) groups.set(t.group, [...(groups.get(t.group) ?? []), t]);

  return (
    <div className="explorer explorer-db">
      <div className="card table-nav">
        <ErrorNote error={tables.error} />
        {!tables.data && <Spinner />}
        {[...groups.entries()].map(([group, list]) => (
          <div key={group} className="stack-tight">
            <div className="muted small nav-group">{group}</div>
            {list.map((t) => (
              <button
                key={t.name}
                className={`table-link${t.name === name ? " table-link-active" : ""}`}
                onClick={() => setParams({ tab: "db", table: t.name })}
              >
                <span className="mono">{t.name}</span>
                <span className="muted small">{t.rows}</span>
              </button>
            ))}
          </div>
        ))}
        <p className="muted small">Rows where the organisation id is this organisation's.</p>
      </div>
      <div className="card table-view">
        <ErrorNote error={page.error} />
        {!page.data ? (
          <Spinner label="Loading rows…" />
        ) : (
          <>
            <div className="row between">
              <h3 className="mono">{page.data.name}</h3>
              <Pager
                total={page.data.total}
                offset={offset}
                onChange={(o) => setParams({ tab: "db", table: page.data!.name, offset: String(o) })}
              />
            </div>
            {page.data.rows.length === 0 ? (
              <Empty title="No rows for this organisation" />
            ) : (
              <div className="grid-scroll">
                <table className="table data-grid">
                  <thead>
                    <tr>
                      {page.data.columns.map((c) => (
                        <ColumnHead key={c.name} column={c} />
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {page.data.rows.map((row, i) => (
                      <tr key={i}>
                        {page.data!.columns.map((c) => (
                          <Cell key={c.name} value={row[c.name]} />
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function ColumnHead({ column }: { column: TableColumn }) {
  return (
    <th>
      <div className="mono">
        {column.primary_key && <span title="primary key">🔑 </span>}
        {column.name}
      </div>
      <div className="muted small mono">
        {column.type.toLowerCase()}
        {column.nullable ? "" : " not null"}
      </div>
    </th>
  );
}

function Cell({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <td className="muted small">null</td>;
  const text = typeof value === "object" ? JSON.stringify(value) : String(value);
  return (
    <td className="mono small cell" title={text.length > 80 ? text : undefined}>
      {text.length > 160 ? `${text.slice(0, 160)}…` : text}
    </td>
  );
}

function Pager({ total, offset, onChange }: { total: number; offset: number; onChange: (o: number) => void }) {
  const end = Math.min(offset + PAGE_SIZE, total);
  return (
    <div className="row small">
      <span className="muted">
        {total === 0 ? "0 rows" : `${offset + 1}–${end} of ${total}`}
      </span>
      <button className="btn" disabled={offset === 0} onClick={() => onChange(Math.max(0, offset - PAGE_SIZE))}>
        ←
      </button>
      <button className="btn" disabled={end >= total} onClick={() => onChange(offset + PAGE_SIZE)}>
        →
      </button>
    </div>
  );
}
