// Output of the APIs page: the equivalent curl command and the response.

import { useState } from "react";
import type { FieldState } from "./form";

export interface ApiResponse {
  status: number;
  statusText: string;
  ms: number;
  contentType: string;
  size: number;
  text?: string;
  json?: unknown;
  blobUrl?: string;
  filename?: string;
}

const shellQuote = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`;

export function CurlLine({
  method,
  url,
  json,
  multipart,
}: {
  method: string;
  url: string;
  json: unknown;
  multipart: [string, FieldState][] | null;
}) {
  const [copied, setCopied] = useState(false);
  const parts = [`curl -X ${method.toUpperCase()}`, shellQuote(`${window.location.origin}${url}`)];
  if (multipart) {
    for (const [name, st] of multipart) {
      if (st.files.length) st.files.forEach((f) => parts.push(`-F ${shellQuote(`${name}=@${f.name}`)}`));
      else parts.push(`-F ${shellQuote(`${name}=${st.value}`)}`);
    }
  } else if (json !== undefined) {
    parts.push(`-H 'Content-Type: application/json'`, `-d ${shellQuote(JSON.stringify(json))}`);
  }
  const cmd = parts.join(" ");
  return (
    <details className="api-curl">
      <summary className="small muted">curl</summary>
      <div className="row">
        <pre className="preview-text api-curl-cmd">{cmd}</pre>
        <button
          className="btn"
          onClick={() => {
            navigator.clipboard?.writeText(cmd).then(() => {
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1200);
            });
          }}
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
    </details>
  );
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function ResponseView({ response }: { response: ApiResponse }) {
  const ok = response.status >= 200 && response.status < 300;
  let content;
  if (response.blobUrl && response.contentType.startsWith("image/")) {
    content = <img className="preview-image" src={response.blobUrl} alt="response" />;
  } else if (response.blobUrl && response.contentType === "application/pdf") {
    content = <iframe className="preview-pdf" src={response.blobUrl} title="response PDF" />;
  } else if (response.blobUrl) {
    content = (
      <a className="btn" href={response.blobUrl} download={response.filename ?? "response"}>
        Download {response.filename ?? "response"}
      </a>
    );
  } else if (response.status === 204 || !response.text) {
    content = <span className="muted">No content.</span>;
  } else {
    content = (
      <pre className="preview-text api-response">
        {response.json !== undefined ? JSON.stringify(response.json, null, 2) : response.text}
      </pre>
    );
  }
  return (
    <div className="card stack">
      <div className="row wrap between">
        <h3>Response</h3>
        <div className="row wrap small">
          <span className={`pill ${ok ? "pill-succeeded" : "pill-failed"}`}>
            {response.status} {response.statusText}
          </span>
          <span className="muted">{response.ms} ms</span>
          <span className="muted">{formatBytes(response.size)}</span>
          {response.contentType && <span className="muted mono">{response.contentType}</span>}
          {response.text && (
            <button className="btn-link" onClick={() => navigator.clipboard?.writeText(response.text!)}>
              copy
            </button>
          )}
        </div>
      </div>
      {content}
    </div>
  );
}
