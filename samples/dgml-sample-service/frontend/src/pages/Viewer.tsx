// Document + DGML viewer: the rendered pages on the left; on the right the file's
// extracted values, its DGML XML, and its details — for one of its docsets.

import { useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api";
import type { Location } from "../api";
import { jobSummary } from "../components/ActivityPanel";
import { PageStack } from "../components/PageStack";
import type { Box } from "../components/PageStack";
import { ErrorNote, Spinner, StatusPill, formatDate, useAction, useLoad } from "../components/ui";
import { ValuesTree, collectBoxes } from "../components/ValuesTree";
import { XmlTree } from "../components/XmlTree";
import { useApp, useOrgId } from "../state";

type Tab = "values" | "xml" | "details";

export function ViewerPage() {
  const org = useOrgId();
  const { fileId = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const { version, track } = useApp();
  const file = useLoad(() => api.file(org, fileId), [org, fileId, version]);
  const docsets = useLoad(() => api.docsets(org), [org, version]);
  const [tab, setTab] = useState<Tab>("values");
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [xmlSelection, setXmlSelection] = useState<{ el: Element; boxes: Box[] } | null>(null);
  const [showAll, setShowAll] = useState(true);
  const action = useAction();

  const assigned = file.data?.docsets ?? [];
  const docsetId = params.get("docset") ?? assigned[0]?.id ?? null;
  const docset = docsets.data?.find((d) => d.id === docsetId) ?? null;
  const isAssigned = assigned.some((d) => d.id === docsetId);

  const pair = useLoad(
    () => (docsetId && isAssigned ? api.dgml(org, docsetId, fileId) : Promise.resolve(null)),
    [org, docsetId, fileId, isAssigned, version],
  );

  const boxes = useMemo(() => (pair.data?.values ? collectBoxes(pair.data.values) : []), [pair.data]);
  const selected: Box[] = xmlSelection
    ? xmlSelection.boxes
    : selectedPath
      ? boxes.filter((b) => b.key.startsWith(`${selectedPath}#`))
      : [];

  const selectValue = (path: string) => {
    setXmlSelection(null);
    setSelectedPath(path === selectedPath ? null : path);
  };
  const selectXml = (el: Element, locations: Location[]) => {
    setSelectedPath(null);
    setXmlSelection(
      xmlSelection?.el === el
        ? null
        : { el, boxes: locations.map((l, i) => ({ ...l, key: `xml#${i}`, label: el.tagName })) },
    );
  };

  if (file.error) {
    return (
      <div className="page">
        <Link to="/files">← Files</Link>
        <ErrorNote error={file.error} />
      </div>
    );
  }
  if (!file.data) return <Spinner label="Loading file…" />;
  const f = file.data;

  const extract = () =>
    docsetId && action.run(async () => track(await api.extract(org, docsetId, fileId)));
  const classify = (mode: string) => action.run(async () => track(await api.classify(org, fileId, mode, true)));
  const assign = (id: string) =>
    action.run(async () => {
      const res = await api.assign(org, id, fileId, true);
      if (res.job) track(res.job);
      setParams({ docset: id });
      file.reload();
    });

  const running = f.jobs.find((j) => j.status === "queued" || j.status === "running");

  return (
    <div className="viewer">
      <div className="viewer-head">
        <div>
          <Link to="/files" className="muted small">
            ← Files
          </Link>
          <h1 className="viewer-title">{f.original_filename}</h1>
          <div className="muted small">
            {f.page_count ?? "?"} page(s) · {f.text_mode} text · rendered by {f.page_image_renderer} at{" "}
            {f.page_image_dpi} dpi · <a href={api.sourceUrl(org, fileId)} target="_blank" rel="noreferrer">PDF</a>
          </div>
        </div>
        <div className="row wrap">
          {assigned.length > 0 && (
            <select
              aria-label="Docset"
              value={docsetId ?? ""}
              onChange={(e) => {
                setSelectedPath(null);
                setXmlSelection(null);
                setParams({ docset: e.target.value });
              }}
            >
              {assigned.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          )}
          {isAssigned && (
            <button
              className="btn btn-primary"
              disabled={!docset?.has_schema || action.busy || !!running}
              title={docset?.has_schema ? "" : "The docset has no extraction schema yet"}
              onClick={extract}
            >
              {pair.data?.has_extraction ? "Re-extract" : "Extract values"}
            </button>
          )}
          <select aria-label="Classify" value="" disabled={action.busy} onChange={(e) => e.target.value && classify(e.target.value)}>
            <option value="">Classify…</option>
            <option value="existing">into existing docsets</option>
            <option value="existing-or-new">existing or new docset</option>
          </select>
        </div>
      </div>
      <ErrorNote error={action.error} />
      {running && (
        <div className="note">
          <Spinner label={`${running.label ?? running.kind} — ${running.status}…`} />
        </div>
      )}

      <div className="viewer-body">
        <div className="viewer-pages">
          <PageStack
            pageCount={f.page_count ?? 0}
            pageUrl={(p) => api.pageUrl(org, fileId, p)}
            boxes={showAll && !xmlSelection ? boxes : []}
            selected={selected}
            onSelect={(b) => selectValue(b.key.split("#")[0])}
          />
        </div>

        <div className="viewer-side card">
          <div className="tabs">
            {(["values", "xml", "details"] as Tab[]).map((t) => (
              <button key={t} className={`tab${tab === t ? " tab-active" : ""}`} onClick={() => setTab(t)}>
                {t === "values" ? "Extracted values" : t === "xml" ? "DGML XML" : "Details"}
              </button>
            ))}
          </div>

          {tab !== "details" && !isAssigned && (
            <NotAssigned
              docsets={(docsets.data ?? []).filter((d) => !assigned.some((a) => a.id === d.id))}
              onAssign={assign}
              busy={action.busy}
            />
          )}
          {tab !== "details" && isAssigned && pair.error ? <ErrorNote error={pair.error} /> : null}
          {tab !== "details" && isAssigned && !pair.data && !pair.error && <Spinner label="Loading DGML…" />}

          {tab === "values" && isAssigned && pair.data && (
            pair.data.values ? (
              <>
                <label className="check small">
                  <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
                  Outline every value on the page
                </label>
                <ValuesTree values={pair.data.values} selectedPath={selectedPath} onSelect={selectValue} />
              </>
            ) : (
              <div className="muted">
                Nothing extracted for this docset yet.{" "}
                {docset?.has_schema ? "Run Extract values." : (
                  <>
                    The docset needs an <Link to={`/docsets/${docsetId}`}>extraction schema</Link> first.
                  </>
                )}
              </div>
            )
          )}

          {tab === "xml" && isAssigned && pair.data && (
            pair.data.xml ? (
              <>
                <div className="row between small">
                  <span className="muted mono">{pair.data.xml_key}</span>
                  <a href={api.dgmlDownloadUrl(org, docsetId!, fileId)}>Download</a>
                </div>
                <p className="muted small">Click an element with a <code>dg:origin</code> to see where it came from.</p>
                <XmlTree xml={pair.data.xml} selected={xmlSelection?.el ?? null} onSelect={selectXml} />
              </>
            ) : (
              <div className="muted">No DGML for this file in this docset yet — extraction writes it.</div>
            )
          )}

          {tab === "details" && (
            <div className="stack">
              <dl className="kv">
                <dt>File id</dt>
                <dd className="mono">{f.id}</dd>
                <dt>SHA-256</dt>
                <dd className="mono small">{f.sha256}</dd>
                <dt>Added</dt>
                <dd>{formatDate(f.added_at)}</dd>
                <dt>Docsets</dt>
                <dd>
                  {assigned.length
                    ? assigned.map((d) => (
                        <Link key={d.id} className="chip" to={`/docsets/${d.id}`}>
                          {d.name}
                        </Link>
                      ))
                    : "none"}
                </dd>
              </dl>
              <h4>Jobs</h4>
              {f.jobs.length === 0 ? (
                <div className="muted small">No jobs for this file yet.</div>
              ) : (
                <ul className="job-list compact">
                  {f.jobs.map((j) => (
                    <li key={j.id} className="job">
                      <div className="job-line">
                        <StatusPill status={j.status} />
                        <span className="job-label">{j.label ?? j.kind}</span>
                      </div>
                      <div className={j.status === "failed" ? "job-error" : "muted small"}>{jobSummary(j)}</div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function NotAssigned({
  docsets,
  onAssign,
  busy,
}: {
  docsets: { id: string; name: string }[];
  onAssign: (id: string) => void;
  busy: boolean;
}) {
  const [pick, setPick] = useState("");
  return (
    <div className="stack">
      <p className="muted">
        This file is not in a docset yet. DGML output is per docset: classify the file, or assign it.
      </p>
      {docsets.length > 0 && (
        <div className="row">
          <select value={pick} onChange={(e) => setPick(e.target.value)}>
            <option value="">Choose a docset…</option>
            {docsets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
          <button className="btn" disabled={!pick || busy} onClick={() => onAssign(pick)}>
            Assign
          </button>
        </div>
      )}
    </div>
  );
}
