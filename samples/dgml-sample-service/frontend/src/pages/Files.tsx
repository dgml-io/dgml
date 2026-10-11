import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { FileEntry } from "../api";
import { jobSummary } from "../components/ActivityPanel";
import { Empty, ErrorNote, Spinner, StatusPill, formatDate, useAction, useLoad } from "../components/ui";
import { useApp, useOrgId } from "../state";

const CLASSIFY_OPTIONS = [
  { value: "none", label: "Don't classify" },
  { value: "existing", label: "Into an existing docset (may decline)" },
  { value: "existing-or-new", label: "Into an existing docset, or create one" },
];

export function FilesPage() {
  const org = useOrgId();
  const { version, jobs, track, bump } = useApp();
  const files = useLoad(() => api.files(org), [org, version]);
  const settings = useLoad(() => api.settings(org), [org]);

  const uploads = jobs.filter((j) => j.kind === "add_file").slice(0, 6);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Files</h1>
          <p className="muted">
            Uploads go through <code>FileStore.add</code>: the PDF is copied into S3, its pages are
            rendered with PDFium and its text extracted.
          </p>
        </div>
      </div>

      <UploadCard
        defaultTextMode={settings.data?.text_mode ?? "digital"}
        textModes={settings.data?.options.text_modes ?? ["digital"]}
        onQueued={(queued) => track(queued)}
      />

      {uploads.length > 0 && (
        <div className="card">
          <h3>Recent uploads</h3>
          <ul className="job-list compact">
            {uploads.map((job) => (
              <li key={job.id} className="job">
                <div className="job-line">
                  <StatusPill status={job.status} />
                  <span className="job-label">{String(job.params.filename ?? job.label)}</span>
                  <span className="muted small">{jobSummary(job)}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="card">
        <ErrorNote error={files.error} />
        {files.data === null ? (
          <Spinner label="Loading files…" />
        ) : files.data.length === 0 ? (
          <Empty title="No files yet">Upload a PDF above to get started.</Empty>
        ) : (
          <FileTable files={files.data} onChanged={bump} />
        )}
      </div>
    </div>
  );
}

function UploadCard({
  defaultTextMode,
  textModes,
  onQueued,
}: {
  defaultTextMode: string;
  textModes: string[];
  onQueued: (jobs: Awaited<ReturnType<typeof api.upload>>) => void;
}) {
  const org = useOrgId();
  const input = useRef<HTMLInputElement>(null);
  const [classify, setClassify] = useState("none");
  const [textMode, setTextMode] = useState<string>("");
  const [dragging, setDragging] = useState(false);
  const action = useAction();

  const send = (list: FileList | null) => {
    if (!list || list.length === 0) return;
    const chosen = Array.from(list);
    action.run(async () => {
      onQueued(await api.upload(org, chosen, classify, textMode || undefined));
      if (input.current) input.current.value = "";
    });
  };

  return (
    <div className="card">
      <div
        className={`dropzone${dragging ? " dropzone-active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          send(e.dataTransfer.files);
        }}
        onClick={() => input.current?.click()}
      >
        <input
          ref={input}
          type="file"
          multiple
          accept=".pdf,application/pdf"
          hidden
          onChange={(e) => send(e.target.files)}
        />
        <div className="dropzone-title">{action.busy ? <Spinner label="Uploading…" /> : "Drop PDFs here, or click to choose"}</div>
        <div className="muted small">Each file becomes a background job.</div>
      </div>
      <div className="row wrap">
        <label className="field inline">
          <span>After upload</span>
          <select value={classify} onChange={(e) => setClassify(e.target.value)}>
            {CLASSIFY_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field inline">
          <span>Text</span>
          <select value={textMode} onChange={(e) => setTextMode(e.target.value)}>
            <option value="">Settings default ({defaultTextMode})</option>
            {textModes.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
      </div>
      <ErrorNote error={action.error} />
    </div>
  );
}

function FileTable({ files, onChanged }: { files: FileEntry[]; onChanged: () => void }) {
  const org = useOrgId();
  const { track } = useApp();
  const action = useAction();

  const classify = (f: FileEntry, mode: string) =>
    action.run(async () => track(await api.classify(org, f.id, mode, true)));
  const remove = (f: FileEntry) => {
    if (!confirm(`Delete ${f.original_filename}? Its pages, text and DGML output are removed too.`))
      return;
    action.run(async () => {
      await api.deleteFile(org, f.id);
      onChanged();
    });
  };

  return (
    <>
      <ErrorNote error={action.error} />
      <table className="table">
        <thead>
          <tr>
            <th>File</th>
            <th>Pages</th>
            <th>Text</th>
            <th>Docsets</th>
            <th>Added</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {files.map((f) => (
            <tr key={f.id}>
              <td>
                <Link to={`/files/${f.id}`} className="strong">
                  {f.original_filename}
                </Link>
                <div className="muted small mono">{f.id}</div>
              </td>
              <td>{f.page_count ?? "—"}</td>
              <td>
                {f.text_mode ?? "—"}
                <div className="muted small">{f.page_image_renderer}</div>
              </td>
              <td>
                {f.docsets.length === 0 ? (
                  <span className="muted">unassigned</span>
                ) : (
                  f.docsets.map((d) => (
                    <Link key={d.id} to={`/files/${f.id}?docset=${d.id}`} className="chip">
                      {d.name}
                    </Link>
                  ))
                )}
              </td>
              <td className="small">{formatDate(f.added_at)}</td>
              <td className="actions">
                <select
                  aria-label="Classify"
                  value=""
                  disabled={action.busy}
                  onChange={(e) => e.target.value && classify(f, e.target.value)}
                >
                  <option value="">Classify…</option>
                  <option value="existing">into existing docsets</option>
                  <option value="existing-or-new">existing or new docset</option>
                </select>
                <button className="btn btn-danger-quiet" onClick={() => remove(f)}>
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
