import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ApiError, api } from "../api";
import type { DocSetDetail } from "../api";
import { Empty, ErrorNote, Spinner, useAction, useLoad } from "../components/ui";
import { useApp, useOrgId } from "../state";
import { parseQuestions } from "./Docsets";

export function DocsetDetailPage() {
  const org = useOrgId();
  const { docsetId = "" } = useParams();
  const { version } = useApp();
  const docset = useLoad(() => api.docset(org, docsetId), [org, docsetId, version]);

  if (docset.error) {
    return (
      <div className="page">
        <Link to="/docsets">← Docsets</Link>
        <ErrorNote error={docset.error} />
      </div>
    );
  }
  if (!docset.data) return <Spinner label="Loading docset…" />;
  const ds = docset.data;

  return (
    <div className="page">
      <Link to="/docsets" className="muted small">
        ← Docsets
      </Link>
      <Header ds={ds} onSaved={docset.setData} />
      <AssignedFiles ds={ds} onChanged={docset.reload} />
      <SchemaEditor ds={ds} onChanged={docset.reload} />
      <GuidanceEditor ds={ds} onChanged={docset.reload} />
    </div>
  );
}

function Header({ ds, onSaved }: { ds: DocSetDetail; onSaved: (d: DocSetDetail) => void }) {
  const org = useOrgId();
  const navigate = useNavigate();
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(ds.name);
  const [description, setDescription] = useState(ds.description);
  const [questions, setQuestions] = useState(ds.key_questions.join("\n"));
  const action = useAction();

  const save = (e: React.FormEvent) => {
    e.preventDefault();
    action.run(async () => {
      const updated = await api.updateDocset(org, ds.id, {
        name,
        description,
        key_questions: parseQuestions(questions),
      });
      onSaved({ ...ds, ...updated });
      setEditing(false);
    });
  };

  const remove = () => {
    if (!confirm(`Delete docset "${ds.name}"? Files stay; their assignments and DGML output for this docset go.`))
      return;
    action.run(async () => {
      await api.deleteDocset(org, ds.id);
      navigate("/docsets");
    });
  };

  if (editing) {
    return (
      <form className="card stack" onSubmit={save}>
        <label className="field">
          <span>Name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="field">
          <span>Description</span>
          <input value={description} onChange={(e) => setDescription(e.target.value)} />
        </label>
        <label className="field">
          <span>Key questions (one per line)</span>
          <textarea rows={4} value={questions} onChange={(e) => setQuestions(e.target.value)} />
        </label>
        <ErrorNote error={action.error} />
        <div className="row">
          <button className="btn btn-primary" disabled={action.busy || !name.trim()}>
            Save
          </button>
          <button type="button" className="btn" onClick={() => setEditing(false)}>
            Cancel
          </button>
        </div>
      </form>
    );
  }

  return (
    <div className="page-head">
      <div>
        <h1>{ds.name}</h1>
        <p className="muted">{ds.description || "No description."}</p>
        {ds.key_questions.length > 0 && (
          <ul className="questions">
            {ds.key_questions.map((q) => (
              <li key={q}>{q}</li>
            ))}
          </ul>
        )}
        <div className="muted small mono">{ds.id}</div>
        <ErrorNote error={action.error} />
      </div>
      <div className="row">
        <button className="btn" onClick={() => setEditing(true)}>
          Edit
        </button>
        <button className="btn btn-danger-quiet" onClick={remove}>
          Delete
        </button>
      </div>
    </div>
  );
}

function AssignedFiles({ ds, onChanged }: { ds: DocSetDetail; onChanged: () => void }) {
  const org = useOrgId();
  const { track, version } = useApp();
  const allFiles = useLoad(() => api.files(org), [org, version]);
  const [pick, setPick] = useState("");
  const [extractNow, setExtractNow] = useState(true);
  const action = useAction();

  const unassigned = (allFiles.data ?? []).filter((f) => !ds.file_ids.includes(f.id));

  const assign = () =>
    action.run(async () => {
      const res = await api.assign(org, ds.id, pick, extractNow);
      if (res.job) track(res.job);
      setPick("");
      onChanged();
    });

  const extract = (fileId: string) =>
    action.run(async () => track(await api.extract(org, ds.id, fileId)));

  const unassign = (fileId: string) =>
    action.run(async () => {
      await api.unassign(org, ds.id, fileId);
      onChanged();
    });

  return (
    <div className="card">
      <h3>Files</h3>
      <ErrorNote error={action.error} />
      {ds.files.length === 0 ? (
        <Empty title="No files in this docset">Assign one below, or classify files into it.</Empty>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>File</th>
              <th>Pages</th>
              <th>DGML</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {ds.files.map((f) => (
              <tr key={f.id}>
                <td>
                  <Link to={`/files/${f.id}?docset=${ds.id}`} className="strong">
                    {f.original_filename ?? f.id}
                  </Link>
                  {f.missing && <span className="chip chip-warn">missing</span>}
                </td>
                <td>{f.page_count ?? "—"}</td>
                <td>
                  <span className={`chip ${f.has_dgml ? "chip-ok" : ""}`}>
                    {f.has_dgml ? "extracted" : "not yet"}
                  </span>
                </td>
                <td className="actions">
                  <Link className="btn" to={`/files/${f.id}?docset=${ds.id}`}>
                    View
                  </Link>
                  <button
                    className="btn"
                    disabled={!ds.has_schema || action.busy}
                    title={ds.has_schema ? "" : "Set an extraction schema first"}
                    onClick={() => extract(f.id)}
                  >
                    {f.has_dgml ? "Re-extract" : "Extract"}
                  </button>
                  <button className="btn btn-danger-quiet" onClick={() => unassign(f.id)}>
                    Unassign
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="row wrap assign-row">
        <select value={pick} onChange={(e) => setPick(e.target.value)}>
          <option value="">Assign a file…</option>
          {unassigned.map((f) => (
            <option key={f.id} value={f.id}>
              {f.original_filename}
            </option>
          ))}
        </select>
        <label className="check">
          <input
            type="checkbox"
            checked={extractNow}
            disabled={!ds.has_schema}
            onChange={(e) => setExtractNow(e.target.checked)}
          />
          Extract right away {ds.has_schema ? "" : "(needs a schema)"}
        </label>
        <button className="btn btn-primary" disabled={!pick || action.busy} onClick={assign}>
          Assign
        </button>
      </div>
    </div>
  );
}

const SCHEMA_PLACEHOLDER = `namespace docset = "http://dgml.io/<org>/<Docset>"

MonthlyRent =
  element docset:MonthlyRent {
    xsd:decimal
  }`;

function SchemaEditor({ ds, onChanged }: { ds: DocSetDetail; onChanged: () => void }) {
  const org = useOrgId();
  const { track, version } = useApp();
  const [text, setText] = useState("");
  const [saved, setSaved] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<unknown>(null);
  const action = useAction();

  useEffect(() => {
    let live = true;
    if (!ds.has_schema) {
      setSaved(null);
      setText("");
      return;
    }
    api
      .schema(org, ds.id)
      .then((r) => {
        if (!live) return;
        setSaved(r.schema);
        setText(r.schema);
        setLoadError(null);
      })
      .catch((e) => live && !(e instanceof ApiError && e.status === 404) && setLoadError(e));
    return () => {
      live = false;
    };
  }, [org, ds.id, ds.has_schema, version]);

  const save = () =>
    action.run(async () => {
      const r = await api.setSchema(org, ds.id, text);
      setSaved(r.schema);
      onChanged();
    });
  const clear = () => {
    if (!confirm("Clear the extraction schema? Existing extractions are kept.")) return;
    action.run(async () => {
      await api.clearSchema(org, ds.id);
      onChanged();
    });
  };
  const generate = () => action.run(async () => track(await api.generateSchema(org, ds.id)));

  const dirty = text !== (saved ?? "");

  return (
    <div className="card stack">
      <div className="row between">
        <h3>Extraction schema</h3>
        <button
          className="btn"
          disabled={ds.file_ids.length === 0 || action.busy}
          title={ds.file_ids.length ? "" : "Assign files first"}
          onClick={generate}
        >
          Generate from {Math.min(ds.file_ids.length, 3) || "assigned"} file(s)
        </button>
      </div>
      <p className="muted small">
        RELAX NG Compact, in DGML's supported subset. <code>set_schema</code> validates it before
        storing; generation asks the expert-tier model to propose one from sample files.
      </p>
      <textarea
        className="code"
        rows={Math.max(8, Math.min(24, text.split("\n").length + 1))}
        value={text}
        placeholder={SCHEMA_PLACEHOLDER}
        spellCheck={false}
        onChange={(e) => setText(e.target.value)}
      />
      <ErrorNote error={loadError ?? action.error} />
      <div className="row">
        <button className="btn btn-primary" disabled={!dirty || !text.trim() || action.busy} onClick={save}>
          Save schema
        </button>
        {dirty && saved !== null && (
          <button className="btn" onClick={() => setText(saved)}>
            Revert
          </button>
        )}
        {saved !== null && (
          <button className="btn btn-danger-quiet" onClick={clear}>
            Clear
          </button>
        )}
      </div>
    </div>
  );
}

function GuidanceEditor({ ds, onChanged }: { ds: DocSetDetail; onChanged: () => void }) {
  const org = useOrgId();
  const [text, setText] = useState("");
  const [saved, setSaved] = useState("");
  const action = useAction();

  useEffect(() => {
    if (!ds.has_guidance) {
      setText("");
      setSaved("");
      return;
    }
    api.guidance(org, ds.id).then((r) => {
      setText(r.guidance);
      setSaved(r.guidance);
    });
  }, [org, ds.id, ds.has_guidance]);

  const save = () =>
    action.run(async () => {
      if (text.trim()) await api.setGuidance(org, ds.id, text);
      else await api.clearGuidance(org, ds.id);
      setSaved(text);
      onChanged();
    });

  return (
    <div className="card stack">
      <h3>Extraction guidance</h3>
      <p className="muted small">Free-form notes added to every extraction prompt for this docset.</p>
      <textarea
        rows={4}
        value={text}
        placeholder="e.g. Rent is always monthly; ignore amounts in the appendix."
        onChange={(e) => setText(e.target.value)}
      />
      <ErrorNote error={action.error} />
      <div className="row">
        <button className="btn btn-primary" disabled={text === saved || action.busy} onClick={save}>
          Save guidance
        </button>
      </div>
    </div>
  );
}
