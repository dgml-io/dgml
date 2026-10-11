import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";
import { Empty, ErrorNote, Spinner, useAction, useLoad } from "../components/ui";
import { useApp, useOrgId } from "../state";

export const parseQuestions = (text: string) =>
  text
    .split("\n")
    .map((q) => q.trim())
    .filter(Boolean);

export function DocsetsPage() {
  const org = useOrgId();
  const { version } = useApp();
  const docsets = useLoad(() => api.docsets(org), [org, version]);
  const [creating, setCreating] = useState(false);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Docsets</h1>
          <p className="muted">
            A docset groups documents of one kind. Classification routes files into docsets by their
            description and key questions; an extraction schema says which values to pull out.
          </p>
        </div>
        {!creating && (
          <button className="btn btn-primary" onClick={() => setCreating(true)}>
            New docset
          </button>
        )}
      </div>

      {creating && <CreateDocset onClose={() => setCreating(false)} />}

      <ErrorNote error={docsets.error} />
      {docsets.data === null ? (
        <Spinner label="Loading docsets…" />
      ) : docsets.data.length === 0 ? (
        !creating && (
          <div className="card">
            <Empty title="No docsets yet">
              Create one by hand, or classify a file with "existing or new" and let DGML propose one.
            </Empty>
          </div>
        )
      ) : (
        <div className="grid">
          {docsets.data.map((ds) => (
            <Link key={ds.id} to={`/docsets/${ds.id}`} className="card card-link">
              <div className="card-title">{ds.name}</div>
              <div className="muted clamp">{ds.description || "No description"}</div>
              <div className="row small">
                <span className="chip">{ds.file_ids.length} file(s)</span>
                <span className={`chip ${ds.has_schema ? "chip-ok" : "chip-warn"}`}>
                  {ds.has_schema ? "schema" : "no schema"}
                </span>
                {ds.key_questions.length > 0 && (
                  <span className="chip">{ds.key_questions.length} key question(s)</span>
                )}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

function CreateDocset({ onClose }: { onClose: () => void }) {
  const org = useOrgId();
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [questions, setQuestions] = useState("");
  const action = useAction();

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    action.run(async () => {
      const ds = await api.createDocset(org, {
        name,
        description,
        key_questions: parseQuestions(questions),
      });
      navigate(`/docsets/${ds.id}`);
    });
  };

  return (
    <form className="card stack" onSubmit={submit}>
      <h3>New docset</h3>
      <label className="field">
        <span>Name</span>
        <input value={name} autoFocus placeholder="Commercial Lease" onChange={(e) => setName(e.target.value)} />
      </label>
      <label className="field">
        <span>Description</span>
        <input
          value={description}
          placeholder="A lease agreement for commercial property"
          onChange={(e) => setDescription(e.target.value)}
        />
        <small className="muted">Classification judges fit by document type from this.</small>
      </label>
      <label className="field">
        <span>Key questions (one per line)</span>
        <textarea
          rows={3}
          value={questions}
          placeholder={"Who is the tenant?\nWhat is the monthly rent?"}
          onChange={(e) => setQuestions(e.target.value)}
        />
      </label>
      <ErrorNote error={action.error} />
      <div className="row">
        <button className="btn btn-primary" disabled={!name.trim() || action.busy}>
          Create
        </button>
        <button type="button" className="btn" onClick={onClose}>
          Cancel
        </button>
      </div>
    </form>
  );
}
