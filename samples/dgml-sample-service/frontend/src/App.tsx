import { useState } from "react";
import { Navigate, NavLink, Route, Routes } from "react-router-dom";
import { ActivityPanel } from "./components/ActivityPanel";
import { CreateOrgForm } from "./components/CreateOrgForm";
import { Spinner } from "./components/ui";
import { ApisPage } from "./pages/Apis";
import { DataPage } from "./pages/Data";
import { DocsetDetailPage } from "./pages/DocsetDetail";
import { DocsetsPage } from "./pages/Docsets";
import { FilesPage } from "./pages/Files";
import { SettingsPage } from "./pages/Settings";
import { ViewerPage } from "./pages/Viewer";
import { useApp } from "./state";

export default function App() {
  const { orgs, org, selectOrg, activeJobs } = useApp();
  const [activityOpen, setActivityOpen] = useState(false);
  const [creating, setCreating] = useState(false);

  if (orgs === null) {
    return (
      <div className="center-screen">
        <Spinner label="Connecting to the service…" />
      </div>
    );
  }

  if (!org || creating) {
    return (
      <div className="center-screen">
        <div className="card onboarding">
          <h1>DGML Sample Service</h1>
          <p className="muted">
            Each organisation is one DGML workspace: its documents live in Postgres, its files in S3
            (SeaweedFS locally), and its settings configure DGML in memory.
          </p>
          <CreateOrgForm onDone={() => setCreating(false)} />
          {creating && (
            <button className="btn-link" onClick={() => setCreating(false)}>
              Cancel
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">DG</span>
          <span>DGML Sample Service</span>
        </div>
        <nav className="nav">
          <NavLink to="/files">Files</NavLink>
          <NavLink to="/docsets">Docsets</NavLink>
          <NavLink to="/data">Data</NavLink>
          <NavLink to="/apis">APIs</NavLink>
          <NavLink to="/settings">Settings</NavLink>
        </nav>
        <div className="topbar-right">
          <select
            aria-label="Organisation"
            value={org.id}
            onChange={(e) =>
              e.target.value === "__new__" ? setCreating(true) : selectOrg(e.target.value)
            }
          >
            {orgs.map((o) => (
              <option key={o.id} value={o.id}>
                {o.name}
              </option>
            ))}
            <option value="__new__">+ New organisation…</option>
          </select>
          <button className="btn" onClick={() => setActivityOpen((v) => !v)}>
            {activeJobs > 0 ? <Spinner /> : null} Activity
            {activeJobs > 0 && <span className="badge">{activeJobs}</span>}
          </button>
        </div>
      </header>
      <main className="main" key={org.id}>
        <Routes>
          <Route path="/" element={<Navigate to="/files" replace />} />
          <Route path="/files" element={<FilesPage />} />
          <Route path="/files/:fileId" element={<ViewerPage />} />
          <Route path="/docsets" element={<DocsetsPage />} />
          <Route path="/docsets/:docsetId" element={<DocsetDetailPage />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="/apis" element={<ApisPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/files" replace />} />
        </Routes>
      </main>
      {activityOpen && <ActivityPanel onClose={() => setActivityOpen(false)} />}
    </div>
  );
}
