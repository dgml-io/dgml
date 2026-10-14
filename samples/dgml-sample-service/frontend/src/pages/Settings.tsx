import { useEffect, useState } from "react";
import { api } from "../api";
import type { Settings, SettingsUpdate } from "../api";
import { ErrorNote, Spinner, formatDate, useAction, useLoad } from "../components/ui";
import { useApp, useOrgId } from "../state";

const FAMILY_LABELS: Record<string, string> = {
  anthropic: "Anthropic (Claude)",
  google: "Google (Gemini)",
  openai: "OpenAI (GPT)",
  anthropic_google: "Mixed: Gemini for light work, Claude for the rest",
};

const KEY_LABELS: Record<string, string> = {
  anthropic: "Anthropic API key",
  google: "Gemini API key",
  openai: "OpenAI API key",
  anthropic_google: "Anthropic API key",
};

export function SettingsPage() {
  const org = useOrgId();
  const settings = useLoad(() => api.settings(org), [org]);

  if (settings.error) return <ErrorNote error={settings.error} />;
  if (!settings.data) return <Spinner label="Loading settings…" />;
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Settings</h1>
          <p className="muted">
            Stored in Postgres (<code>dgml_settings</code>) and turned into a DGML{" "}
            <code>Configuration</code> on every request — there is no <code>config.toml</code>.
            Changes apply to the next DGML call.
          </p>
        </div>
      </div>
      <SettingsForm initial={settings.data} onSaved={settings.setData} />
      <DangerZone />
    </div>
  );
}

function SecretField({
  label,
  kind,
  settings,
  value,
  onChange,
}: {
  label: string;
  kind: string;
  settings: Settings;
  value: string | undefined;
  onChange: (v: string | undefined) => void;
}) {
  const status = settings.secrets[kind];
  const replacing = value !== undefined;
  return (
    <label className="field">
      <span>{label}</span>
      {status?.set && !replacing ? (
        <div className="row">
          <span className="chip chip-ok">set {formatDate(status.updated_at)}</span>
          <button type="button" className="btn" onClick={() => onChange("")}>
            Replace
          </button>
        </div>
      ) : (
        <div className="row">
          <input
            type="password"
            autoComplete="off"
            value={value ?? ""}
            placeholder={status?.set ? "Leave empty to remove" : "Not set"}
            onChange={(e) => onChange(e.target.value)}
          />
          {status?.set && (
            <button type="button" className="btn" onClick={() => onChange(undefined)}>
              Keep current
            </button>
          )}
        </div>
      )}
    </label>
  );
}

function SettingsForm({ initial, onSaved }: { initial: Settings; onSaved: (s: Settings) => void }) {
  const org = useOrgId();
  const [s, setS] = useState(initial);
  const [secrets, setSecrets] = useState<Record<string, string | undefined>>({});
  const [createBucket, setCreateBucket] = useState(false);
  const [saved, setSaved] = useState(false);
  const action = useAction();
  const probe = useAction();
  const [probeResult, setProbeResult] = useState<string | null>(null);

  useEffect(() => setS(initial), [initial]);

  const set = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    setSaved(false);
    setS((prev) => ({ ...prev, [key]: value }));
  };
  const setSecret = (kind: string, value: string | undefined) => {
    setSaved(false);
    setSecrets((prev) => ({ ...prev, [kind]: value }));
  };

  const save = (e: React.FormEvent) => {
    e.preventDefault();
    const update: SettingsUpdate = {
      llm_family: s.llm_family,
      text_mode: s.text_mode,
      ocr_provider: s.ocr_provider || null,
      create_bucket: createBucket,
      secrets: Object.fromEntries(
        Object.entries(secrets).filter((entry): entry is [string, string] => entry[1] !== undefined),
      ),
    };
    if (!s.storage_locked) {
      Object.assign(update, {
        s3_endpoint_url: s.s3_endpoint_url ?? "",
        s3_region: s.s3_region ?? "",
        s3_bucket: s.s3_bucket,
        blob_folder: s.blob_folder,
      });
    }
    action.run(async () => {
      const next = await api.saveSettings(org, update);
      setSecrets({});
      setCreateBucket(false);
      onSaved(next);
      setSaved(true);
    });
  };

  const testStorage = () =>
    probe.run(async () => {
      setProbeResult(null);
      const r = await api.testStorage(org, createBucket);
      setProbeResult(r.bucket_created ? `Created bucket ${r.bucket} and wrote a probe object.` : `Wrote and deleted a probe object in ${r.bucket}.`);
    });

  const needsOcr = s.text_mode !== "digital";
  const folder = s.blob_folder.replace(/^\/+|\/+$/g, "");
  const preview = `s3://${s.s3_bucket || "<bucket>"}/${folder ? `${folder}/` : ""}${org}/`;

  return (
    <form className="stack" onSubmit={save}>
      <section className="card stack">
        <h3>Language models</h3>
        <label className="field">
          <span>Model family</span>
          <select value={s.llm_family} onChange={(e) => set("llm_family", e.target.value)}>
            {s.options.llm_families.map((f) => (
              <option key={f} value={f}>
                {FAMILY_LABELS[f] ?? f}
              </option>
            ))}
          </select>
          <small className="muted">
            Expands to DGML's four model tiers (classification uses <em>light</em>, extraction{" "}
            <em>advanced</em>, schema generation <em>expert</em>).
          </small>
        </label>
        <SecretField
          label={KEY_LABELS[s.llm_family] ?? "API key"}
          kind="llm_api_key"
          settings={s}
          value={secrets.llm_api_key}
          onChange={(v) => setSecret("llm_api_key", v)}
        />
        {s.llm_family === "anthropic_google" && (
          <SecretField
            label="Gemini API key"
            kind="llm_api_key_google"
            settings={s}
            value={secrets.llm_api_key_google}
            onChange={(v) => setSecret("llm_api_key_google", v)}
          />
        )}
        <small className="muted">
          Keys are passed to DGML by value for each call — never through the environment. This
          sample keeps them in a Postgres table; a real deployment would use a vault.
        </small>
      </section>

      <section className="card stack">
        <h3>Text extraction</h3>
        <label className="field">
          <span>Default text mode for uploads</span>
          <select value={s.text_mode} onChange={(e) => set("text_mode", e.target.value)}>
            {s.options.text_modes.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <small className="muted">
            <em>digital</em> reads the PDF's text layer; <em>ocr</em> and <em>hybrid</em> need an OCR
            provider.
          </small>
        </label>
        <label className="field">
          <span>OCR provider</span>
          <select
            value={s.ocr_provider ?? ""}
            onChange={(e) => set("ocr_provider", e.target.value || null)}
          >
            <option value="">None</option>
            {s.options.ocr_providers.map((p) => (
              <option key={p} value={p}>
                {p === "macos" ? "Apple Vision (on-device, macOS)" : p}
              </option>
            ))}
          </select>
          {needsOcr && !s.ocr_provider && (
            <small className="warn-text">The {s.text_mode} text mode needs an OCR provider.</small>
          )}
        </label>
      </section>

      <section className="card stack">
        <div className="row between">
          <h3>Blob storage (S3)</h3>
          {s.storage_locked && <span className="chip chip-warn">locked — the workspace holds files</span>}
        </div>
        <p className="muted small">
          Source PDFs, page images, page text, schemas and <code>.dgml.xml</code> output. Pages
          are rendered with PDFium, so no Ghostscript is needed.
        </p>
        <div className="grid-2">
          <label className="field">
            <span>Endpoint URL</span>
            <input
              value={s.s3_endpoint_url ?? ""}
              disabled={s.storage_locked}
              placeholder="empty = AWS S3"
              onChange={(e) => set("s3_endpoint_url", e.target.value)}
            />
          </label>
          <label className="field">
            <span>Region</span>
            <input
              value={s.s3_region ?? ""}
              disabled={s.storage_locked}
              placeholder="optional"
              onChange={(e) => set("s3_region", e.target.value)}
            />
          </label>
          <label className="field">
            <span>Bucket</span>
            <input
              value={s.s3_bucket}
              disabled={s.storage_locked}
              onChange={(e) => set("s3_bucket", e.target.value)}
            />
          </label>
          <label className="field">
            <span>Folder</span>
            <input
              value={s.blob_folder}
              disabled={s.storage_locked}
              placeholder="empty = bucket root"
              onChange={(e) => set("blob_folder", e.target.value)}
            />
          </label>
        </div>
        <div className="muted small">
          Workspace path: <code>{s.storage_locked ? s.storage_path : preview}</code>
        </div>
        <div className="grid-2">
          <SecretField
            label="Access key id"
            kind="s3_access_key_id"
            settings={s}
            value={secrets.s3_access_key_id}
            onChange={(v) => setSecret("s3_access_key_id", v)}
          />
          <SecretField
            label="Secret access key"
            kind="s3_secret_access_key"
            settings={s}
            value={secrets.s3_secret_access_key}
            onChange={(v) => setSecret("s3_secret_access_key", v)}
          />
        </div>
        <div className="row wrap">
          <label className="check">
            <input type="checkbox" checked={createBucket} onChange={(e) => setCreateBucket(e.target.checked)} />
            Create the bucket if it does not exist
          </label>
          <button type="button" className="btn" disabled={probe.busy} onClick={testStorage}>
            {probe.busy ? <Spinner /> : null} Test saved connection
          </button>
        </div>
        {probeResult && <div className="note note-ok">{probeResult}</div>}
        <ErrorNote error={probe.error} />
      </section>

      <ErrorNote error={action.error} />
      <div className="row sticky-actions">
        <button className="btn btn-primary" disabled={action.busy}>
          {action.busy ? <Spinner /> : null} Save settings
        </button>
        {saved && <span className="ok-text">Saved.</span>}
        <span className="muted small">Storage changes are probed (write + delete) before saving.</span>
      </div>
    </form>
  );
}

function DangerZone() {
  const { org, reloadOrgs } = useApp();
  const action = useAction();
  if (!org) return null;
  const remove = () => {
    if (!confirm(`Delete organisation "${org.name}" and all of its files, docsets and DGML output?`))
      return;
    action.run(async () => {
      await api.deleteOrg(org.id);
      await reloadOrgs();
    });
  };
  return (
    <section className="card stack danger">
      <h3>Delete organisation</h3>
      <p className="muted small">
        Removes every file (records and blobs) and docset through DGML, then the organisation's
        settings, secrets and jobs.
      </p>
      <ErrorNote error={action.error} />
      <div className="row">
        <button className="btn btn-danger" disabled={action.busy} onClick={remove}>
          Delete {org.name}
        </button>
      </div>
    </section>
  );
}
