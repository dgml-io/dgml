import { useState } from "react";
import { api } from "../api";
import { useApp } from "../state";
import { ErrorNote, useAction } from "./ui";

const slugify = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);

export function CreateOrgForm({ onDone }: { onDone?: () => void }) {
  const { reloadOrgs } = useApp();
  const [name, setName] = useState("");
  const [slug, setSlug] = useState("");
  const [slugEdited, setSlugEdited] = useState(false);
  const action = useAction();

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    action.run(async () => {
      const org = await api.createOrg(name, slug);
      await reloadOrgs(org.id);
      onDone?.();
    });
  };

  return (
    <form className="stack" onSubmit={submit}>
      <label className="field">
        <span>Organisation name</span>
        <input
          value={name}
          autoFocus
          placeholder="Acme Corp"
          onChange={(e) => {
            setName(e.target.value);
            if (!slugEdited) setSlug(slugify(e.target.value));
          }}
        />
      </label>
      <label className="field">
        <span>Slug</span>
        <input
          value={slug}
          placeholder="acme"
          onChange={(e) => {
            setSlug(e.target.value);
            setSlugEdited(true);
          }}
        />
        <small className="muted">
          DGML namespace segment: <code>http://dgml.io/{slug || "<slug>"}/…</code>. It cannot change
          later.
        </small>
      </label>
      <ErrorNote error={action.error} />
      <div className="row">
        <button className="btn btn-primary" disabled={!name.trim() || !slug || action.busy}>
          Create organisation
        </button>
      </div>
    </form>
  );
}
