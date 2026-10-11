// Small shared UI pieces.

import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { ApiError } from "../api";
import type { Job } from "../api";

export function errorText(err: unknown): string {
  if (err instanceof ApiError) return `${err.message}${err.code ? ` (${err.code})` : ""}`;
  return err instanceof Error ? err.message : String(err);
}

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  return <div className="note note-error">{errorText(error)}</div>;
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="spinner-wrap">
      <span className="spinner" aria-hidden />
      {label && <span className="muted">{label}</span>}
    </span>
  );
}

export function StatusPill({ status }: { status: Job["status"] }) {
  return <span className={`pill pill-${status}`}>{status}</span>;
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty-title">{title}</div>
      {children && <div className="muted">{children}</div>}
    </div>
  );
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

/** Load data for a page; reloads when any dependency changes. */
export function useLoad<T>(
  load: () => Promise<T>,
  deps: unknown[],
): { data: T | null; error: unknown; loading: boolean; reload: () => void; setData: (d: T) => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  useEffect(() => {
    let live = true;
    setLoading(true);
    load()
      .then((d) => {
        if (!live) return;
        setData(d);
        setError(null);
      })
      .catch((e) => live && setError(e))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  return { data, error, loading, reload, setData };
}

/** Wrap an async action with busy/error state. */
export function useAction(): {
  busy: boolean;
  error: unknown;
  run: (fn: () => Promise<unknown>) => Promise<void>;
  clear: () => void;
} {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const run = useCallback(async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }, []);
  return { busy, error, run, clear: () => setError(null) };
}
