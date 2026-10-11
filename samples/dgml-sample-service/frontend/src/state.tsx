// App-wide state: the organisation being worked in, and the background jobs it runs.
//
// Jobs are polled while any is queued or running. Every time one finishes, `version`
// increments; pages list it in their effect dependencies to refetch what the job
// may have changed (a new file, an assignment, a schema, a .dgml.xml).

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { api } from "./api";
import type { Job, Org } from "./api";

const ORG_KEY = "dgml-sample.org";

function readStoredOrg(): string | null {
  try {
    return localStorage.getItem(ORG_KEY);
  } catch {
    return null;
  }
}

function storeOrg(id: string | null): void {
  try {
    if (id) localStorage.setItem(ORG_KEY, id);
    else localStorage.removeItem(ORG_KEY);
  } catch {
    /* storage unavailable: the choice just isn't remembered */
  }
}

interface AppState {
  orgs: Org[] | null;
  org: Org | null;
  selectOrg: (id: string) => void;
  reloadOrgs: (select?: string) => Promise<void>;
  jobs: Job[];
  activeJobs: number;
  track: (jobs: Job | Job[]) => void;
  version: number;
  bump: () => void;
}

const Ctx = createContext<AppState | null>(null);

export function useApp(): AppState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useApp outside <AppProvider>");
  return ctx;
}

/** The current organisation's id — only call below a guard that ensures one is selected. */
export function useOrgId(): string {
  const { org } = useApp();
  if (!org) throw new Error("no organisation selected");
  return org.id;
}

const isActive = (j: Job) => j.status === "queued" || j.status === "running";

export function AppProvider({ children }: { children: ReactNode }) {
  const [orgs, setOrgs] = useState<Org[] | null>(null);
  const [orgId, setOrgId] = useState<string | null>(readStoredOrg());
  const [jobs, setJobs] = useState<Job[]>([]);
  const [version, setVersion] = useState(0);
  const known = useRef<Map<string, Job>>(new Map());

  const reloadOrgs = useCallback(async (select?: string) => {
    const list = await api.orgs();
    setOrgs(list);
    setOrgId((current) => {
      const want = select ?? current;
      const next = list.find((o) => o.id === want)?.id ?? list[0]?.id ?? null;
      storeOrg(next);
      return next;
    });
  }, []);

  useEffect(() => {
    reloadOrgs().catch(() => setOrgs([]));
  }, [reloadOrgs]);

  const selectOrg = useCallback((id: string) => {
    storeOrg(id);
    setOrgId(id);
  }, []);

  const org = useMemo(() => orgs?.find((o) => o.id === orgId) ?? null, [orgs, orgId]);

  const refreshJobs = useCallback(async () => {
    if (!org) return;
    const list = await api.jobs(org.id);
    let finished = false;
    for (const job of list) {
      const prev = known.current.get(job.id);
      if (prev && isActive(prev) && !isActive(job)) finished = true;
      known.current.set(job.id, job);
    }
    setJobs(list);
    if (finished) setVersion((v) => v + 1);
  }, [org]);

  // Reset and load the job list whenever the organisation changes.
  useEffect(() => {
    known.current = new Map();
    setJobs([]);
    refreshJobs().catch(() => undefined);
  }, [refreshJobs]);

  const activeJobs = jobs.filter(isActive).length;

  useEffect(() => {
    if (!activeJobs) return;
    const timer = window.setInterval(() => refreshJobs().catch(() => undefined), 1500);
    return () => window.clearInterval(timer);
  }, [activeJobs, refreshJobs]);

  const track = useCallback((added: Job | Job[]) => {
    const list = Array.isArray(added) ? added : [added];
    for (const job of list) known.current.set(job.id, job);
    setJobs((prev) => [...list, ...prev.filter((j) => !list.some((a) => a.id === j.id))]);
  }, []);

  const bump = useCallback(() => setVersion((v) => v + 1), []);

  const value: AppState = {
    orgs,
    org,
    selectOrg,
    reloadOrgs,
    jobs,
    activeJobs,
    track,
    version,
    bump,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
