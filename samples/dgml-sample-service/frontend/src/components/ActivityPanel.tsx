import { Link } from "react-router-dom";
import type { Job } from "../api";
import { useApp } from "../state";
import { Empty, StatusPill, formatDate } from "./ui";

/** One line summarising what a finished job produced. */
export function jobSummary(job: Job): string | null {
  if (job.status === "failed") return job.error;
  const r = job.result;
  if (!r) return null;
  switch (job.kind) {
    case "add_file": {
      const soft = [r.page_render_error, r.text_extraction_error, r.conversion_error, r.page_count_error]
        .filter(Boolean)
        .join("; ");
      const cls = r.classification
        ? r.classification.error
          ? ` · classification failed: ${r.classification.error}`
          : ` · ${classifySummary(r.classification)}`
        : "";
      return `${r.file.page_count ?? "?"} page(s)${soft ? ` · ${soft}` : ""}${cls}`;
    }
    case "classify":
      return classifySummary(r);
    case "extract":
      return `${r.field_count} field(s) extracted with ${r.model}`;
    case "generate_schema":
      return `schema generated with ${r.model}`;
  }
  return null;
}

function classifySummary(r: Record<string, any>): string {
  if (r.decision === "none") return `no matching docset${r.reason ? ` — ${r.reason}` : ""}`;
  const ex = r.extraction;
  const extraction = ex ? (ex.error ? ` · extraction failed: ${ex.error}` : " · extracted") : "";
  return `${r.created_docset ? "new docset" : "assigned"}${extraction}`;
}

export function ActivityPanel({ onClose }: { onClose: () => void }) {
  const { jobs } = useApp();
  return (
    <aside className="activity">
      <div className="activity-head">
        <h2>Activity</h2>
        <button className="btn-link" onClick={onClose}>
          Close
        </button>
      </div>
      {jobs.length === 0 ? (
        <Empty title="No jobs yet">Uploads, classification and extraction run here.</Empty>
      ) : (
        <ul className="job-list">
          {jobs.map((job) => (
            <li key={job.id} className="job">
              <div className="job-line">
                <StatusPill status={job.status} />
                <span className="job-label">{job.label ?? job.kind}</span>
              </div>
              {jobSummary(job) && (
                <div className={job.status === "failed" ? "job-error" : "muted small"}>
                  {jobSummary(job)}
                </div>
              )}
              <div className="muted small">
                {formatDate(job.created_at)}
                {(job.file_id ?? job.result?.file?.id) && (
                  <>
                    {" · "}
                    <Link to={`/files/${job.file_id ?? job.result?.file?.id}`} onClick={onClose}>
                      open file
                    </Link>
                  </>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
