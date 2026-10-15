// APIs: an interactive console for the service's REST API, built from its OpenAPI
// document (/openapi.json). Like Swagger UI, but every parameter that names something —
// an organisation, a docset, a file, a job, a table, an S3 key, a page, a mode — offers
// the values that actually exist, so a call can be tried without copying ids around.
//
// The pieces live in ./apis: the OpenAPI model (openapi.ts), the suggested values
// (catalog.ts), the body form (form.ts), and the console for one endpoint.

import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Empty, ErrorNote, Spinner, useLoad } from "../components/ui";
import { useApp } from "../state";
import { loadCatalog } from "./apis/catalog";
import { EndpointConsole, MethodBadge } from "./apis/EndpointConsole";
import { endpointsOf, fetchSpec, groupEndpoints, shortPath } from "./apis/openapi";

export function ApisPage() {
  const { org, version } = useApp();
  const spec = useLoad(fetchSpec, []);
  const [params, setParams] = useSearchParams();
  const [filter, setFilter] = useState("");
  const endpoints = useMemo(() => (spec.data ? endpointsOf(spec.data) : []), [spec.data]);
  const groups = useMemo(() => groupEndpoints(endpoints, filter), [endpoints, filter]);
  const selected = endpoints.find((e) => e.id === params.get("op")) ?? endpoints[0];

  // One catalog for the page, for the organisation the open endpoint targets (the
  // current one unless its org_id says otherwise). Refetched after every write.
  const [catalogOrg, setCatalogOrg] = useState(org?.id ?? "");
  const catalog = useLoad(() => loadCatalog(catalogOrg), [catalogOrg, version]);

  return (
    <div className="page page-wide">
      <div className="page-head">
        <div>
          <h1>APIs</h1>
          <p className="muted">
            Try the service's REST API. Endpoints come from its OpenAPI document; parameters offer the
            organisations, docsets, files, jobs and objects that exist, so you can pick rather than paste.
            Calls are real — writes and deletes change data.
          </p>
        </div>
        <div className="row">
          <a className="btn" href="/openapi.json" target="_blank" rel="noreferrer">
            openapi.json
          </a>
        </div>
      </div>
      <ErrorNote error={spec.error} />
      {!spec.data ? (
        !spec.error && <Spinner label="Loading the API description…" />
      ) : (
        <div className="explorer explorer-db apis">
          <nav className="table-nav apis-nav">
            <input
              type="search"
              placeholder="Filter endpoints"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
            {groups.map((g) => (
              <div key={g.name} className="stack-tight">
                <div className="nav-group small muted">{g.name}</div>
                {g.items.map((e) => (
                  <button
                    key={e.id}
                    className={`table-link api-link${selected?.id === e.id ? " table-link-active" : ""}`}
                    onClick={() => setParams({ op: e.id })}
                    title={`${e.method.toUpperCase()} ${e.path}`}
                  >
                    <MethodBadge method={e.method} />
                    <span className="api-link-text">
                      <span className="api-link-summary">{e.op.summary ?? e.op.operationId}</span>
                      <span className="mono small muted api-link-path">{shortPath(e.path)}</span>
                    </span>
                  </button>
                ))}
              </div>
            ))}
            {!groups.length && <span className="muted small">No endpoint matches.</span>}
          </nav>
          <div className="table-view">
            {selected ? (
              <EndpointConsole
                key={selected.id}
                spec={spec.data}
                endpoint={selected}
                catalog={catalog.data}
                catalogLoading={catalog.loading}
                onTargetOrg={setCatalogOrg}
              />
            ) : (
              <Empty title="No endpoints" />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
