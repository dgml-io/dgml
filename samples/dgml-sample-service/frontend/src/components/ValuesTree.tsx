// The extracted values tree: `{tag: {text, value?, locations}}`, nested for groups
// and arrays for collections. Clicking a leaf selects it (and its page boxes).

import type { ValueLeaf, ValueNode } from "../api";
import type { Box } from "./PageStack";

export const isLeaf = (node: ValueNode): node is ValueLeaf =>
  !Array.isArray(node) && typeof node === "object" && node !== null && "text" in node;

/** Every leaf's boxes, keyed by its path — what the page overlay draws. */
export function collectBoxes(values: Record<string, ValueNode>): Box[] {
  const out: Box[] = [];
  const walk = (node: ValueNode, path: string) => {
    if (isLeaf(node)) {
      (node.locations ?? []).forEach((loc, i) =>
        out.push({ ...loc, key: `${path}#${i}`, label: `${path}: ${node.text}` }),
      );
    } else if (Array.isArray(node)) {
      node.forEach((child, i) => walk(child, `${path}[${i}]`));
    } else {
      for (const [tag, child] of Object.entries(node)) walk(child, path ? `${path}.${tag}` : tag);
    }
  };
  walk(values, "");
  return out;
}

export function ValuesTree({
  values,
  selectedPath,
  onSelect,
}: {
  values: Record<string, ValueNode>;
  selectedPath: string | null;
  onSelect: (path: string) => void;
}) {
  const entries = Object.entries(values);
  if (entries.length === 0) return <div className="muted">The extraction found no values.</div>;
  return (
    <div className="values">
      {entries.map(([tag, node]) => (
        <ValueRow key={tag} tag={tag} node={node} path={tag} selectedPath={selectedPath} onSelect={onSelect} />
      ))}
    </div>
  );
}

function ValueRow({
  tag,
  node,
  path,
  selectedPath,
  onSelect,
}: {
  tag: string;
  node: ValueNode;
  path: string;
  selectedPath: string | null;
  onSelect: (path: string) => void;
}) {
  if (isLeaf(node)) {
    const pages = [...new Set((node.locations ?? []).map((l) => l.page_number))];
    const typed = node.value !== undefined && node.value !== node.text ? node.value : null;
    return (
      <button
        className={`value-leaf${selectedPath === path ? " value-selected" : ""}`}
        onClick={() => onSelect(path)}
      >
        <span className="value-tag">{tag}</span>
        <span className="value-text">{node.text || <em className="muted">empty</em>}</span>
        {typed && <span className="value-typed mono">= {typed}</span>}
        <span className="value-meta muted small">
          {node.computed ? "computed" : pages.length ? `p. ${pages.join(", ")}` : "no location"}
        </span>
      </button>
    );
  }
  const children: [string, ValueNode, string][] = Array.isArray(node)
    ? node.map((c, i) => [`#${i + 1}`, c, `${path}[${i}]`])
    : Object.entries(node).map(([t, c]) => [t, c, `${path}.${t}`]);
  return (
    <details className="value-group" open>
      <summary>
        <span className="value-tag">{tag}</span>
        {Array.isArray(node) && <span className="muted small"> {node.length} item(s)</span>}
      </summary>
      <div className="value-children">
        {children.map(([t, c, p]) => (
          <ValueRow key={p} tag={t} node={c} path={p} selectedPath={selectedPath} onSelect={onSelect} />
        ))}
      </div>
    </details>
  );
}
