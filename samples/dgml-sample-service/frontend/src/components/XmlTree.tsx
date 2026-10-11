// A collapsible view of a DGML XML document. Elements carrying dg:origin are
// clickable: selecting one highlights where it came from on the page.

import { useMemo, useState } from "react";
import type { Location } from "../api";
import { parseOrigin } from "./PageStack";

const DG_NS = "http://dgml.io/ns/dg#";

export function XmlTree({
  xml,
  selected,
  onSelect,
}: {
  xml: string;
  selected: Element | null;
  onSelect: (el: Element, locations: Location[]) => void;
}) {
  const doc = useMemo(() => new DOMParser().parseFromString(xml, "application/xml"), [xml]);
  const error = doc.getElementsByTagName("parsererror")[0];
  if (error) return <pre className="xml-raw">{xml}</pre>;
  return (
    <div className="xml">
      <XmlElement el={doc.documentElement} depth={0} selected={selected} onSelect={onSelect} />
    </div>
  );
}

function attrs(el: Element) {
  return Array.from(el.attributes).map((a) => (
    <span key={a.name}>
      {" "}
      <span className="xml-attr">{a.name}</span>=<span className="xml-val">"{a.value}"</span>
    </span>
  ));
}

function XmlElement({
  el,
  depth,
  selected,
  onSelect,
}: {
  el: Element;
  depth: number;
  selected: Element | null;
  onSelect: (el: Element, locations: Location[]) => void;
}) {
  const [open, setOpen] = useState(depth < 3);
  const children = Array.from(el.childNodes).filter(
    (n) => n.nodeType === Node.ELEMENT_NODE || (n.nodeType === Node.TEXT_NODE && n.textContent?.trim()),
  );
  const origin = parseOrigin(el.getAttributeNS(DG_NS, "origin"));
  const clickable = origin.length > 0;
  const isSelected = selected === el;
  const onlyText = children.length === 1 && children[0].nodeType === Node.TEXT_NODE;
  const tagClass = `xml-tag${clickable ? " xml-clickable" : ""}${isSelected ? " xml-selected" : ""}`;
  const select = clickable ? () => onSelect(el, origin) : undefined;

  if (children.length === 0) {
    return (
      <div className="xml-line">
        <span className={tagClass} onClick={select}>
          &lt;{el.tagName}
          {attrs(el)}/&gt;
        </span>
      </div>
    );
  }
  if (onlyText) {
    return (
      <div className="xml-line">
        <span className={tagClass} onClick={select}>
          &lt;{el.tagName}
          {attrs(el)}&gt;
        </span>
        <span className="xml-text">{children[0].textContent?.trim()}</span>
        <span className="xml-tag">&lt;/{el.tagName}&gt;</span>
      </div>
    );
  }
  return (
    <div className="xml-node">
      <div className="xml-line">
        <button className="xml-toggle" onClick={() => setOpen((v) => !v)} aria-label={open ? "Collapse" : "Expand"}>
          {open ? "▾" : "▸"}
        </button>
        <span className={tagClass} onClick={select}>
          &lt;{el.tagName}
          {attrs(el)}&gt;
        </span>
        {!open && <span className="muted"> … {children.length} child(ren) &lt;/{el.tagName}&gt;</span>}
      </div>
      {open && (
        <>
          <div className="xml-children">
            {children.map((n, i) =>
              n.nodeType === Node.ELEMENT_NODE ? (
                <XmlElement key={i} el={n as Element} depth={depth + 1} selected={selected} onSelect={onSelect} />
              ) : (
                <div key={i} className="xml-line xml-text">
                  {n.textContent?.trim()}
                </div>
              ),
            )}
          </div>
          <div className="xml-line">
            <span className="xml-tag">&lt;/{el.tagName}&gt;</span>
          </div>
        </>
      )}
    </div>
  );
}
