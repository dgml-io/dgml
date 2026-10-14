// The document's rendered pages with bounding boxes drawn over them.
//
// DGML locations (dg:origin) are integer pixels in the page image DGML rendered —
// page_images/page_N.png, top-left origin — so an overlay is just that box scaled by
// the image's displayed size over its natural size.

import { useEffect, useRef, useState } from "react";
import type { Location } from "../api";

export interface Box extends Location {
  key: string;
  label?: string;
}

export function PageStack({
  pageCount,
  pageUrl,
  boxes,
  selected,
  onSelect,
}: {
  pageCount: number;
  pageUrl: (page: number) => string;
  boxes: Box[];
  selected: Box[];
  onSelect?: (box: Box) => void;
}) {
  const refs = useRef<Record<number, HTMLDivElement | null>>({});
  const firstSelected = selected[0];

  useEffect(() => {
    if (!firstSelected) return;
    refs.current[firstSelected.page_number]?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [firstSelected]);

  if (pageCount <= 0) {
    return <div className="muted">No page images — rendering failed or the file has no pages.</div>;
  }
  const selectedKeys = new Set(selected.map((b) => b.key));
  return (
    <div className="pages">
      {Array.from({ length: pageCount }, (_, i) => i + 1).map((page) => (
        <div key={page} ref={(el) => (refs.current[page] = el)} className="page-frame">
          <div className="page-label muted small">Page {page}</div>
          <PageImage
            src={pageUrl(page)}
            boxes={boxes.filter((b) => b.page_number === page && !selectedKeys.has(b.key))}
            selected={selected.filter((b) => b.page_number === page)}
            onSelect={onSelect}
          />
        </div>
      ))}
    </div>
  );
}

function PageImage({
  src,
  boxes,
  selected,
  onSelect,
}: {
  src: string;
  boxes: Box[];
  selected: Box[];
  onSelect?: (box: Box) => void;
}) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const style = (b: Box) => {
    if (!size) return { display: "none" };
    const [x1, y1, x2, y2] = b.bounding_box;
    return {
      left: `${(x1 / size.w) * 100}%`,
      top: `${(y1 / size.h) * 100}%`,
      width: `${((x2 - x1) / size.w) * 100}%`,
      height: `${((y2 - y1) / size.h) * 100}%`,
    };
  };
  return (
    <div className="page-image">
      <img
        src={src}
        loading="lazy"
        alt=""
        onLoad={(e) => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
      />
      {boxes.map((b) => (
        <button
          key={b.key}
          className="box"
          style={style(b)}
          title={b.label}
          onClick={() => onSelect?.(b)}
        />
      ))}
      {selected.map((b) => (
        <div key={`sel-${b.key}`} className="box box-selected" style={style(b)} title={b.label} />
      ))}
    </div>
  );
}

/** Parse a dg:origin attribute: "page x1 y1 x2 y2; page x1 y1 x2 y2". */
export function parseOrigin(origin: string | null): Location[] {
  if (!origin) return [];
  const out: Location[] = [];
  for (const part of origin.split(";")) {
    const n = part.trim().split(/\s+/).map(Number);
    if (n.length === 5 && n.every(Number.isFinite)) {
      out.push({ page_number: n[0], bounding_box: [n[1], n[2], n[3], n[4]] });
    }
  }
  return out;
}
