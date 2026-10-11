// Inputs for the APIs page: a choice from the data (with a custom value), a parameter
// input, and a request-body field input per field kind.

import { useState } from "react";
import type { Option } from "./catalog";
import type { FieldState } from "./form";
import { kindOf, typeLabel, unwrapNullable } from "./openapi";
import type { BodyField, Schema } from "./openapi";

const CUSTOM = "__custom__";

/** A choice from the data, with "Custom…" to type anything else. */
function Choice({
  value,
  options,
  onChange,
  allowEmpty,
  placeholder,
}: {
  value: string;
  options: Option[];
  onChange: (v: string) => void;
  allowEmpty?: boolean;
  placeholder?: string;
}) {
  const known = options.some((o) => o.value === value);
  const [custom, setCustom] = useState(!known && value !== "");
  const showCustom = custom || (!known && value !== "");
  return (
    <div className="api-choice">
      <select
        value={showCustom ? CUSTOM : value}
        onChange={(e) => {
          if (e.target.value === CUSTOM) {
            setCustom(true);
          } else {
            setCustom(false);
            onChange(e.target.value);
          }
        }}
      >
        {(allowEmpty || (!known && !showCustom)) && <option value="">{placeholder ?? "—"}</option>}
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
            {o.hint ? `  ·  ${o.hint}` : ""}
          </option>
        ))}
        <option value={CUSTOM}>Custom value…</option>
      </select>
      {showCustom && (
        <input className="mono" value={value} autoFocus placeholder="type a value" onChange={(e) => onChange(e.target.value)} />
      )}
      {!showCustom && value && options.length > 0 && (
        <span className="muted small mono api-choice-value" title={value}>
          {value}
        </span>
      )}
    </div>
  );
}

function FieldLabel({
  name,
  location,
  required,
  schema,
}: {
  name: string;
  location: string;
  required: boolean;
  schema: Schema;
}) {
  return (
    <span className="api-label">
      <span className="mono strong">{name}</span>
      {required && <span className="danger">*</span>}
      <span className="muted small">
        {typeLabel(schema)} · {location}
      </span>
    </span>
  );
}

export function ParamInput({
  name,
  location,
  required,
  schema,
  value,
  options,
  onChange,
}: {
  name: string;
  location: string;
  required: boolean;
  schema: Schema;
  value: string;
  options: Option[] | null;
  onChange: (v: string) => void;
}) {
  const kind = kindOf(schema);
  let input;
  if (kind === "bool") {
    input = (
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {!required && <option value="">(omit)</option>}
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    );
  } else if (options && options.length) {
    input = (
      <Choice
        value={value}
        options={options}
        onChange={onChange}
        allowEmpty={!required}
        placeholder={required ? "Choose…" : "(omit)"}
      />
    );
  } else {
    input = (
      <input
        className="mono"
        type={kind === "number" ? "number" : "text"}
        value={value}
        placeholder={options ? "nothing to choose from — type a value" : required ? "required" : "(omit)"}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }
  return (
    <label className="field api-field">
      <FieldLabel name={name} location={location} required={required} schema={schema} />
      {input}
    </label>
  );
}

export function BodyFieldInput({
  field,
  state,
  options,
  onChange,
}: {
  field: BodyField;
  state: FieldState;
  options: Option[] | null;
  onChange: (patch: Partial<FieldState>) => void;
}) {
  const kind = kindOf(field.schema);
  const { nullable } = unwrapNullable(field.schema);
  let input;
  switch (kind) {
    case "bool":
      input = (
        <select value={state.value} onChange={(e) => onChange({ value: e.target.value })}>
          <option value="true">true</option>
          <option value="false">false</option>
        </select>
      );
      break;
    case "number":
      input = <input type="number" value={state.value} onChange={(e) => onChange({ value: e.target.value })} />;
      break;
    case "string":
      input =
        options && options.length ? (
          <Choice value={state.value} options={options} onChange={(v) => onChange({ value: v })} allowEmpty={nullable} placeholder={nullable ? "null" : "Choose…"} />
        ) : field.name === "text" ? (
          <textarea className="code" rows={8} value={state.value} onChange={(e) => onChange({ value: e.target.value })} />
        ) : (
          <input value={state.value} onChange={(e) => onChange({ value: e.target.value })} />
        );
      break;
    case "list":
      input =
        options && options.length ? (
          <div className="api-checks">
            {options.map((o) => (
              <label key={o.value} className="check">
                <input
                  type="checkbox"
                  checked={state.list.includes(o.value)}
                  onChange={(e) =>
                    onChange({
                      list: e.target.checked ? [...state.list, o.value] : state.list.filter((v) => v !== o.value),
                    })
                  }
                />
                <span>{o.label}</span>
                {o.hint && <span className="muted small">{o.hint}</span>}
              </label>
            ))}
          </div>
        ) : (
          <>
            <textarea
              rows={3}
              value={state.list.join("\n")}
              placeholder="one per line"
              onChange={(e) => onChange({ list: e.target.value.split("\n") })}
            />
            <small className="muted">One item per line.</small>
          </>
        );
      break;
    case "map":
      input = (
        <div className="stack-tight">
          {state.map.map(([k, v], i) => (
            <div className="row" key={i}>
              <input className="mono" value={k} placeholder="key" onChange={(e) => onChange({ map: state.map.map((p, j) => (j === i ? [e.target.value, p[1]] : p)) })} />
              <input
                value={v}
                type={field.name === "secrets" ? "password" : "text"}
                placeholder={field.name === "secrets" ? "new value (\"\" clears)" : "value"}
                onChange={(e) => onChange({ map: state.map.map((p, j) => (j === i ? [p[0], e.target.value] : p)) })}
              />
              <button type="button" className="btn-link" onClick={() => onChange({ map: state.map.filter((_, j) => j !== i) })}>
                remove
              </button>
            </div>
          ))}
          <button type="button" className="btn-link" onClick={() => onChange({ map: [...state.map, ["", ""]] })}>
            + add entry
          </button>
        </div>
      );
      break;
    case "files":
      input = (
        <>
          <input type="file" multiple accept=".pdf,.docx,.xlsx" onChange={(e) => onChange({ files: Array.from(e.target.files ?? []) })} />
          {state.files.length > 0 && <small className="muted">{state.files.map((f) => f.name).join(", ")}</small>}
        </>
      );
      break;
    default:
      input = <textarea className="code" rows={4} value={state.value} onChange={(e) => onChange({ value: e.target.value })} />;
  }
  return (
    <div className={`field api-field${state.include ? "" : " api-field-off"}`}>
      <div className="row between">
        <FieldLabel name={field.name} location="body" required={field.required} schema={field.schema} />
        {!field.required && (
          <label className="check small">
            <input type="checkbox" checked={state.include} onChange={(e) => onChange({ include: e.target.checked })} />
            <span>send</span>
          </label>
        )}
      </div>
      {input}
    </div>
  );
}
