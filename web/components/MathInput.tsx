"use client";

// A plain-text input that renders $...$ and $$...$$ with KaTeX, except the formula the
// cursor is inside, which is shown as raw LaTeX so it can be edited.
//
// The source of truth is a plain string plus a selection (offsets into that string). The
// DOM is regenerated from them: text segments become text nodes, rendered formulas become
// non-editable <span data-src="$...$"> atoms. After the browser edits a text node (typing),
// the string is read back from the DOM and the DOM is regenerated. Edits the browser would
// get wrong next to atoms (Enter, paste, copy/cut, undo, Backspace/arrows at an atom) are
// done on the string directly.

import katex from "katex";
import "katex/dist/katex.min.css";
import { useEffect, useEffectEvent, useRef } from "react";

type Seg = { start: number; end: number; math: boolean; display: boolean };
type Sel = [number, number]; // anchor, focus
type Snapshot = { text: string; sel: Sel };

// Display math first; inline math must not start or end with a space and must not be
// followed by a digit, so "costs $5 and $10" stays text (pandoc's rule).
const DISPLAY = /\$\$((?:\\[\s\S]|[^$\\])+?)\$\$/y;
const INLINE = /\$(?!\s)((?:\\[\s\S]|[^$\\])+?)\$(?!\d)/y;

export function parse(text: string): Seg[] {
  const segs: Seg[] = [];
  let i = 0;
  let textStart = 0;
  while (i < text.length) {
    if (text[i] === "\\") {
      i += 2; // escaped char, e.g. \$
      continue;
    }
    if (text[i] === "$") {
      DISPLAY.lastIndex = i;
      let m = DISPLAY.exec(text);
      const display = !!m;
      if (!m) {
        INLINE.lastIndex = i;
        m = INLINE.exec(text);
        if (m && /\s$/.test(m[1])) m = null;
      }
      if (m) {
        if (textStart < i) segs.push({ start: textStart, end: i, math: false, display: false });
        const end = i + m[0].length;
        segs.push({ start: i, end, math: true, display });
        i = textStart = end;
        continue;
      }
    }
    i++;
  }
  if (textStart < text.length) segs.push({ start: textStart, end: text.length, math: false, display: false });
  return segs;
}

/** A formula is shown raw when any selection end is strictly inside it. */
const isRaw = (s: Seg, sel: Sel | null) => !!sel && sel.some((p) => p > s.start && p < s.end);
const rawKey = (segs: Seg[], sel: Sel | null) =>
  segs.filter((s) => s.math && isRaw(s, sel)).map((s) => s.start).join(",");

// --- DOM <-> string ------------------------------------------------------------------

const isAtom = (n: Node): n is HTMLElement => n instanceof HTMLElement && n.dataset.src !== undefined;
const isSentinel = (n: Node) => n instanceof HTMLBRElement && n.dataset.sentinel !== undefined;

function length(n: Node): number {
  if (n.nodeType === Node.TEXT_NODE) return (n as Text).data.length;
  if (isAtom(n)) return n.dataset.src!.length;
  if (n instanceof HTMLBRElement) return isSentinel(n) ? 0 : 1;
  let total = 0;
  n.childNodes.forEach((c) => (total += length(c)));
  return total;
}

function textOf(n: Node): string {
  if (n.nodeType === Node.TEXT_NODE) return (n as Text).data;
  if (isAtom(n)) return n.dataset.src!;
  if (n instanceof HTMLBRElement) return isSentinel(n) ? "" : "\n";
  let out = "";
  n.childNodes.forEach((c) => (out += textOf(c)));
  return out;
}

/** String offset of a DOM position (node, offset) inside root. */
function offsetOf(root: Node, node: Node, offset: number): number {
  let pos = 0;
  const walk = (n: Node): number | null => {
    if (n === node) {
      if (n.nodeType === Node.TEXT_NODE) return pos + offset;
      if (isAtom(n)) return pos + (offset > 0 ? length(n) : 0);
      for (let k = 0; k < offset && k < n.childNodes.length; k++) pos += length(n.childNodes[k]);
      return pos;
    }
    if (isAtom(n)) {
      if (n.contains(node)) return pos + length(n); // inside a rendered formula: its end
      pos += length(n);
      return null;
    }
    if (n.nodeType === Node.TEXT_NODE || n instanceof HTMLBRElement) {
      pos += length(n);
      return null;
    }
    for (const c of Array.from(n.childNodes)) {
      const r = walk(c);
      if (r !== null) return r;
    }
    return null;
  };
  return walk(root) ?? length(root);
}

/** DOM position for a string offset (root's children are flat: text nodes and atoms). */
function locate(root: HTMLElement, pos: number): [Node, number] {
  const kids = Array.from(root.childNodes);
  let p = 0;
  for (let k = 0; k < kids.length; k++) {
    const c = kids[k];
    if (isSentinel(c)) break;
    const l = length(c);
    if (c.nodeType === Node.TEXT_NODE && pos <= p + l) return [c, pos - p];
    if (isAtom(c) && pos === p) return [root, k];
    p += l;
    if (isAtom(c) && pos === p && kids[k + 1]?.nodeType !== Node.TEXT_NODE) return [root, k + 1];
  }
  const sentinel = kids.findIndex(isSentinel);
  return [root, sentinel === -1 ? kids.length : sentinel];
}

function render(root: HTMLElement, text: string, sel: Sel | null) {
  root.textContent = "";
  let pending = ""; // consecutive raw text, merged into one text node
  const flush = () => {
    if (pending) root.appendChild(document.createTextNode(pending));
    pending = "";
  };
  for (const s of parse(text)) {
    const src = text.slice(s.start, s.end);
    if (!s.math || isRaw(s, sel)) {
      pending += src;
      continue;
    }
    flush();
    const span = document.createElement("span");
    span.className = "mi-math rounded-sm px-px hover:bg-primary/10"; // .mi-math: see onMouseDown
    span.contentEditable = "false";
    span.dataset.src = src;
    span.title = src;
    span.innerHTML = katex.renderToString(s.display ? src.slice(2, -2) : src.slice(1, -1), {
      throwOnError: false,
      strict: "ignore",
    });
    root.appendChild(span);
  }
  flush();
  const br = document.createElement("br"); // keeps a trailing newline / empty input visible
  br.dataset.sentinel = "";
  root.appendChild(br);
  root.dataset.empty = String(text.length === 0);
}

// --- component -----------------------------------------------------------------------

type Props = {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  placeholder?: string;
};

export default function MathInput({ value, onChange, onSubmit, placeholder }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const model = useRef<Snapshot>({ text: "", sel: [0, 0] });
  const history = useRef({ undo: [] as Snapshot[], redo: [] as Snapshot[], last: 0 });
  const composing = useRef(false);

  const focused = () => document.activeElement === ref.current;

  function readSel(): Sel | null {
    const s = window.getSelection();
    const root = ref.current;
    if (!root || !s || !s.rangeCount || !s.anchorNode || !s.focusNode) return null;
    if (!root.contains(s.anchorNode) || !root.contains(s.focusNode)) return null;
    return [offsetOf(root, s.anchorNode, s.anchorOffset), offsetOf(root, s.focusNode, s.focusOffset)];
  }

  function paint(text: string, sel: Sel) {
    const root = ref.current!;
    model.current = { text, sel };
    render(root, text, focused() ? sel : null);
    if (!focused()) return;
    const s = window.getSelection()!;
    const [an, ao] = locate(root, sel[0]);
    const [fn, fo] = locate(root, sel[1]);
    s.setBaseAndExtent(an, ao, fn, fo);
  }

  /** Apply a change to the string; `typing` coalesces quick edits into one undo step. */
  function commit(text: string, sel: Sel, typing = false) {
    const prev = model.current;
    if (text !== prev.text) {
      const h = history.current;
      const now = Date.now();
      if (!typing || now - h.last > 600 || !h.undo.length) h.undo.push(prev);
      if (h.undo.length > 200) h.undo.shift();
      h.redo = [];
      h.last = now;
    }
    paint(text, sel);
    if (text !== prev.text) onChange(text);
  }

  function replaceSelection(str: string) {
    const { text } = model.current;
    const sel = readSel() ?? (model.current.sel[0] < 0 ? [text.length, text.length] : model.current.sel);
    const [a, b] = [Math.min(...sel), Math.max(...sel)];
    const p = a + str.length;
    commit(text.slice(0, a) + str + text.slice(b), [p, p]);
  }

  function step(dir: "undo" | "redo") {
    const h = history.current;
    const snap = (dir === "undo" ? h.undo : h.redo).pop();
    if (!snap) return;
    (dir === "undo" ? h.redo : h.undo).push(model.current);
    h.last = 0;
    paint(snap.text, snap.sel);
    onChange(snap.text);
  }

  // Initial render, and external changes (e.g. cleared after sending).
  useEffect(() => {
    if (value === model.current.text && ref.current!.childNodes.length) return;
    if (value === "") history.current = { undo: [], redo: [], last: 0 };
    paint(value, [value.length, value.length]);
  }, [value]);

  // Moving the cursor into or out of a formula switches it between raw and rendered.
  useEffect(() => {
    const check = (force = false) => {
      if (composing.current || !focused()) return;
      const sel = readSel();
      if (!sel) return;
      const { text, sel: old } = model.current;
      const collapsed = sel[0] === sel[1];
      const segs = parse(text);
      if ((collapsed || force) && rawKey(segs, sel) !== rawKey(segs, old)) paint(text, sel);
      else model.current = { text, sel };
    };
    const onSelection = () => check();
    const onMouseUp = () => check(true); // after a drag-selection ends
    document.addEventListener("selectionchange", onSelection);
    document.addEventListener("mouseup", onMouseUp);
    return () => {
      document.removeEventListener("selectionchange", onSelection);
      document.removeEventListener("mouseup", onMouseUp);
    };
  }, []);

  const syncFromDom = () => {
    const root = ref.current!;
    const text = textOf(root);
    const sel = readSel() ?? [text.length, text.length];
    commit(text, sel, true);
  };

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.nativeEvent.isComposing) return;
    const mod = e.metaKey || e.ctrlKey;
    if (e.key === "Enter") {
      e.preventDefault();
      if (e.shiftKey) replaceSelection("\n");
      else onSubmit();
      return;
    }
    if (mod && e.key.toLowerCase() === "z") {
      e.preventDefault();
      step(e.shiftKey ? "redo" : "undo");
      return;
    }
    if (mod && e.key.toLowerCase() === "y") {
      e.preventDefault();
      step("redo");
      return;
    }
    // Next to a rendered formula, these keys step inside it instead of jumping over or
    // deleting the whole thing.
    // Read the live selection: selectionchange (which updates the model) fires
    // asynchronously, so with fast key repeat the model can be one keystroke behind.
    const text = model.current.text;
    const sel = readSel() ?? model.current.sel;
    if (sel[0] !== sel[1] || e.shiftKey || e.altKey || mod) return;
    const c = sel[0];
    const segs = parse(text).filter((s) => s.math);
    const enter = (pos: number) => {
      e.preventDefault();
      paint(text, [pos, pos]);
    };
    if ((e.key === "ArrowLeft" || e.key === "Backspace") && segs.some((s) => s.end === c)) enter(c - 1);
    else if ((e.key === "ArrowRight" || e.key === "Delete") && segs.some((s) => s.start === c)) enter(c + 1);
  }

  // A native listener: React's onBeforeInput is a synthetic event without inputType.
  const onBeforeInput = useEffectEvent((e: InputEvent) => {
    const type = e.inputType ?? "";
    if (type === "historyUndo" || type === "historyRedo") {
      e.preventDefault();
      step(type === "historyUndo" ? "undo" : "redo");
    } else if (type === "insertParagraph" || type === "insertLineBreak") {
      e.preventDefault();
      replaceSelection("\n");
    } else if (type.startsWith("format") || type === "insertFromDrop") {
      e.preventDefault();
    }
  });
  useEffect(() => {
    const root = ref.current!;
    const listener = (e: Event) => onBeforeInput(e as InputEvent);
    root.addEventListener("beforeinput", listener);
    return () => root.removeEventListener("beforeinput", listener);
  }, []);

  function onMouseDown(e: React.MouseEvent) {
    const atom = (e.target as HTMLElement).closest?.(".mi-math");
    if (!atom || !ref.current) return;
    e.preventDefault(); // clicking a formula: put the cursor before its closing $
    const start = offsetOf(ref.current, ref.current, Array.from(ref.current.childNodes).indexOf(atom));
    const end = start + length(atom);
    ref.current.focus();
    paint(model.current.text, [end - 1, end - 1]);
  }

  function onCopy(e: React.ClipboardEvent, cut = false) {
    const { text, sel } = model.current;
    const [a, b] = [Math.min(...sel), Math.max(...sel)];
    if (a === b) return;
    e.preventDefault();
    e.clipboardData.setData("text/plain", text.slice(a, b)); // the LaTeX, not KaTeX's HTML
    if (cut) replaceSelection("");
  }

  return (
    <div
      ref={ref}
      // Grows with its content up to 60% of the window, then scrolls; the corner handle
      // resizes it by hand. 22px line + 2×8px padding + 2px border = 40px, the button height
      // (daisyUI's btn and textarea are both 2.5rem fields).
      className={[
        "math-input textarea relative max-h-[60vh] min-h-10 w-auto flex-1 cursor-text resize-y overflow-y-auto",
        "whitespace-pre-wrap wrap-anywhere px-3 py-2 text-[15px] leading-[22px]",
      ].join(" ")}
      role="textbox"
      aria-multiline="true"
      contentEditable="plaintext-only"
      spellCheck={false}
      data-placeholder={placeholder}
      onInput={() => !composing.current && syncFromDom()}
      onCompositionStart={() => (composing.current = true)}
      onCompositionEnd={() => {
        composing.current = false;
        syncFromDom();
      }}
      onKeyDown={onKeyDown}
      onMouseDown={onMouseDown}
      onPaste={(e) => {
        e.preventDefault();
        replaceSelection(e.clipboardData.getData("text/plain"));
      }}
      onCopy={(e) => onCopy(e)}
      onCut={(e) => onCopy(e, true)}
      onBlur={() => {
        // No cursor: render every formula. Forget the selection so that on refocus the
        // selectionchange check compares against what is actually on screen.
        render(ref.current!, model.current.text, null);
        model.current = { text: model.current.text, sel: [-1, -1] };
      }}
    />
  );
}
