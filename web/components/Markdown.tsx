import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";

// remark-math understands $...$ and $$...$$; models also write \( \) and \[ \].
function normalizeMath(text: string): string {
  return text
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, m) => `\n$$\n${m.trim()}\n$$\n`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, m) => `$${m.trim()}$`);
}

/**
 * Markdown (GitHub flavour) with LaTeX math rendered by KaTeX.
 * `inline`: for short text inside other elements (e.g. a link): no <p> wrapper, no nested links.
 */
export default function Markdown({ text, inline = false }: { text: string; inline?: boolean }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: "ignore" }]]}
      disallowedElements={inline ? ["a"] : undefined}
      unwrapDisallowed
      components={{
        a: (props) => <a {...props} target="_blank" rel="noreferrer" />,
        ...(inline && { p: ({ children }) => <>{children}</> }),
      }}
    >
      {normalizeMath(text)}
    </ReactMarkdown>
  );
}
