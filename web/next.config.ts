import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Memoizes components and JSX automatically (babel-plugin-react-compiler). It is what keeps
  // typing fast: the message list isn't re-rendered (Markdown + KaTeX) on every keystroke.
  reactCompiler: true,
};

export default nextConfig;
