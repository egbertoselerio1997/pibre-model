# How to Mark Changes in LaTeX Journal Revisions

This guide outlines a simple, low-effort method for highlighting revised text and figures in LaTeX documents using custom markup. 

## 1. Marking Text Changes

To highlight new or modified text, you can define a custom command that changes the text color (e.g., to blue). 

### Preamble Setup
Ensure you have a color package imported, then define the `\rev` command in your document's preamble:

```latex
\usepackage{color} % or \usepackage{xcolor}
\newcommand{\rev}[1]{\textcolor{blue}{#1}}
```

### Usage in Document
Wrap any new or changed text inside the `\rev{}` command while writing:

```latex
Our approach outperforms all other approaches by 100%.
\rev{This sentence represents newly added or modified text.}
Hence, the world will be changed after this manuscript is accepted.
```

### Preparing the Final Version
When you are ready to submit the final camera-ready version, you do not need to manually remove the `\rev{}` tags throughout the document. Simply update the command in your preamble to render the text in black:

```latex
\newcommand{\rev}[1]{\textcolor{black}{#1}}
```

---

## 2. Marking Figure Changes

Changing text color does not work for figures. Instead, you can create a custom environment that draws blue lines above and below a modified figure to indicate changes.

### Preamble Setup
Import the `mdframed` and `ntheorem` packages, and define a new `figrev` environment in your preamble:

```latex
\usepackage{mdframed}
\usepackage{ntheorem}

\theoremstyle{nonumberplain}
\newmdtheoremenv[%
  linecolor=blue,
  linewidth=2pt,
  rightline=false,
  leftline=false]{figrev}{}
```

### Usage in Document
Wrap your figure inclusions (or the specific changed elements) inside the `\begin{figrev}` and `\end{figrev}` tags:

```latex
\begin{figrev}
  \includegraphics[width=\linewidth]{img/updated_figure.pdf}
\end{figrev}
```

### Preparing the Final Version
Unlike the text markup, the `figrev` environment tags must be manually deleted from the document before compiling your final camera-ready version (unless you implement conditional compilation using a package like `ifthen`).