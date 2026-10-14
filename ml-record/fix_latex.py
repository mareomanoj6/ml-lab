
import re

def fix_spacing():
    with open('main.tex', 'r') as f:
        content = f.read()

    # 1. Tasks -> Code
    # Replace \end{itemize} and any following whitespace with \end{itemize}\n\n\vspace{\baselineskip}\n\n
    content = re.sub(r'\\end\{itemize\}\s*', r'\\end{itemize}\n\n\\vspace{\\baselineskip}\n\n', content)

    # 2. Code -> Output
    # Find \end{lstlisting} followed by optional whitespace and \noindent \textbf{Output:}
    content = re.sub(r'(\\end\{lstlisting\})\s*(\\noindent \\textbf\{Output:\})', r'\1\n\n\\vspace{\\baselineskip}\n\n\2', content)

    # 3. Output -> Result
    # Find \end{lstlisting} followed by optional whitespace and \noindent \textbf{Result:}
    content = re.sub(r'(\\end\{lstlisting\})\s*(\\noindent \\textbf\{Result:\})', r'\1\n\n\\vspace{\\baselineskip}\n\n\2', content)

    # Final cleanup: Remove accidental double vspaces
    while r'\vspace{\baselineskip}' + '\n\n' + r'\vspace{\baselineskip}' in content:
        content = content.replace(r'\vspace{\baselineskip}' + '\n\n' + r'\vspace{\baselineskip}', r'\vspace{\baselineskip}')

    with open('main.tex', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    fix_spacing()
