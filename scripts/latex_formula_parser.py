"""
Zero-trust LaTeX formula parser for TEV audit (v2 - improved).

Parses equation environments from .tex files WITHOUT relying on @tev tags.
Extracts: symbolic formula, numerical values, labels, and dependencies.

Design principle: the script derives everything from raw LaTeX text.
No trust in embedded metadata — only mathematical content is analyzed.

v2 improvements (over v1):
- Sanitization layer applied to EVERY formula before parsing/eval:
  \text{}, \mathrm{} removed; \mathbf{}, \boldsymbol{} unwrapped;
  \hat{}, \mathcal{}, \check{} etc. unwrapped.
- align/align* environments: '&' replaced by space so each line is a
  complete equation (fixes fragments like "alpha_G &=").
- Matrices masked as single tokens before splitting on '\\\\'.
- \implies treated as = for parsing purposes.
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


MATRIX_ENVS = ('pmatrix', 'bmatrix', 'Bmatrix', 'vmatrix', 'Vmatrix',
               'smallmatrix')

# Sentinel for the row separator inside masked matrices.
MATRIX_ROW = '\x00ROW\x00'
RELATION_PATTERN = re.compile(
    r'(?<!\\)(?:>=|<=|>|<)|\\(?:geq?|leq?|neq|ne|ll|gg)\b'
)


@dataclass
class Formula:
    """A parsed formula with its context."""
    id: str                      # unique identifier (auto-generated)
    label: Optional[str]         # \label{...} if present
    equation_env: bool           # inside equation environment
    formula_text: str            # raw LaTeX formula text
    numerical_value: Optional[float]  # numeric value stated in article (if any)
    unit: Optional[str]          # unit of the value (e.g. deg, eV, meV)
    dependencies: List[str]      # variable names appearing in formula
    line_no: int                 # line number in source file
    section: str                 # current section name
    lhs_variable: Optional[str] = None  # LHS variable (excluded from deps)
    is_matrix: bool = False              # contains a matrix environment
    matrix_rows: List[List[str]] = field(default_factory=list)  # cells per row
    source_file: Optional[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class LaTeXFormulaParser:
    """Parses equation environments from LaTeX source files."""

    EQUATION_ENV_PATTERN = re.compile(
        r'\\begin\{equation\}(.*?)\\end\{equation\}',
        re.DOTALL
    )

    ALIGN_ENV_PATTERN = re.compile(
        r'\\begin\{(align|align\*)\}(.*?)\\end\{\1\}',
        re.DOTALL
    )

    EQUATION_STAR_PATTERN = re.compile(
        r'\\begin\{equation\*\}(.*?)\\end\{equation\*\}',
        re.DOTALL
    )

    LABEL_PATTERN = re.compile(r'\\label\{([^}]*)\}')

    # Matrix environment pattern (any of the standard matrix envs)
    MATRIX_ENV_RE = re.compile(
        r'\\begin\{(pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|smallmatrix)\}'
        r'(.*?)'
        r'\\end\{\1\}',
        re.DOTALL
    )

    # @tev tag pattern - must be stripped for zero-trust parsing
    TEV_TAG_PATTERN = re.compile(r'%\s*@tev\s*\{[^}]*\}')

    def __init__(self):
        self.formulas: List[Formula] = []
        self.variables: Dict[str, List[int]] = {}  # var -> line numbers

    # ------------------------------------------------------------------
    # Sanitization layer (from the reference sample)
    # ------------------------------------------------------------------
    @staticmethod
    def sanitize_latex_math(raw_math: str) -> str:
        """Очищает сырую LaTeX-математику от типографического шума.

        1. Удаляет комментарии TeX
        2. Вырезает текстовые метки и единицы: \text{...}, \mathrm{...}
        3. Снимает шрифтовые обертки: \mathbf{u} -> u, \boldsymbol{\sigma} -> \sigma
        4. Убирает alignment-маркеры таблиц и выравнивания (& -> space)
        5. Изолирует матрицы от разбивки по \\
        """
        text = raw_math.strip()

        # 1. Удаляем комментарии TeX
        text = re.sub(r"%.*$", "", text, flags=re.MULTILINE)

        # Preserve roman text when it names a subscripted or superscripted quantity.
        text = re.sub(r'_\{\\mathrm\{([^{}]*)\}\}', r'_{\1}', text)
        text = re.sub(r'\^\{\\mathrm\{([^{}]*)\}\}', r'^{\1}', text)

        # 2. Вырезаем текстовые метки и единицы: \text{...}, \mathrm{...}, \mbox{...}
        text = re.sub(r"\\(text|mathrm|textnormal|mbox)\{[^}]*\}", "", text)

        # 3. Снимаем шрифтовые обертки: \mathbf{u} -> u, \boldsymbol{\sigma} -> \sigma
        text = re.sub(r"\\(mathbf|boldsymbol|bm|mathit|underline)\{([^}]+)\}", r"\2", text)

        # 3b. \hat{}, \mathcal{}, \check{} etc. -> base char
        text = re.sub(r"\\(hat|mathcal|check|ddot|dot|tilde|bar|breve|grave|acute)\{([^}]+)\}", r"\2", text)

        # 4. Убираем alignment-маркеры таблиц и выравнивания
        text = text.replace("&", " ")
        for op in ['equiv']:
            text = re.sub(r'\\%s(?=[\s=(){}]|$)' % op, '=', text)

        # 4c. \cdot -> * (явное умножение)
        text = text.replace("\\cdot", "*")
        text = text.replace("\\times", "*")

        # 4d. \otimes -> * (тензорное произведение)
        text = text.replace("\\otimes", "*")

        # 4e. \mathbb{} -> unwrap (матрицы, числа)
        text = re.sub(r"\\mathbb\{([^}]*)\}", r"\1", text)

        # 4f. Убираем trailing-знаки: ',', '.' в конце формулы
        text = text.rstrip(',.;')

        # 5. Изолируем матрицы от разбивки по \\
        def mask_matrix(m):
            return m.group(0).replace(r"\\", " ")

        text = re.sub(
            r"\\begin\{(pmatrix|bmatrix|matrix|vmatrix)\}.*?\\end\{\1\}",
            mask_matrix,
            text,
            flags=re.DOTALL,
        )

        # 6. \implies -> = (логическая эквивалентность как равенство)
        text = text.replace("\\implies", "=")

        return text

    # ------------------------------------------------------------------
    # Matrix helpers
    # ------------------------------------------------------------------
    @staticmethod
    def extract_matrix_rows(content: str) -> Tuple[bool, List[List[str]]]:
        """Detect the first matrix environment in `content` and return
        (is_matrix, rows_of_cells). Cells are split on '\\\\' row separators
        (two literal backslash chars in the source)."""
        m = LaTeXFormulaParser.MATRIX_ENV_RE.search(content)
        if not m:
            return False, []
        body = m.group(2)
        # Row separator in the raw LaTeX source is '\\\\' (two literal backslash
        # chars). The pattern r'\\\\' has 4 chars and therefore matches exactly
        # 2 literal backslashes.
        rows_raw = re.split(r'\\\\', body)
        rows = []
        for row in rows_raw:
            cells = [c.strip() for c in row.split('&')]
            cells = [c for c in cells if c != '']
            if cells:
                rows.append(cells)
        return True, rows

    @staticmethod
    def mask_matrices(content: str) -> str:
        """Replace matrix environments by a placeholder so that line-splitting
        on '\\\\' does not break inside matrices.

        The row separator is replaced by a sentinel (MATRIX_ROW) which keeps
        the cell structure intact; it is restored by restore_matrix_rows()
        before the rows are extracted.
        """
        def _sub(m):
            env = m.group(1)
            body = m.group(2)
            # keep cell structure but neutralize row separators
            safe_body = re.sub(r'\\', MATRIX_ROW, body)
            return f"\\begin{{{env}}}{safe_body}\\end{{{env}}}"
        return LaTeXFormulaParser.MATRIX_ENV_RE.sub(_sub, content)

    @staticmethod
    def restore_matrix_rows(content: str) -> str:
        """Restore the row separators that were neutralized by mask_matrices()."""
        return content.replace(MATRIX_ROW, '\\')

    def _split_statements(self, content: str) -> List[str]:
        """Split an equation-environment body into separate statements on
        top-level '\\\\quad' or ';' (outside matrices and braces)."""
        masked = self.mask_matrices(content)
        # protect braced groups so we don't split inside {...}
        parts = []
        depth = 0
        last = 0
        i = 0
        n = len(masked)
        while i < n:
            ch = masked[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            elif depth == 0 and (masked.startswith('\\quad', i) or ch == ';'):
                step = 5 if masked.startswith('\\quad', i) else 1
                parts.append(masked[last:i])
                last = i + step
                i += step
                continue
            i += 1
        parts.append(masked[last:])
        # Keep equalities, relations, and matrices for downstream classification.
        out = []
        for p in parts:
            if ('=' in p or r'\approx' in p
                    or RELATION_PATTERN.search(p) or 'matrix' in p):
                # restore the row separators inside matrices before returning
                out.append(self.restore_matrix_rows(p.strip()))
        return [p for p in out if p]

    def _split_align_lines(self, content: str) -> List[str]:
        """Split align/align* environment body into individual equations.

        v2: each line becomes a complete equation ( '&' -> space ), and the
        whole line is kept as ONE formula (no chain splitting).
        """
        masked = self.mask_matrices(content)
        # Split on the row separator '\\\\' (two literal backslash chars).
        # Matrices are already masked so their internal separators do not
        # trigger a split.
        line_bodies = re.split(r'\\\\', masked)

        lines = []
        for lb in line_bodies:
            if not lb.strip():
                continue
            lb_clean = lb.replace('&', ' ')
            if lb_clean.strip() and (
                    '=' in lb_clean or r'\approx' in lb_clean
                    or RELATION_PATTERN.search(lb_clean)
                    or 'matrix' in lb_clean):
                lines.append(lb_clean)

        return lines

    # ------------------------------------------------------------------
    # Main parsing
    # ------------------------------------------------------------------
    def parse_file(self, filepath: str) -> List[Formula]:
        """Parse all equations from a LaTeX file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        text = ''.join(lines)
        self.formulas = []

        # STRIP @tev tags BEFORE parsing (zero-trust principle)
        text_clean = self.TEV_TAG_PATTERN.sub('', text)

        # Track current section per line
        section_pattern = re.compile(r'\\section\{([^}]*)\}')
        line_sections: List[str] = []
        current_section = ''
        for line in lines:
            m = section_pattern.search(line)
            if m:
                current_section = m.group(1)
            line_sections.append(current_section)

        def make_formula(content: str, start_pos: int, env_name: str):
            """Build a Formula object from an environment body."""
            line_no = text[:start_pos].count('\n') + 1
            section = (line_sections[line_no - 1] if line_no <= len(line_sections) else '')

            label_match = self.LABEL_PATTERN.search(content)
            label = label_match.group(1) if label_match else None

            # Remove \label{...}, \nonumber, \notag (single-escape regex!)
            formula_raw_text = content.strip()
            formula_raw_text = re.sub(r'\\(?:label|nonumber|notag)\s*\{[^}]*\}', '', formula_raw_text)

            # v2: SANITIZE the formula text itself before anything else.
            # This removes \text{}, unwraps \mathbf{} etc., so that both
            # dependency extraction AND evaluation work on clean LaTeX.
            formula_text = self.sanitize_latex_math(formula_raw_text)

            # Matrix detection + cell extraction (restore row separators first)
            formula_matrix = self.restore_matrix_rows(formula_text)
            is_matrix, matrix_rows = self.extract_matrix_rows(formula_matrix)

            # Determine LHS variable (before first '=')
            lhs_var = self._extract_lhs_variable(formula_text)

            deps = self._extract_dependencies(formula_text, lhs_var)

            num_val, unit = self._extract_numerical_value(formula_raw_text)

            return Formula(
                id=f"formula_{len(self.formulas)+1}",
                label=label,
                equation_env=True,
                formula_text=formula_text,
                numerical_value=num_val,
                unit=unit,
                dependencies=deps,
                line_no=line_no,
                section=section,
                lhs_variable=lhs_var,
                is_matrix=is_matrix,
                matrix_rows=matrix_rows,
                source_file=filepath,
            )

        # --- equation / equation* environments: split multi-statement bodies
        for env_pattern in [self.EQUATION_ENV_PATTERN, self.EQUATION_STAR_PATTERN]:
            for match in env_pattern.finditer(text_clean):
                content = match.group(1)
                start_pos = match.start()
                statements = self._split_statements(content)
                if not statements:
                    statements = [content.strip()]
                for st in statements:
                    # v2: sanitize the statement before creating a formula
                    st_clean = self.sanitize_latex_math(st)
                    if st_clean.strip():
                        self.formulas.append(make_formula(st, start_pos, 'equation'))

        # --- align / align* environments: each line is a complete equation
        for match in self.ALIGN_ENV_PATTERN.finditer(text_clean):
            content = match.group(2)
            start_pos = match.start()
            lines = self._split_align_lines(content)
            for lb in lines:
                if not lb.strip():
                    continue
                for statement in self._split_statements(lb):
                    st_clean = self.sanitize_latex_math(statement)
                    if st_clean.strip():
                        self.formulas.append(
                            make_formula(statement, start_pos, 'align'))

        # Build variable -> line mapping
        for f in self.formulas:
            for var in f.dependencies:
                if var not in self.variables:
                    self.variables[var] = []
                self.variables[var].append(f.line_no)

        return self.formulas

    def _extract_lhs_variable(self, formula_text: str) -> Optional[str]:
        """Extract the LHS variable name from a formula (before first '=')."""
        parts = re.split(r'=|\\approx', formula_text, maxsplit=1)
        if len(parts) < 2:
            return None

        lhs = parts[0].strip()
        proton_ratio = re.match(
            r'M_p\s*\^\s*(?:\{([^{}]*)\}|(\*))', lhs)
        if proton_ratio:
            superscript = proton_ratio.group(1) or proton_ratio.group(2)
            suffix = 'star' if superscript == '*' else superscript
            return f'M_p_{suffix}'

        braced_subscript = re.match(
            r'\?([a-zA-Z][a-zA-Z0-9]*)_\{([^}]*)\}', lhs)
        if braced_subscript:
            sub = braced_subscript.group(2).replace('\\', '').replace(' ', '')
            return f'{braced_subscript.group(1)}_{sub}'

        # Strip \boldsymbol{...}, \mathbf{...} wrappers to get the base name.
        # The wrapper must be at the START of the LHS (anchored), otherwise
        # a nested command like \mathbf{x} inside \boldsymbol{\Phi}(x) would
        # match the inner part instead of the outer one.
        m = re.match(r'\\(?:boldsymbol|mathbf|mathrm|mathcal)\{\\?([a-zA-Z][a-zA-Z0-9]*)\}', lhs)
        if m:
            return m.group(1)
        # Function-call LHS like \boldsymbol{\Phi}(\mathbf{x}) -> base name
        m = re.match(r'\\(?:boldsymbol|mathbf|mathrm|mathcal)\{\\?([a-zA-Z][a-zA-Z0-9]*)\}\(', lhs)
        if m:
            return m.group(1)
        # Greek-command LHS with optional subscript: \alpha_G, \theta_m
        structural = {'equiv', 'propto', 'approx', 'frac', 'sqrt', 'left',
                      'right', 'text', 'mathrm', 'mathbf', 'boldsymbol',
                      'operatorname', 'partial', 'nabla', 'infty', 'sum',
                      'int', 'cdot', 'times', 'lambdabar', 'nonumber'}
        m = re.match(r'\\([a-zA-Z]+)(?:_([a-zA-Z0-9]+))?', lhs)
        if m and m.group(1) not in structural:
            return f"{m.group(1)}_{m.group(2)}" if m.group(2) else m.group(1)
        m = re.match(r'[a-zA-Z_][a-zA-Z0-9_]*', lhs)
        if m:
            return m.group(0)
        return None

    def _extract_dependencies(self, formula_text: str, exclude_var: Optional[str] = None) -> List[str]:
        """Extract variable names from LaTeX formula.

        Strategy: scan raw text for real mathematical variables only.
        - Greek letter commands: \\theta, \\alpha, \\sigma... (as variables)
        - Latin letters with subscripts: w_1, v_F, q_j, theta_m
        - NOT LaTeX structural commands: \\frac, \\begin, \\label, etc.

        Zero-trust: ignore all @tev tag metadata words.
        Also excludes the LHS variable (it's defined by this formula, not an input).
        """
        vars_found = set()

        # 1. Greek letter commands used as variables (e.g. \\theta, \\alpha)
        greek_vars = [
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon',
            'zeta', 'eta', 'theta', 'vartheta', 'iota', 'kappa',
            'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho',
            'sigma', 'tau', 'upsilon', 'phi', 'varphi', 'chi', 'psi', 'omega',
            'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Theta', 'Lambda',
            'Mu', 'Nu', 'Xi', 'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon', 'Phi',
            'Chi', 'Psi', 'Omega',
        ]

        for m in re.finditer(r'\\([a-zA-Z]+)', formula_text):
            cmd = m.group(1)
            if cmd in greek_vars and formula_text[m.end():m.end() + 1] != '_':
                vars_found.add(cmd)

        # 2. Latin variable names with subscripts (e.g. w_1, v_F, q_j, theta_m)
        for m in re.finditer(
                r'\\?([a-zA-Z][a-zA-Z0-9]*)_'
                r'(?:\{([a-zA-Z0-9]+)\}|([a-zA-Z0-9]+))',
                formula_text):
            full = m.group(0).lstrip('\\').replace('{', '').replace('}', '')
            vars_found.add(full)

        # 3. Standalone Latin variable names (length > 2 to avoid single letters)
        standalone_text = re.sub(
            r'_[{]?[A-Za-z0-9]+[}]?', '', formula_text)
        standalone_text = re.sub(
            r'\^{[A-Za-z]+}', '', standalone_text)
        standalone_text = standalone_text.replace(r'\hbar', 'hbar')
        standalone_text = re.sub(
            r'\\[A-Za-z]+(?:_\{?[A-Za-z0-9]+\}?)?', '', standalone_text)
        for m in re.finditer(r'\b([A-Za-z][A-Za-z0-9]*)\b', standalone_text):
            name = m.group(1)
            if len(name) <= 2:
                continue
            # Skip LaTeX commands that look like Latin words
            skip_list = {'begin', 'end', 'label', 'text', 'mathbf', 'mathrm',
                       'mathcal', 'mathbb', 'it', 'quad', 'left', 'right',
                       'frac', 'sqrt', 'sum', 'prod', 'int', 'eq', 'pmatrix'}
            if name in skip_list:
                continue
            vars_found.add(name)

        # 4. Latin variable names with Greek letter subscripts (e.g. m_{\nu 1})
        for m in re.finditer(
                r'([a-zA-Z])_\{\\([a-zA-Z]+)(?: ([a-zA-Z0-9]+))?\}',
                formula_text):
            suffix = f'{m.group(2)}{m.group(3) or ""}'
            vars_found.add(f'{m.group(1)}_{suffix}')

        # Remove known constants that are not variables
        constants = {'pi', 'euler', 'gamma'}
        vars_found -= constants

        # Remove LaTeX structural commands (not real variables)
        latex_commands = {
            'boldsymbol', 'cdot', 'dagger', 'hat', 'nabla', 'otimes', 'times',
            'circ', 'ddots', 'dots', 'vdots', 'sup', 'implies', 'approx',
            'mathrm', 'mathbf', 'mathcal', 'mathbb', 'it', 'quad', 'left',
            'right', 'frac', 'sqrt', 'sum', 'prod', 'int', 'eq', 'pmatrix',
            'begin', 'end', 'label', 'text', 'sin', 'cos', 'tan', 'ln', 'log',
            'exp', 'abs', 'floor', 'ceil', 'arcsin', 'arccos', 'arctan',
            'BPS', 'Delta', 'max', 'times10', 'cell', 'T_c', 'eta_H',
        }
        vars_found -= latex_commands

        # EXCLUDE LHS variable (it's defined by this formula)
        if exclude_var:
            vars_found.discard(exclude_var)

        return sorted(vars_found)

    def _extract_numerical_value(self, formula_text: str) -> Tuple[Optional[float], Optional[str]]:
        """Extract numerical value and unit from formula text.

        Handles: \\approx 1.08, \\approx 1.10 \\times 10^{-10}, \\approx 2.818 фм
        For derivation chains (A = B \\approx C \\approx D) the LAST stated
        value is taken as the article's claimed result.

        Returns (value_in_base_unit, unit_name).
        """
        # Scientific notation: \\approx 1.10 \\times 10^{-10} [unit]
        # (caret before the exponent braces is optional; unit may be in \\text{...})
        sci_re = re.compile(
            r'\\approx\s*(-?[\d.]+)(?:\(\d+\))?\s*'
            r'\\times\s*10\^?\{(-?\d+)\}'
            r'\s*(?:\\text\{([^}]*)\}|([A-Za-z°Ω/м-]*))?'
        )

        # Simple value with optional unit (not followed by \times 10^{...});
        # degrees may be written as °, deg or ^\circ / ^{\circ}
        simple_re = re.compile(
            r'\\approx\s*(-?[\d.]+)(?:\(\d+\))?'
            r'(?!\s*\\times\s*10)'
            r'\s*(?:\\text\{([^}]*)\}'
            r'|(\^\{\\circ\}|\^\\circ|°|deg|meV|eV|GeV|К|нм|фм|м|с|kg|Pa|Ω))?'
        )

        # Fractions like \approx 1/137 or \approx \frac{1}{137}
        frac_re = re.compile(
            r'\\approx\s*(?:\\frac\{(\d+)\}\{(\d+)\}|(\d+)\s*/\s*(\d+))'
        )

        # Collect candidates; the LAST stated value in a derivation chain wins.
        # At equal position, scientific notation outranks plain numbers.
        cands: List[Tuple[int, int, float, Optional[str]]] = []  # (pos, prio, val, unit)


        for m in sci_re.finditer(formula_text):
            try:
                base = float(m.group(1))
                exp = int(m.group(2))
                val = base * (10 ** exp)
                unit = m.group(3) or m.group(4)
                if unit is not None and not unit.strip():
                    unit = None
                cands.append((m.start(), 2, val, unit))
            except ValueError:
                continue

        for m in simple_re.finditer(formula_text):
            try:
                # extend the number greedily (regex backtracking can truncate "1.10"->"1.1")
                num = m.group(1)
                j = m.start(1) + len(num)
                while j < len(formula_text) and (
                        formula_text[j].isdigit() or formula_text[j] == '.'):
                    num += formula_text[j]
                    j += 1
                val = float(num)
                unit = m.group(2) or m.group(3)
                if unit is not None and not unit.strip():
                    unit = None
                # normalize degree spellings to 'deg'
                if unit in ('^\\circ', '^\\{\\circ\\}', '\\circ'):
                    unit = 'deg'
                cands.append((m.start(), 1, val, unit))
            except ValueError:
                continue

        for m in frac_re.finditer(formula_text):
            try:
                if m.group(1) is not None:
                    val = float(m.group(1)) / float(m.group(2))
                else:
                    val = float(m.group(3)) / float(m.group(4))
                cands.append((m.start(), 1, val, None))
            except (ValueError, ZeroDivisionError):
                continue

        last_approx = formula_text.rfind(r'\approx')
        last_eq = formula_text.rfind('=')
        if last_approx >= 0 and last_eq > last_approx:
            final_value_re = re.compile(
                r'\s*(-?(?:\d+(?:\.\d*)?|\.\d+))'
                r'(?:\s*\\times\s*10\^?\{(-?\d+)\})?'
                r'\s*(?:\\text\{([^}]*)\}|([A-Za-z°Ω/м-]+))?\s*[.,;]?\s*$'
            )
            final_match = final_value_re.search(formula_text[last_eq + 1:])
            if final_match:
                try:
                    val = float(final_match.group(1))
                    if final_match.group(2) is not None:
                        val *= 10 ** int(final_match.group(2))
                    unit = final_match.group(3) or final_match.group(4)
                    if unit in ('^\\circ', '^\\{\\circ\\}', '\\circ'):
                        unit = 'deg'
                    return val, unit
                except ValueError:
                    pass

        if cands:
            cands.sort(key=lambda t: (t[0], t[1]))
            _, _, val, unit = cands[-1]
            return val, unit

        # Fallback: last bare \\approx number wins in chains
        matches = list(re.finditer(r'\\approx\\s*([\\d.]+)', formula_text))
        if matches:
            m = matches[-1]
            try:
                val = float(m.group(1))
                after = formula_text[m.end():]
                tm = re.search(r'\\text\\{([^}]*)\\}', after)
                if tm:
                    return val, tm.group(1).strip()
                return val, None
            except ValueError:
                pass

        return None, None


# Module-level alias for the matrix environment regex (used by formula_graph)
MATRIX_ENV_RE = LaTeXFormulaParser.MATRIX_ENV_RE


def parse_all_files(directory: str, pattern: str = "*.tex") -> Dict[str, List[Formula]]:
    """Parse all .tex files in a directory."""
    import os

    results = {}
    for root, dirs, files in os.walk(directory):
        for fname in sorted(files):
            if fname.endswith('.tex'):
                filepath = os.path.join(root, fname)
                parser = LaTeXFormulaParser()
                formulas = parser.parse_file(filepath)
                results[filepath] = formulas

    return results
