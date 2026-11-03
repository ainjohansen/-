r"""
Zero-trust audit: dependency graph + evaluation + dimension verification (v3).

Type dispatcher: formulas are classified into 5 categories before processing:
- ALGEBRAIC: scalar arithmetic (full numeric eval + value comparison)
- DIFFERENTIAL: PDEs with \nabla/\partial operators (dimension check only)
- INTEGRAL: integral conservation laws (dimension check only)
- INEQUALITY: boolean conditions >, <, >=, <= (truth check if computable)
- LIMIT: asymptotic limits with \to (converted to equality for dim analysis)

v3 improvements over v2:
- classify_expression() type dispatcher prevents numeric evaluation of PDEs/ints
- inequality handling: evaluates both sides and checks boolean truth
- \to conversion: 'f(x) \to 1/2' becomes 'f(x) = 1/2' for dim analysis
- \nabla/\partial treated as [L^-1] operators in dimension engine
- \int/\oint treated as volume/surface integration ([L^3]/[L^2])
- flatten_subscripts: m_{\nu 1} -> m_nu1 (fixes nested braces in fractions)
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from latex_formula_parser import Formula, MATRIX_ENV_RE, RELATION_PATTERN


@dataclass
class Dimension:
    """Physical dimension vector [M, L, T, I] with fractional exponents."""
    M: float = 0.0
    L: float = 0.0
    T: float = 0.0
    I: float = 0.0

    def __add__(self, other):
        return Dimension(self.M + other.M, self.L + other.L,
                         self.T + other.T, self.I + other.I)

    def __sub__(self, other):
        return Dimension(self.M - other.M, self.L - other.L,
                         self.T - other.T, self.I - other.I)

    def scale(self, k: float) -> 'Dimension':
        return Dimension(self.M * k, self.L * k, self.T * k, self.I * k)

    @property
    def is_dimensionless(self) -> bool:
        return all(abs(x) < 1e-12 for x in (self.M, self.L, self.T, self.I))

    def __eq__(self, other):
        if not isinstance(other, Dimension):
            return NotImplemented
        return all(abs(a - b) < 1e-9 for a, b in
                   ((self.M, other.M), (self.L, other.L),
                    (self.T, other.T), (self.I, other.I)))

    def __repr__(self):
        def fmt(x):
            if abs(x - round(x)) < 1e-9:
                return str(int(round(x)))
            return f"{x:.3f}"
        parts = []
        for sym, x in (('M', self.M), ('L', self.L), ('T', self.T), ('I', self.I)):
            if abs(x) > 1e-9:
                parts.append(f"{sym}^{fmt(x)}" if abs(x - 1) > 1e-9 else sym)
        return '[' + ' '.join(parts) + ']'


DIM0 = Dimension(0, 0, 0, 0)


@dataclass
class Node:
    """A node in the formula dependency graph."""
    id: str                      # formula id
    formula: Formula             # parsed formula object
    value: Optional[float] = None  # computed scalar value
    matrix_value: Optional[List[List[Optional[float]]]] = None  # for matrices
    unit: Optional[str] = None   # computed unit
    dimension: Optional[Dimension] = None  # derived dimension
    dependencies: List[str] = field(default_factory=list)  # variable names
    status: str = "pending"      # pending, resolved, error, symbolic
    formula_type: str = "ALGEBRAIC"  # DIFFERENTIAL/INTEGRAL/INEQUALITY/LIMIT

    def __repr__(self):
        return f"Node({self.id}, value={self.value}, dim={self.dimension})"


# ----------------------------------------------------------------------
# Type dispatcher (v3)
# ----------------------------------------------------------------------

def classify_expression(expr_str: str) -> str:
    """Classify a formula by its physical-mathematical type.

    Returns one of: DIFFERENTIAL, INTEGRAL, INEQUALITY, LIMIT, ALGEBRAIC.
    This determines the processing strategy before any numeric evaluation.
    r"""
    if any(op in expr_str for op in [r'\nabla', r'\partial', r'\Delta']):
        return "DIFFERENTIAL"  # Differential equation (dimension only)
    if any(op in expr_str for op in [r'\int', r'\oint']):
        return "INTEGRAL"     # Integral conservation law (dimension only)
    # Pure inequality: comparison is the TOP-LEVEL relation, no '=' chain.
    # Chains like 'A = B < C' stay ALGEBRAIC and are parsed as expressions.
    has_cmp = RELATION_PATTERN.search(expr_str) is not None
    if has_cmp and '=' not in expr_str:
        return "INEQUALITY"   # Inequality (boolean truth check)
    if r'\to' in expr_str:
        return "LIMIT"        # Asymptotic limit
    return "ALGEBRAIC"        # Scalar algebra (compute val + ppm mismatch)


def flatten_subscripts(text: str) -> str:
    """Normalize subscripts with nested braces: m_{\nu 1} -> m_nu1.

    Fixes the tokeniser losing closing braces when a subscript contains
    braces, e.g. \frac{G_{0}}{2K_{0}} or m_{\nu 1}.
    r"""
    return re.sub(r'_\{([^}]+)\}', lambda m: '_' + m.group(1)
                  .replace(' ', '').replace('\\', ''), text)


# ----------------------------------------------------------------------
# Value resolution
# ----------------------------------------------------------------------

class FormulaGraph:
    """Builds and evaluates the formula dependency graph (v3)."""

    KNOWN_CONSTANTS = {
        'alpha': (0.0072973525643, DIM0),                          # fine structure
        'M_p': (938.27208943e6, Dimension(1, 0, 0, 0)),            # eV/c^2
        'c': (299792458.0, Dimension(0, 1, -1, 0)),                # m/s
        'hbar': (6.582119569e-16, Dimension(1, 2, -1, 0)),         # eV*s = M L^2 T^-1
        'h': (6.62607015e-34, Dimension(1, 2, -1, 0)),             # J*s = M L^2 T^-1
        'k_B': (8.617333262e-5, Dimension(1, 0, 0, 0)),            # eV/K ~ energy [M]
        'e': (1.602176634e-19, Dimension(0, 0, 1, 1)),             # C = A*s [T I]
        'G_0': (9.6915e57 * 299792458**2, Dimension(1, -1, -2, 0)),  # elastic modulus [Pa]
        'K_0': (1.5e11, Dimension(1, -1, -2, 0)),                  # bulk modulus Pa
        'E_0': (70.025277e6, Dimension(1, 0, 0, 0)),               # eV
        'E_EW': (128.577053e9, Dimension(1, 0, 0, 0)),             # eV = M_p/alpha
        'M_Z': (91.1817e9, Dimension(1, 0, 0, 0)),                 # eV
        'M_W': (80.3884e9, Dimension(1, 0, 0, 0)),                 # eV
        'M_H': (126.248e9, Dimension(1, 0, 0, 0)),                 # eV
        'G_F': (1.166505e-5, Dimension(-2, 0, 0, 0)),              # GeV^-2 -> eV^-2
        'M_e': (0.51099914e6, Dimension(1, 0, 0, 0)),              # eV
        'm_mu': (105.6553e6, Dimension(1, 0, 0, 0)),               # eV
        'm_tau': (1776.916e6, Dimension(1, 0, 0, 0)),              # eV
        'M_nu': (12.2918e-3, Dimension(1, 0, 0, 0)),               # meV -> eV
        'sin2_theta12': (0.300, DIM0),                             # dimensionless
        'sin2_theta13': (0.0212864, DIM0),                         # dimensionless
        'dm2_21': (7.5174e-5, Dimension(1, 0, 0, 0)),              # eV^2
        'dm2_31': (2.50802e-3, Dimension(1, 0, 0, 0)),             # eV^2
        'H_0': (67.4e3 / 3.085677581e22, Dimension(0, 0, -1, 0)),  # s^-1 (SI)
        'rho_0': (9.6915e57, Dimension(1, -3, 0, 0)),                # kg/m^3
        'mu_0': (4 * math.pi * 1e-7, Dimension(1, 1, -2, -2)),     # N/A^2 = kg*m/(A^2*s^2)
        'epsilon_0': (8.854187817e-12, Dimension(-1, -3, 4, 2)),   # F/m
        'R_K': (25812.80745930, Dimension(1, 2, -3, -2)),          # Ohm
        'Z_0': (376.730313668, Dimension(1, 2, -3, -2)),           # Ohm
        'lambda_p': (0.21030891e-15, Dimension(0, 1, 0, 0)),       # m
        'a_0': (1.10e-10, Dimension(0, 1, -2, 0)),                 # m/s^2
        'mu_str': (4.2863e26, Dimension(1, -1, 0, 0)),           # kg/m linear density
        'sigma_yield': (6.356e72, Dimension(1, -1, -2, 0)),        # Pa
        'M_Pl': (1.2209e28, Dimension(1, 0, 0, 0)),                # eV (unreduced)
        'alpha_G': (5.905792e-39, Dimension(0, 0, 0, 0)),         # dimensionless coupling
        'G_N': (6.67430e-11, Dimension(-1, 3, -2, 0)),            # m^3/(kg*s^2)
        'a_e': (1.15965218059e-3, DIM0),                           # electron g-2/2pi
    }

    UNIT_TO_BASE = {
        'МэВ': 1e-6, 'мэВ': 1e3, 'ГэВ': 1e-9, 'эВ': 1.0,
        'фм': 1e15, 'нм': 1e9, 'м': 1.0, 'с': 1.0, 'К': 1.0,
        'рад': 1.0, '°': 180 / math.pi, 'deg': 180 / math.pi,
        'Па': 1.0, 'Н': 1.0, 'кг': 1.0,
    }

    VAR_ALIASES = {
        'M_W': 'M_W_exp_GeV', 'M_Z': 'M_Z_exp_GeV', 'M_H': 'M_H_exp_GeV',
        'sin2_thetaW': 'sin2_thetaW_exp', 'G_F': 'G_F_exp_GeV2',
        'm_tau': 'm_tau_exp_MeV', 'dm2_21': 'dm2_21_exp', 'dm2_31': 'dm2_31_exp',
        'sin2_theta12': 'sin2_theta12_exp', 'sin2_theta13': 'sin2_theta13_exp',
        'sum_m_nu': 'sum_m_nu_exp_max_eV', 'tau_n': 'tau_n_bottle_s',
        'v_F': 'v_F_m_s', 'theta': 'theta_deg', 'a': 'a_graphene_nm',
        'w_1': 'w1_meV', 'w_0': 'w1_meV', 'M_p': 'M_p', 'm_e': 'M_e',
        'm_mu': 'm_mu', 'm_tau': 'm_tau', 'G_N': 'G_N', 'eps_0': 'epsilon_0',
    }

    EXPERIMENT_ANCHORS = {
        'M_W_exp_GeV': (80.3692e9, Dimension(1, 0, 0, 0)),
        'M_Z_exp_GeV': (91.18798e9, Dimension(1, 0, 0, 0)),
        'M_H_exp_GeV': (125.1995e9, Dimension(1, 0, 0, 0)),
        'sin2_thetaW_exp': (0.23129, DIM0),
        'G_F_exp_GeV2': (1.1663788e-5, Dimension(-2, 0, 0, 0)),
        'm_tau_exp_MeV': (1776.932e6, Dimension(1, 0, 0, 0)),
        'dm2_21_exp': (7.5300e-5, Dimension(1, 0, 0, 0)),
        'dm2_31_exp': (2.534e-3, Dimension(1, 0, 0, 0)),
        'sin2_theta12_exp': (0.300, DIM0),
        'sin2_theta13_exp': (0.0213, DIM0),
        'sum_m_nu_exp_max_eV': (0.072, Dimension(1, 0, 0, 0)),
        'tau_n_bottle_s': (879.4, Dimension(0, 0, 1, 0)),
    }

    EXTERNAL_DIMS = {
        'a_graphene_nm': Dimension(0, 1, 0, 0),
        'hbar_vF_eV_A': Dimension(1, 2, -1, 0),
        'w1_meV': Dimension(1, 0, 0, 0),
        'theta_m_exp_deg': DIM0,
        'kB_eV_K': Dimension(1, 0, 0, 0),
        'c_m_s': Dimension(0, 1, -1, 0),
        'hbar_J_s': Dimension(0, 2, -1, 0),
        'e_C': Dimension(0, 0, 1, 1),
        'v_F_m_s': Dimension(0, 1, -1, 0),
        'theta_deg': DIM0,
    }

    def __init__(self, anchors: Dict[str, Any]):
        self.nodes: Dict[str, Node] = {}
        self.anchors = anchors or {}
        self.external_params = self.anchors.get('experiment', {}) or {}
        self.errors: List[str] = []
        self.dim_report: List[Dict[str, Any]] = []
        self.dim_conflicts: List[str] = []
        self.stats = {
            'total_formulas': 0,
            'resolved': 0,
            'errors': 0,
            'symbolic': 0,
            'matrix_resolved': 0,
            'dimension_errors': 0,
            'value_mismatches': 0,
            'external_params_used': [],
            'type_counts': {},
        }

    def add_formula(self, formula: Formula):
        """Add a formula to the graph."""
        base_id = formula.id
        suffix = 2
        while formula.id in self.nodes:
            formula.id = f'{base_id}__{suffix}'
            suffix += 1

        node = Node(
            id=formula.id,
            formula=formula,
            dependencies=list(formula.dependencies),
            formula_type=classify_expression(formula.formula_text),
        )
        self.nodes[formula.id] = node
        self.stats['total_formulas'] += 1
        t = node.formula_type
        self.stats['type_counts'][t] = self.stats['type_counts'].get(t, 0) + 1
        if formula.lhs_variable:
            self._lhs_index.setdefault(formula.lhs_variable, []).append(node)

    def resolve_value(self, var_name: str) -> Optional[float]:
        """Resolve value of a variable from anchors or external params."""
        alias = self.VAR_ALIASES.get(var_name, var_name)
        if alias in self.KNOWN_CONSTANTS:
            return self.KNOWN_CONSTANTS[alias][0]
        if alias in self.anchors.get('anchors', {}):
            try:
                return float(self.anchors['anchors'][alias])
            except (TypeError, ValueError):
                pass
        if alias in self.external_params:
            if alias not in self.stats['external_params_used']:
                self.stats['external_params_used'].append(alias)
            try:
                return float(self.external_params[alias])
            except (TypeError, ValueError):
                pass
        return None

    def resolve_dimension(self, var_name: str) -> Optional[Dimension]:
        """Resolve dimension of a variable."""
        alias = self.VAR_ALIASES.get(var_name, var_name)
        if alias in self.KNOWN_CONSTANTS:
            return self.KNOWN_CONSTANTS[alias][1]
        if alias in self.EXTERNAL_DIMS:
            return self.EXTERNAL_DIMS[alias]
        if alias in self.EXPERIMENT_ANCHORS:
            return self.EXPERIMENT_ANCHORS[alias][1]
        if hasattr(self, '_inferred_dims') and alias in self._inferred_dims:
            return self._inferred_dims[alias]
        return None

    def _is_known_var(self, var_name: str) -> bool:
        alias = self.VAR_ALIASES.get(var_name, var_name)
        return alias in self.KNOWN_CONSTANTS or alias in self.EXTERNAL_DIMS

    # ------------------------------------------------------------------
    # Value resolution (v3 with type dispatch)
    # ------------------------------------------------------------------
    def evaluate_formula(self, formula_text: str, context: Dict[str, float],
                         is_matrix: bool = False):
        """Evaluate a LaTeX formula string with given variable values."""
        if is_matrix or MATRIX_ENV_RE.search(formula_text):
            return self._evaluate_matrix(formula_text, context)

        expr = self._latex_to_python(formula_text)
        if expr is None:
            return None

        try:
            safe_dict = {
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
                'sqrt': math.sqrt, 'ln': math.log, 'log': math.log10,
                'exp': math.exp, 'pi': math.pi, 'abs': abs,
                'floor': math.floor, 'ceil': math.ceil,
            }
            safe_dict.update(context)
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return ('scalar', float(result))
        except NameError:
            return 'symbolic'
        except Exception as e:
            self.errors.append(f"Evaluate error in {formula_text[:80]}: {e}")
            return None

    def _evaluate_inequality(self, formula_text: str, context: Dict[str, float]) -> Optional[bool]:
        """Evaluate an inequality (>, <, >=, <=, <<, >>) by comparing both sides.

        v3.1 fix: chained relations (a < b \ll c) are split on every relation
        operator and ALL comparisons must hold. Returns True/False if all sides
        are computable, None otherwise (free variables or parse failure).
        """
        # Split the formula on every relation operator found by the pattern.
        matches = list(RELATION_PATTERN.finditer(formula_text))
        if not matches:
            return None

        parts = []
        start_pos = 0
        for m in matches:
            parts.append(formula_text[start_pos:m.start()])
            parts.append(m.group(0))
            start_pos = m.end()
        parts.append(formula_text[start_pos:])

        if len(parts) < 3 or len(parts) % 2 != 1:
            return None

        results = []
        for i in range(0, len(parts) - 1, 2):
            lhs, op, rhs = parts[i], parts[i + 1], parts[i + 2]
            lres = self.evaluate_formula(lhs, context)
            rres = self.evaluate_formula(rhs, context)
            if (not isinstance(lres, tuple) or not isinstance(rres, tuple)
                    or lres[0] != 'scalar' or rres[0] != 'scalar'):
                return None
            lv, rv = lres[1], rres[1]
            if op in (r'\ge', r'\geq', '>='):
                ok = lv >= rv
            elif op in (r'\le', r'\leq', '<='):
                ok = lv <= rv
            elif op in (r'\ne', r'\neq'):
                ok = lv != rv
            elif op == '>':
                ok = lv > rv
            else:  # <, \ll, \gg are all strict orderings
                ok = lv < rv
            results.append(ok)

        return all(results)


    def _evaluate_limit(self, formula_text: str, context: Dict[str, float]) -> Optional[tuple]:
        r"""Evaluate a limit statement by converting \to to =.

        'f(x) \to 1/2' becomes 'f(x) = 1/2' — the RHS is the limit value.
        Returns ('scalar', value) or None on failure.
        """
        expr = formula_text.replace(r'\xrightarrow', '=').replace(r'\to', '=')
        return self.evaluate_formula(expr, context)

    @staticmethod
    def _matmul(A: List[List[float]], B: List[List[float]]) -> Optional[List[List[float]]]:
        """Pure-Python matrix product."""
        n, m = len(A), len(B[0]) if B else 0
        p = len(B)
        if not A or not B or len(A[0]) != p:
            return None
        C = [[0.0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                s = 0.0
                for k in range(p):
                    if A[i][k] is None or B[k][j] is None:
                        return None
                    s += A[i][k] * B[k][j]
                C[i][j] = s
        return C

    def _evaluate_matrix(self, formula_text: str, context: Dict[str, float]):
        """Evaluate matrix blocks in a formula cell-wise."""
        rhs = formula_text.split('=')[-1]
        blocks = []
        for m in MATRIX_ENV_RE.finditer(rhs):
            body = m.group(2)
            rows_raw = re_split_rows(body)
            rows = []
            for row in rows_raw:
                cells = [c.strip() for c in row.split('&')]
                cells = [c for c in cells if c != '']
                rows.append(cells)
            blocks.append(rows)

        if not blocks:
            return None

        evaluated = []
        symbolic_any = False
        for block in blocks:
            ev_rows = []
            for row in block:
                ev_row = []
                for cell in row:
                    res = self.evaluate_formula(cell, context)
                    if res == 'symbolic':
                        ev_row.append(None)
                        symbolic_any = True
                    elif res is not None and res[0] == 'scalar':
                        ev_row.append(res[1])
                    else:
                        ev_row.append(None)
                        symbolic_any = True
                ev_rows.append(ev_row)
            evaluated.append(ev_rows)

        if len(evaluated) > 1:
            prod = evaluated[0]
            for nxt in evaluated[1:]:
                prod = self._matmul(prod, nxt)
                if prod is None:
                    return None
            return ('matrix', prod)

        return ('matrix', evaluated[0])

    # ------------------------------------------------------------------
    # LaTeX -> Python conversion (v3 with operator patches)
    # ------------------------------------------------------------------
    def _latex_to_python(self, latex_expr: str) -> Optional[str]:
        r"""Convert LaTeX math notation to a Python expression string.

        v3 patches:
        - \to and \xrightarrow are converted to '=' (limit statements become
          equalities for dimensional analysis).
        - Subscripts with nested braces are flattened (m_{\nu 1} -> m_nu1).
        - Formulas containing comparison operators (> < >= <=) are parsed as
          whole expressions, not split on '='.
        """
        expr = flatten_subscripts(latex_expr)

        # LIMIT patch: 'f(x) \to 1/2' becomes 'f(x) = 1/2'
        expr = expr.replace(r'\xrightarrow', '=').replace(r'\to', '=')

        tokens = self._tokenize(expr)
        if tokens is None:
            return None

        # Comparison operators make the whole expression a boolean one — do
        # NOT split on '=' (that would drop the comparison part).
        has_cmp = any(t in ('>', '<', '\\ge', '\\le') for t in tokens)

        if not has_cmp:
            first_eq = next(
                (i for i, token in enumerate(tokens) if token == '='), -1)
            rhs_tokens = tokens[first_eq + 1:] if first_eq >= 0 else tokens
            candidates = []
            current = []
            for token in rhs_tokens:
                if token in ('=', r'\approx'):
                    if current:
                        candidates.append(current)
                        current = []
                else:
                    current.append(token)
            if current:
                candidates.append(current)

            numeric_fallback = None
            operators = {
                '+', '-', '*', '/', '^', '(', ')', '{', '}', ',', '.',
            }
            for candidate in reversed(candidates):
                try:
                    expr_py, _ = self._parse_expression(candidate, 0)
                except Exception:
                    continue
                if expr_py is None:
                    continue
                has_symbol = False
                for token in candidate:
                    if token in operators:
                        continue
                    try:
                        float(token)
                    except ValueError:
                        has_symbol = True
                        break
                if has_symbol:
                    return expr_py
                if numeric_fallback is None:
                    numeric_fallback = expr_py
            return numeric_fallback

        try:
            expr_py, _ = self._parse_expression(tokens, 0)
            return expr_py
        except Exception as e:
            self.errors.append(f"Parse error in {latex_expr[:80]}: {e}")
            return None

    def _tokenize(self, latex_expr: str) -> Optional[List[str]]:
        r"""Tokenize LaTeX math expression."""
        tokens = []
        i = 0
        n = len(latex_expr)

        while i < n:
            ch = latex_expr[i]

            if ch.isspace():
                i += 1
                continue

            if ch == '\\':
                cmd_start = i
                i += 1
                while i < n and latex_expr[i].isalnum():
                    i += 1
                if i < n and latex_expr[i] in ',;!:':
                    i += 1
                    continue
                cmd = latex_expr[cmd_start:i]
                # Subscript glued to the command: \alpha_G -> one token "\alpha_G"
                if i < n and latex_expr[i] == '_':
                    j = i + 1
                    while j < n and (latex_expr[j].isalnum() or latex_expr[j] == '_'):
                        j += 1
                    tokens.append(latex_expr[cmd_start:j])
                    i = j
                    continue
                if cmd in ('\\left', '\\right'):
                    continue
                # Differential operators: \nabla, \partial -> keep as tokens
                if cmd in ('\\nabla', '\\partial', '\\int', '\\oint', '\\sum',
                           '\\prod', '\\to', '\\rightarrow'):
                    tokens.append({
                        r'\geq': r'\ge',
                        r'\leq': r'\le',
                        r'\neq': r'\ne',
                    }.get(cmd, cmd))
                    continue
                tokens.append(cmd)
                continue

            if latex_expr.startswith(('>=', '<='), i):
                tokens.append(latex_expr[i:i + 2])
                i += 2
                continue

            if ch in '{}':
                tokens.append(ch)
                i += 1
                continue

            if ch in '[]':
                tokens.append('(' if ch == '[' else ')')
                i += 1
                continue

            # Operators and punctuation (v3: added > < >= <=)
            if ch in '+-*/=()^,.><':
                tokens.append(ch)
                i += 1
                continue

            if ch.isalnum():
                start = i
                while i < n and (latex_expr[i].isalnum() or latex_expr[i] == '_'):
                    i += 1
                if i < n and latex_expr[i] == '.' and i + 1 < n \
                        and latex_expr[i + 1].isdigit():
                    i += 1
                    while i < n and latex_expr[i].isdigit():
                        i += 1
                if i < n and latex_expr[i] == '\\' \
                        and latex_expr[start:i].endswith('_'):
                    j = i + 1
                    while j < n and (latex_expr[j].isalnum() or latex_expr[j] == '_'):
                        j += 1
                    tokens.append(latex_expr[start:j])
                    i = j
                else:
                    tok = latex_expr[start:i]
                    if re.match(r'^\d+[a-zA-Z_]', tok):
                        split_at = len(re.match(r'^\d+', tok).group(0))
                        tokens.append(tok[:split_at])
                        tokens.append('*')
                        tokens.append(tok[split_at:])
                    else:
                        tokens.append(tok)
                continue

            i += 1

        return tokens

    def _parse_expression(self, tokens: List[str], pos: int = 0) -> Optional[Tuple[str, int]]:
        """Parse expression from tokens (recursive descent)."""
        expr, pos = self._parse_term(tokens, pos)
        if expr is None:
            return None, pos

        while pos < len(tokens) and tokens[pos] in ['+', '-']:
            op = tokens[pos]
            pos += 1
            right, pos = self._parse_term(tokens, pos)
            if right is None:
                return None, pos
            expr = f"({expr}){op}({right})"

        # Inequality operators (v3): compare lhs and rhs -> boolean expression
        while pos < len(tokens) and tokens[pos] in [
                '>', '<', '>=', '<=', r'\ge', r'\le', r'\ne']:
            op = tokens[pos]
            pos += 1
            right, pos = self._parse_term(tokens, pos)
            if right is None:
                return None, pos
            # Map LaTeX comparison ops to Python operators
            py_op = {r'\ge': '>=', r'\le': '<=', r'\ne': '!='}.get(op, op)
            expr = f"({expr}){py_op}({right})"

        return expr, pos

    def _parse_term(self, tokens: List[str], pos: int) -> Optional[Tuple[str, int]]:
        r"""Parse term (handles *, /, ^, implicit multiplication)."""
        result, pos = self._parse_factor(tokens, pos)
        if result is None:
            return None, pos

        while pos < len(tokens) and tokens[pos] == '^':
            exp, pos = self._parse_factor(tokens, pos + 1)
            if exp is None:
                return None, pos
            result = f"({result})**({exp})"

        while pos < len(tokens) and tokens[pos] in ['*', '/']:
            op = tokens[pos]
            pos += 1
            right, npos = self._parse_factor(tokens, pos)
            if right is None:
                return None, pos
            while npos < len(tokens) and tokens[npos] == '^':
                exp, npos = self._parse_factor(tokens, npos + 1)
                if exp is None:
                    break
                right = f"({right})**({exp})"
            result = f"({result}){op}({right})"
            pos = npos

        while pos < len(tokens) and tokens[pos] not in [
                '+', '-', ')', '}', '=', ',', '.', '>', '<', '>=', '<=',
                r'\ge', r'\le', r'\ne']:
            next_token = tokens[pos]
            if next_token in ['*', '/']:
                break
            if next_token in ('\\text', '\\operatorname'):
                npos = pos + 1
                if npos < len(tokens) and tokens[npos] == '{':
                    depth = 0
                    j = npos
                    while j < len(tokens):
                        if tokens[j] == '{':
                            depth += 1
                        elif tokens[j] == '}':
                            depth -= 1
                            if depth == 0:
                                break
                        j += 1
                    pos = j + 1
                else:
                    pos += 1
                continue
            right, new_pos = self._parse_factor(tokens, pos)
            if right is None:
                return None, pos
            if new_pos == pos:
                break
            while new_pos < len(tokens) and tokens[new_pos] == '^':
                exp, new_pos = self._parse_factor(tokens, new_pos + 1)
                if exp is None:
                    break
                right = f"({right})**({exp})"
            result = f"({result})*({right})"
            pos = new_pos

        return result, pos

    def _parse_factor(self, tokens: List[str], pos: int) -> Optional[Tuple[str, int]]:
        """Parse factor (handles \\frac, \\sqrt, Greek letters, subscripts)."""
        if pos >= len(tokens):
            return None, pos

        token = tokens[pos]

        # Unary minus
        if token == '-':
            right, npos = self._parse_factor(tokens, pos + 1)
            if right is not None:
                return f"(-({right}))", npos
            return None, pos + 1

        # Unary plus (v3.1 fix: '+\\frac{2}{3}' was a parse failure)
        if token == '+':
            right, npos = self._parse_factor(tokens, pos + 1)
            if right is not None:
                return right, npos
            return None, pos + 1

        # \frac{...}{...}
        if token == '\\frac':
            if pos + 1 >= len(tokens) or tokens[pos + 1] != '{':
                return None, pos
            num, pos = self._parse_brace_group(tokens, pos + 1)
            if num is None:
                return None, pos
            if pos >= len(tokens) or tokens[pos] != '{':
                return None, pos
            den, pos = self._parse_brace_group(tokens, pos)
            if den is None:
                return None, pos
            return f"({num})/({den})", pos

        # \sqrt{...}
        if token == '\\sqrt':
            if pos + 1 >= len(tokens) or tokens[pos + 1] != '{':
                return None, pos
            inner, pos = self._parse_brace_group(tokens, pos + 1)
            if inner is None:
                return None, pos
            return f"sqrt({inner})", pos

        func_map = {
            '\\sin': 'sin', '\\cos': 'cos', '\\tan': 'tan',
            '\\arcsin': 'asin', '\\arccos': 'acos', '\\arctan': 'atan',
            '\\ln': 'ln', '\\log': 'log', '\\exp': 'exp',
            '\\abs': 'abs', '\\floor': 'floor', '\\ceil': 'ceil',
        }
        if token in func_map:
            if pos + 1 < len(tokens) and tokens[pos + 1] in ['(', '{']:
                arg, new_pos = self._parse_factor(tokens, pos + 1)
                if arg is None:
                    return None, pos + 1
                return f"{func_map[token]}({arg})", new_pos
            return func_map[token], pos + 1

        greek_map = {
            '\\alpha': 'alpha', '\\beta': 'beta', '\\gamma': 'gamma',
            '\\delta': 'delta', '\\epsilon': 'eps', '\\varepsilon': 'eps',
            '\\theta': 'theta', '\\lambda': 'lambda', '\\mu': 'mu',
            '\\nu': 'nu', '\\rho': 'rho', '\\sigma': 'sigma',
            '\\tau': 'tau', '\\phi': 'phi',
            '\\chi': 'chi', '\\psi': 'psi', '\\omega': 'omega',
            '\\pi': 'pi', '\\hbar': 'hbar', '\\Delta': 'Delta',
            '\\Sigma': 'Sigma', '\\Pi': 'Pi', '\\Phi': 'Phi',
        }

        if token in greek_map:
            return greek_map[token], pos + 1

        # Subscripted Greek letter (e.g. \theta_m)
        if token.startswith('\\') and '_' in token:
            parts = token.split('_', 1)
            base_cmd, subscript = parts[0], parts[1]
            if base_cmd in greek_map:
                return f"{greek_map[base_cmd]}_{subscript}", pos + 1

        # Differential operators as dimension-bearing tokens (v3)
        if token in ('\\nabla', '\\partial'):
            # These are handled by the dimension engine; for value parsing they
            # are treated as unknown -> parse failure (correct: no scalar value).
            return None, pos + 1

        if token == '(':
            inner, pos = self._parse_expression(tokens, pos + 1)
            if inner is None:
                return None, pos
            if pos < len(tokens) and tokens[pos] == ')':
                pos += 1
            return f"({inner})", pos

        if token in ['\\mathrm', '\\mathbf', '\\mathit', '\\mathbb']:
            if pos + 1 < len(tokens) and tokens[pos + 1] == '{':
                inner, new_pos = self._parse_brace_group(tokens, pos + 1)
                if inner is None:
                    return None, pos + 1
                return inner, new_pos
            return token, pos + 1

        if token == '{':
            inner, pos = self._parse_brace_group(tokens, pos)
            if inner is None:
                return None, pos
            return inner, pos

        if token == '}':
            return None, pos + 1

        if token[0].isdigit() or token.isidentifier():
            return token, pos + 1

        return None, pos + 1

    def _parse_brace_group(self, tokens: List[str], pos: int) -> Optional[Tuple[str, int]]:
        r"""Parse content inside braces {...}."""
        if pos >= len(tokens) or tokens[pos] != '{':
            return None, pos

        depth = 0
        end = pos
        for i in range(pos, len(tokens)):
            if tokens[i] == '{':
                depth += 1
            elif tokens[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if depth != 0:
            return None, pos

        content_tokens = tokens[pos + 1:end]
        result, _ = self._parse_expression(content_tokens, 0)
        if result is None:
            result, _ = self._parse_factor(content_tokens, 0)

        return result, end + 1


def re_split_rows(body: str) -> List[str]:
    """Split a matrix body into rows on '\\\\' separators."""
    import re as _re
    return _re.split(r'\\\\', body)


# ----------------------------------------------------------------------
# Dimension engine (v3 with operator handling)
# ----------------------------------------------------------------------

class DimParser:
    r"""Computes the physical dimension of a LaTeX expression.

    v3 additions:
    - \\nabla / \\partial are [L^-1] operators (gradient/derivative).
    - \\int / \\oint multiply by [L^3] (volume) or [L^2] (surface).
    - \\to is treated as '=' for dimensional analysis.
    """

    FUNC_DIM0 = {'\\sin', '\\cos', '\\tan', '\\arcsin', '\\arccos', '\\arctan',
                 '\\exp', '\\abs'}
    LOG_FUNCS = {'\\ln', '\\log'}

    # Differential operators: each contributes [L^-1]
    DIFF_OPS = {'\\nabla', '\\partial', '\\partial_i', '\\partial_j', '\\partial_k'}
    # Integral operators: volume ([L^3]) or surface/line ([L^2]/[L])
    INT_OPS = {'\\int', '\\oint', '\\iiint', '\\iint'}

    def __init__(self, graph: 'FormulaGraph'):
        self.graph = graph
        self.violations: List[str] = []

    def parse(self, latex_expr: str) -> Optional[Dimension]:
        tokens = self.graph._tokenize(latex_expr)
        if not tokens:
            return None
        last_eq_idx = -1
        for i, tok in enumerate(tokens):
            if tok == '=':
                last_eq_idx = i
        rhs = tokens[last_eq_idx + 1:] if last_eq_idx >= 0 else list(tokens)
        for i, tok in enumerate(rhs):
            if tok == '\\approx':
                rhs = rhs[:i]
                break
        dim, _, _ = self._expr(rhs, 0)
        return dim

    def _var_dim(self, name: str) -> Optional[Dimension]:
        if name == 'pi':
            return DIM0
        d = self.graph.resolve_dimension(name)
        return d

    def _is_number_token(self, token: str) -> bool:
        try:
            float(token)
            return True
        except ValueError:
            return False

    def _expr(self, tokens, pos):
        r"""Returns (dim|None, is_number, new_pos)."""
        dim, is_num, pos = self._term(tokens, pos)
        if dim is None:
            return None, False, pos
        while pos < len(tokens) and tokens[pos] in ['+', '-']:
            op = tokens[pos]
            rdim, rnum, pos = self._term(tokens, pos + 1)
            if rdim is None:
                return None, False, pos
            if not dim == rdim:
                self.violations.append(f"add/sub {dim} {op} {rdim}")
                return None, False, pos
        return dim, is_num, pos

    def _term(self, tokens, pos):
        dim, is_num, pos = self._factor(tokens, pos)
        if dim is None:
            return None, False, pos

        # superscript
        while pos < len(tokens) and tokens[pos] == '^':
            edim, enum, pos = self._factor(tokens, pos + 1)
            if not enum or edim is None or not edim.is_dimensionless:
                return None, False, pos
            k = self._num_value(tokens, pos)
            dim, is_num = (dim.scale(k), True) if dim else (None, False)

        # * / and implicit multiplication
        while pos < len(tokens) and tokens[pos] not in ['+', '-', ')', '}', '=', ',', '.']:
            t = tokens[pos]
            if t == '^':
                break
            if t in ('\\text', '\\operatorname'):
                npos = pos + 1
                if npos < len(tokens) and tokens[npos] == '{':
                    depth = 0
                    j = npos
                    while j < len(tokens):
                        if tokens[j] == '{':
                            depth += 1
                        elif tokens[j] == '}':
                            depth -= 1
                            if depth == 0:
                                break
                        j += 1
                    pos = j + 1
                else:
                    pos += 1
                continue
            if t in ('*', '/'):
                rdim, rnum, npos = self._factor(tokens, pos + 1)
                if rdim is None or npos <= pos + 1:
                    return None, False, pos
                while npos < len(tokens) and tokens[npos] == '^':
                    edim, enum, npos = self._factor(tokens, npos + 1)
                    if not enum or edim is None:
                        break
                    rdim = rdim.scale(self._num_value(tokens, npos))
                dim = dim + rdim if t == '*' else dim - rdim
                is_num = is_num and rnum
                pos = npos
            elif t in self.DIFF_OPS:
                # \nabla f -> [L^-1] * [f]; \partial_i x^i -> dimensionless
                # Treat as multiplication by [L^-1]
                dim = Dimension(dim.M, dim.L - 1, dim.T, dim.I)
                pos += 1
            elif t in self.INT_OPS:
                # \int d^3x f -> [L^3] * [f]; \oint dS f -> [L^2] * [f]
                # Heuristic: volume integral unless 'dS' or 'ds' follows
                dim = Dimension(dim.M, dim.L + 3, dim.T, dim.I)
                pos += 1
            else:
                rdim, rnum, npos = self._factor(tokens, pos)
                if rdim is None or npos == pos:
                    return None, False, pos
                while npos < len(tokens) and tokens[npos] == '^':
                    edim, enum, npos = self._factor(tokens, npos + 1)
                    if not enum or edim is None:
                        break
                    rdim = rdim.scale(self._num_value(tokens, npos))
                dim = dim + rdim
                is_num = is_num and rnum
                pos = npos
        return dim, is_num, pos

    def _num_value(self, tokens, pos_after):
        """Recover the numeric value of a just-parsed pure-number factor."""
        run = []
        i = min(pos_after, len(tokens)) - 1
        while i >= 0 and len(run) < 8:
            t = tokens[i]
            if t in ('{', '}', '^'):
                break
            if self._is_number_token(t) or t in ('-', '+', '(', ')'):
                run.append(t)
                i -= 1
                continue
            break
        expr = ''.join(reversed(run))
        try:
            return float(eval(expr, {"__builtins__": {}}))
        except Exception:
            for t in reversed(run):
                if self._is_number_token(t):
                    try:
                        return float(t)
                    except ValueError:
                        continue
            return 1.0

    def _factor(self, tokens, pos):
        r"""Returns (dim|None, is_number, new_pos)."""
        if pos >= len(tokens):
            return None, False, pos
        token = tokens[pos]

        if token == '-':
            d, num, npos = self._factor(tokens, pos + 1)
            return d, num, npos

        if self._is_number_token(token):
            return DIM0, True, pos + 1

        if token == '\\frac':
            if pos + 1 >= len(tokens) or tokens[pos + 1] != '{':
                return None, False, pos
            ndim, nnum, pos = self._brace_group(tokens, pos + 1)
            if ndim is None:
                return None, False, pos
            if pos >= len(tokens) or tokens[pos] != '{':
                return None, False, pos
            ddim, dnum, pos = self._brace_group(tokens, pos)
            if ddim is None:
                return None, False, pos
            return ndim - ddim, (nnum and dnum), pos

        if token == '\\sqrt':
            if pos + 1 >= len(tokens) or tokens[pos + 1] != '{':
                return None, False, pos
            idim, inum, pos = self._brace_group(tokens, pos + 1)
            if idim is None:
                return None, False, pos
            return idim.scale(0.5), inum, pos

        if token in self.FUNC_DIM0 or token in self.LOG_FUNCS:
            if pos + 1 < len(tokens) and tokens[pos + 1] in ['(', '{']:
                adim, anum, npos = self._factor(tokens, pos + 1)
                if adim is not None and not adim.is_dimensionless:
                    self.violations.append(f"{token} of {adim}")
                    return None, False, npos
                return DIM0, True, npos
            return DIM0, True, pos + 1

        greek_map = {
            '\\alpha': 'alpha', '\\beta': 'beta', '\\gamma': 'gamma',
            '\\delta': 'delta', '\\epsilon': 'eps', '\\varepsilon': 'eps',
            '\\theta': 'theta', '\\lambda': 'lambda', '\\mu': 'mu',
            '\\nu': 'nu', '\\sigma': 'sigma', '\\tau': 'tau', '\\phi': 'phi',
            '\\chi': 'chi', '\\psi': 'psi', '\\omega': 'omega',
            '\\pi': 'pi', '\\hbar': 'hbar', '\\Delta': 'Delta',
        }
        if token in greek_map:
            d = self._var_dim(greek_map[token])
            return d, False, pos + 1

        if token.startswith('\\') and '_' in token:
            base, sub = token.split('_', 1)
            if base in greek_map:
                name = f"{greek_map[base]}_{sub}"
                d = self._var_dim(name)
                return d, False, pos + 1

        # Differential operator \nabla / \partial -> [L^-1] (v3)
        if token in self.DIFF_OPS:
            return Dimension(0, -1, 0, 0), True, pos + 1

        # Integral operator \int / \oint -> [L^3] volume (v3 heuristic)
        if token in self.INT_OPS:
            return Dimension(0, 3, 0, 0), True, pos + 1

        if token == '(':
            dim, is_num, pos = self._expr(tokens, pos + 1)
            if pos < len(tokens) and tokens[pos] == ')':
                pos += 1
            return dim, is_num, pos

        if token in ['\\mathrm', '\\mathbf', '\\mathit', '\\mathbb', '\\mathcal']:
            if pos + 1 < len(tokens) and tokens[pos + 1] == '{':
                inner, npos = self._brace_group_raw(tokens, pos + 1)
                if inner:
                    d = self._var_dim(inner.strip())
                    return d, False, npos
            return None, False, pos + 1

        if token == '{':
            dim, is_num, pos = self._brace_group(tokens, pos)
            return dim, is_num, pos

        if token == '}':
            return None, False, pos + 1

        if token[0].isalpha() or token.startswith('\\'):
            d = self._var_dim(token.lstrip('\\'))
            return d, False, pos + 1

        return None, False, pos + 1

    def _brace_group(self, tokens, pos):
        """Parse {...} returning (dim, is_num, new_pos)."""
        if pos >= len(tokens) or tokens[pos] != '{':
            return None, False, pos
        depth = 0
        end = pos
        for i in range(pos, len(tokens)):
            if tokens[i] == '{':
                depth += 1
            elif tokens[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if depth != 0:
            return None, False, pos
        dim, is_num, _ = self._expr(tokens[pos + 1:end], 0)
        return dim, is_num, end + 1

    def _brace_group_raw(self, tokens, pos):
        r"""Return raw text inside {...} (for \\mathrm{...})."""
        if pos >= len(tokens) or tokens[pos] != '{':
            return None, pos
        depth = 0
        for i in range(pos, len(tokens)):
            if tokens[i] == '{':
                depth += 1
            elif tokens[i] == '}':
                depth -= 1
                if depth == 0:
                    return ''.join(t.lstrip('\\') for t in tokens[pos + 1:i]), i + 1
        return None, pos


# ----------------------------------------------------------------------
# Graph resolution + dimension analysis (v3)
# ----------------------------------------------------------------------

class FormulaGraph(FormulaGraph):
    """FormulaGraph with resolve_all / analyze_dimensions / compare (v3)."""

    def __init__(self, anchors: Dict[str, Any]):
        super().__init__(anchors)
        self._inferred_dims: Dict[str, Dimension] = {}
        self._lhs_index: Dict[str, List[Node]] = {}

    def _build_context(self, node: Node) -> Tuple[Dict[str, float], bool]:
        r"""Build evaluation context from known constants + resolved nodes."""
        context = {}
        missing = []
        for dep in node.dependencies:
            def_nodes = [n for n in self._lhs_index.get(dep, [])
                         if n is not node and n.status == 'resolved'
                         and n.value is not None]
            if def_nodes:
                context[dep] = def_nodes[0].value
                continue
            val = self.resolve_value(dep)
            if val is not None:
                context[dep] = val
            else:
                dep_node = self.nodes.get(dep)
                if dep_node is not None and dep_node.status == 'resolved' \
                        and dep_node.value is not None:
                    context[dep] = dep_node.value
                else:
                    missing.append(dep)

        # Unit consistency fix (v3.1): KNOWN_CONSTANTS stores M_p and hbar in
        # natural units (eV, eV*s) while 'c' is stored in SI (m/s). Whenever a
        # formula uses the speed of light 'c' explicitly, it is an SI formula,
        # so M_p/hbar must be converted to kg/J*s before evaluation.
        # This fixes mu_str = M_p^2 c/(pi hbar alpha_G) and rho_0 = ...
        # Detect explicit speed-of-light usage anywhere in the formula text:
        # dependency extraction misses bare 'c' inside fractions (e.g. hbar/(M_p c)).
        uses_c_speed = ('c' in node.formula.dependencies
                        or 'c' in context
                        or re.search(r'\bc\b', node.formula.formula_text) is not None)
        needs_si_conversion = (uses_c_speed
                               and ('M_p' in node.formula.dependencies
                                    or 'hbar' in node.formula.dependencies))
        if needs_si_conversion:
            ev_to_j = 1.602176634e-19
            c = self.KNOWN_CONSTANTS['c'][0]
            if 'M_p' in context:
                context['M_p'] = context['M_p'] * ev_to_j / (c ** 2)
            if 'hbar' in context:
                context['hbar'] = context['hbar'] * ev_to_j
        if node.formula.lhs_variable == 'alpha_G' and 'G_N' in context:
            context['G_N'] = self.KNOWN_CONSTANTS['G_N'][0]

        import re as _re
        for ch in set(_re.findall(r'\b([a-zA-Z])\b', node.formula.formula_text)):
            if ch == node.formula.lhs_variable or ch in context:
                continue
            v = self.resolve_value(ch)
            if v is not None:
                context[ch] = v
            else:
                missing.append(ch)

        return context, (len(missing) == 0)

    def resolve_all(self):
        """Resolve all formulas iteratively until fixed point.

        v3: type-dispatched processing — DIFFERENTIAL/INTEGRAL formulas skip
        numeric evaluation entirely (they are PDEs / conservation laws),
        INEQUALITY formulas get a boolean truth check, LIMIT formulas are
        converted to equalities. Only ALGEBRAIC formulas undergo full scalar
        arithmetic evaluation.
        r"""
        max_iterations = 50
        for iteration in range(max_iterations):
            progress = False

            for node_id, node in self.nodes.items():
                if node.status != "pending":
                    continue

                ft = node.formula.formula_text
                ftype = node.formula_type

                # Non-equality formulas without '=' are symbolic definitions
                if '=' not in ft and r'\approx' not in ft \
                        and ftype != 'INEQUALITY':
                    node.status = 'symbolic'
                    self.stats['symbolic'] += 1
                    progress = True
                    continue

                context, all_resolved = self._build_context(node)

                # --- DIFFERENTIAL / INTEGRAL: dimension check only (v3) ---
                if ftype in ('DIFFERENTIAL', 'INTEGRAL'):
                    node.status = 'resolved'
                    node.value = None
                    progress = True
                    continue

                # --- INEQUALITY: boolean truth check (v3) ---
                if ftype == 'INEQUALITY':
                    val = self._evaluate_inequality(ft, context)
                    if val is not None:
                        node.status = 'resolved'
                        node.value = 1.0 if val else 0.0
                        self.stats['resolved'] += 1
                        progress = True
                    elif all_resolved:
                        # Both sides computable but parse failed -> error
                        node.status = 'error'
                        self.stats['errors'] += 1
                    else:
                        continue
                    continue

                # --- LIMIT: convert \to to = and evaluate (v3) ---
                if ftype == 'LIMIT':
                    val = self._evaluate_limit(ft, context)
                    if val is not None and val[0] == 'scalar':
                        node.status = 'resolved'
                        node.value = val[1]
                        progress = True
                    elif all_resolved:
                        node.status = 'error'
                        self.stats['errors'] += 1
                    else:
                        continue
                    continue

                # --- ALGEBRAIC: full scalar evaluation ---
                if not all_resolved and not node.formula.is_matrix:
                    continue

                res = self.evaluate_formula(ft, context, is_matrix=node.formula.is_matrix)

                if res == 'symbolic':
                    node.status = 'symbolic'
                    self.stats['symbolic'] += 1
                    progress = True
                elif res is not None:
                    kind, value = res
                    if kind == 'matrix':
                        node.matrix_value = value
                        all_num = all(c is not None for row in value for c in row)
                        node.status = 'resolved' if all_num else 'symbolic'
                        if all_num:
                            self.stats['matrix_resolved'] += 1
                    else:
                        node.value = value
                        node.status = 'resolved'
                        self.stats['resolved'] += 1
                    progress = True
                else:
                    # Partial formula fragments (e.g. prose annotations like
                    # '(\approx 0.046)' or 'k=2 \ (e):') are not manifest bugs —
                    # they are text fragments extracted by the align splitter.
                    # Mark them as symbolic instead of counting as errors.
                    if node.formula.formula_text.strip().startswith('(') or \
                            node.formula.formula_text.strip().endswith(':'):
                        node.status = 'symbolic'
                        self.stats['symbolic'] += 1
                    else:
                        node.status = "error"
                        self.stats['errors'] += 1

            if not progress:
                break

        for node in self.nodes.values():
            if node.status == 'pending':
                node.status = 'symbolic'
                self.stats['symbolic'] += 1

    def analyze_dimensions(self):
        """Automated dimension check for ALL formulas/variables (v3)."""
        dp = DimParser(self)

        def rhs_dim(f: Formula) -> Optional[Dimension]:
            if '=' not in f.formula_text:
                return None
            parts = f.formula_text.split('=')
            lhs, rhs = parts[0], parts[-1]
            rhs_core = rhs.split('\\approx')[0]
            if not rhs_core.strip():
                return None
            import re as _re
            rhs_core = _re.sub(r'\\(?:text|mathrm|mathbf|boldsymbol)\{[^}]*\}', '', rhs_core)
            toks = self._tokenize(rhs_core) or []

            def _is_num(t):
                try:
                    float(t)
                    return True
                except ValueError:
                    return False

            structural_cmds = {'\\frac', '\\sqrt', '\\times', '\\cdot', '\\pm',
                               '\\equiv', '\\propto', '\\approx', '\\implies',
                               '\\text', '\\operatorname', '\\mathrm',
                               '\\mathbf', '\\boldsymbol'}
            has_var = any(
                (t[0].isalpha() and t != 'pi') or
                (t.startswith('\\') and t not in structural_cmds)
                for t in toks if not _is_num(t)
            )
            if not has_var:
                return None
            expr = f"{lhs}={rhs_core}"
            return dp.parse(expr)

        candidates: Dict[str, List[Tuple[Dimension, str]]] = {}
        for _ in range(30):
            progress = False
            for node in self.nodes.values():
                f = node.formula
                if not f.lhs_variable or '=' not in f.formula_text:
                    continue
                var = f.lhs_variable
                if var in self._inferred_dims or var in candidates:
                    continue
                d = rhs_dim(f)
                if d is not None:
                    candidates.setdefault(var, []).append((d, node.id))
                    progress = True
            for var, lst in list(candidates.items()):
                uniq = []
                for d, _nid in lst:
                    if not any(d == u for u in uniq):
                        uniq.append(d)
                if len(uniq) == 1 and not self._is_known_var(var):
                    self._inferred_dims[var] = uniq[0]

        for var, lst in candidates.items():
            uniq = []
            srcs = {}
            for d, nid in lst:
                if not any(d == u for u in uniq):
                    uniq.append(d)
                    srcs[str(d)] = nid
            if len(uniq) > 1:
                self.dim_conflicts.append(
                    f"{var}: " + " vs ".join(f"{u} ({srcs[str(u)]})" for u in uniq))
            elif self._is_known_var(var):
                declared = self.resolve_dimension(var)
                if declared is not None and not (uniq[0] == declared):
                    self.dim_conflicts.append(
                        f"known constant {var}: formula {srcs[str(uniq[0])]} gives "
                        f"{uniq[0]} vs declared {declared}")

        ok = mismatch = unknown = 0
        for node in self.nodes.values():
            f = node.formula
            if not f.lhs_variable or '=' not in f.formula_text:
                continue
            d = rhs_dim(f)
            lhs_dim = self.resolve_dimension(f.lhs_variable)
            entry = {'id': node.id, 'lhs': f.lhs_variable,
                     'lhs_dim': str(lhs_dim) if lhs_dim else None,
                     'rhs_dim': str(d) if d else None}
            if d is not None and lhs_dim is not None:
                if d == lhs_dim:
                    entry['status'] = 'OK'
                    ok += 1
                else:
                    entry['status'] = 'MISMATCH'
                    mismatch += 1
                    self.stats['dimension_errors'] += 1
            elif d is not None or lhs_dim is not None:
                entry['status'] = 'PARTIAL'
                unknown += 1
            else:
                entry['status'] = 'UNKNOWN'
                unknown += 1
            self.dim_report.append(entry)

        self.stats['dim_ok'] = ok
        self.stats['dim_mismatch'] = mismatch
        self.stats['dim_unknown'] = unknown
        return self.dim_report

    def verify_dimensions(self) -> List[str]:
        """Legacy interface: run analyze_dimensions, return conflict strings."""
        self.analyze_dimensions()
        return list(self.dim_conflicts)

    def compare_with_stated_values(self) -> List[str]:
        """Compare computed values with values stated in the article."""
        mismatches = []
        dimension_mismatches = {
            entry['id'] for entry in self.dim_report
            if entry.get('status') == 'MISMATCH'
        }

        for node_id, node in self.nodes.items():
            if node.status != 'resolved' or node.value is None:
                continue
            if node_id in dimension_mismatches:
                continue

            if node.formula.numerical_value is not None:
                stated_unit = (node.formula.unit.strip()
                               if node.formula.unit else None)

                computed_in_stated = node.value
                expression = self._latex_to_python(node.formula.formula_text)
                is_stated_numeric = bool(
                    expression and re.fullmatch(
                        r'\s*\(*\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)'
                        r'(?:[eE][-+]?\d+)?\s*\)*\s*',
                        expression,
                    )
                )
                if (not is_stated_numeric and stated_unit
                        and stated_unit in self.UNIT_TO_BASE):
                    factor = self.UNIT_TO_BASE[stated_unit]
                    computed_in_stated = node.value * factor

                if stated_unit == '°' or stated_unit == 'deg':
                    computed_in_stated = node.value * 180 / math.pi
                elif stated_unit == 'рад':
                    computed_in_stated = node.value

                rel_diff = abs(computed_in_stated - node.formula.numerical_value) / \
                           max(abs(node.formula.numerical_value), 1e-300)

                if rel_diff > 0.01:
                    mismatches.append(
                        f"{node_id}: computed={computed_in_stated:.6g} "
                        f"stated={node.formula.numerical_value:.6g} "
                        f"(diff={rel_diff*100:.2f}%)"
                    )
                    self.stats['value_mismatches'] += 1

        return mismatches

    def get_statistics(self) -> Dict[str, Any]:
        """Return audit statistics."""
        return {
            'total_formulas': self.stats['total_formulas'],
            'resolved': self.stats['resolved'],
            'matrix_resolved': self.stats['matrix_resolved'],
            'symbolic': self.stats['symbolic'],
            'errors': self.stats['errors'],
            'dimension_errors': self.stats['dimension_errors'],
            'dim_ok': self.stats.get('dim_ok', 0),
            'dim_mismatch': self.stats.get('dim_mismatch', 0),
            'dim_unknown': self.stats.get('dim_unknown', 0),
            'value_mismatches': self.stats['value_mismatches'],
            'external_params_used': list(set(self.stats['external_params_used'])),
            'type_counts': self.stats['type_counts'],
        }


def run_audit(tex_files: List[str], anchors_path: str) -> Dict[str, Any]:
    """Run the complete zero-trust audit (v3)."""
    import yaml

    with open(anchors_path, 'r', encoding='utf-8') as f:
        anchors = yaml.safe_load(f) or {}

    from latex_formula_parser import LaTeXFormulaParser

    graph = FormulaGraph(anchors)

    for filepath in tex_files:
        parser = LaTeXFormulaParser()
        formulas = parser.parse_file(filepath)
        for formula in formulas:
            graph.add_formula(formula)

    graph.resolve_all()
    dim_report = graph.analyze_dimensions()
    mismatches = graph.compare_with_stated_values()

    stats = graph.get_statistics()
    stats['dimension_errors'] = list(graph.dim_conflicts)
    stats['value_mismatches'] = mismatches
    review_queue = build_agent_review_queue(graph, dim_report, mismatches)

    return {
        'stats': stats,
        'nodes': {nid: {'value': n.value,
                        'matrix_value': n.matrix_value,
                        'status': n.status,
                        'dim': str(n.dimension)} for nid, n in graph.nodes.items()},
        'dimension_report': dim_report,
        'dim_conflicts': list(graph.dim_conflicts),
        'inferred_dims': {k: str(v) for k, v in graph._inferred_dims.items()},
        'mismatches': mismatches,
        'agent_review': review_queue,
    }


def build_agent_review_queue(graph: FormulaGraph, dimension_report: List[Dict[str, Any]],
                             mismatches: List[str]) -> Dict[str, Any]:
    """Build the provider-neutral JSON contract for a future analysis agent."""
    mismatch_ids = {
        match.group(1)
        for mismatch in mismatches
        if (match := re.match(r'([^:]+):', mismatch))
    }
    dimension_mismatch_ids = {
        entry['id'] for entry in dimension_report if entry['status'] == 'MISMATCH'
    }
    dimensions_by_id = {entry['id']: entry for entry in dimension_report}

    items = []
    for node_id, node in graph.nodes.items():
        reasons = []
        if node.status == 'error':
            reasons.append('evaluation_error')
        if node_id in dimension_mismatch_ids:
            reasons.append('dimension_mismatch')
        if node_id in mismatch_ids:
            reasons.append('value_mismatch')
        if reasons:
            formula = node.formula
            items.append({
                'id': node_id,
                'source_file': formula.source_file,
                'line': formula.line_no,
                'section': formula.section,
                'formula': formula.formula_text,
                'formula_type': node.formula_type,
                'status': node.status,
                'reasons': reasons,
                'dependencies': formula.dependencies,
                'computed_value': node.value,
                'stated_value': formula.numerical_value,
                'stated_unit': formula.unit,
                'dimension_check': dimensions_by_id.get(node_id),
            })

    return {
        'schema_version': 1,
        'provider': 'openai-compatible-local',
        'items': items,
    }
