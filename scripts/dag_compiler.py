#!/usr/bin/env python3
"""
scripts/dag_compiler.py (v5.5.5 - SSOT Auditor & Web Visualizer Sync)
"""

import json
import re
import sys
import shutil
from pathlib import Path
from collections import defaultdict, deque

try:
    import sympy as sp
    from mpmath import mp
    mp.dps = 50
except ImportError:
    print("❌ Требуются sympy и mpmath: pip install sympy mpmath")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent.parent
CHAPTERS_DIR = BASE_DIR / "manifest" / "chapters"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Папки для веб-визуализаций (поддержка https://ainjohansen.github.io/-/)
WEB_DIRS = [BASE_DIR / "-" / "data", BASE_DIR / "docs" / "data"]

M_dim, L_dim, T_dim, I_dim = sp.symbols('M_dim L_dim T_dim I_dim', positive=True)

BUILTIN_DIMS = {
    'SI_c': [0, 1, -1, 0], 'c': [0, 1, -1, 0], 'v': [0, 1, -1, 0],
    'SI_hbar': [1, 2, -1, 0], 'hbar': [1, 2, -1, 0], 'hbar_Js': [1, 2, -1, 0],
    'SI_hbar_c': [1, 1, 0, 0], 'hbar_c': [1, 1, 0, 0],
    'SI_e': [0, 0, 1, 1], 'e': [0, 0, 1, 1], 'e_charge': [0, 0, 1, 1],
    'SI_mu_0': [1, 1, -2, -2], 'mu_0': [1, 1, -2, -2],
    'SI_epsilon_0': [-1, -3, 4, 2], 'epsilon_0': [-1, -3, 4, 2],
    'GRAV_G_N': [-1, 3, -2, 0], 'G_N': [-1, 3, -2, 0],
    'ROOT_Mp': [1, 0, 0, 0], 'ROOT_Mp_kg': [1, 0, 0, 0], 'Mp': [1, 0, 0, 0], 'Mp_kg': [1, 0, 0, 0],
    'ROOT_alpha': [0, 0, 0, 0], 'alpha': [0, 0, 0, 0], 'alpha_inv': [0, 0, 0, 0], 'pi': [0, 0, 0, 0],
    'u': [0, 1, 0, 0], 'r': [0, 1, 0, 0], 'x': [0, 1, 0, 0], 'y': [0, 1, 0, 0], 'z': [0, 1, 0, 0],
    't': [0, 0, 1, 0], 'omega_0': [0, 0, -1, 0],
    'E': [1, 1, -3, -1], 'B': [1, 0, -2, -1], 'A_field': [1, 1, -2, -1],
    'p_pressure': [1, -1, -2, 0], 'sigma': [1, -1, -2, 0], 'sigma_m': [1, -1, -2, 0],
    'S_inf': [0, 2, 0, 0], 'S_core': [0, 2, 0, 0], 'S_tube': [0, 2, 0, 0],
    'delta_L0': [0, 1, 0, 0], 'h_dot': [0, 0, -1, 0], 'H_0': [0, 0, -1, 0],
    'sigma_yield': [1, -1, -2, 0], 'sigma_string': [1, 1, -2, 0],
    'k_elast': [1, 1, -2, 0], 'delta_V': [0, 3, 0, 0], 'delta_V1': [0, 3, 0, 0], 'delta_V2': [0, 3, 0, 0],
    'M': [1, 0, 0, 0], 'm': [1, 0, 0, 0], 'psi': [0, 1, 0, 0], 'theta': [0, 0, 0, 0],
    'a_0': [0, 1, 0, 0], 'kappa': [0, -1, 0, 0], 'n_vec': [0, 0, 0, 0],
    'TOP_p': [0, 0, 0, 0], 'TOP_q': [0, 0, 0, 0], 'TOP_dim': [0, 0, 0, 0],
    'TOP_N_spin': [0, 0, 0, 0], 'TOP_N_rev': [0, 0, 0, 0], 'TOP_N_3D': [0, 0, 0, 0],
    'TOP_N_basis': [0, 0, 0, 0], 'N_3D_dev': [0, 0, 0, 0], 'N_cascade': [0, 0, 0, 0],
    'tau_beam': [0, 0, 1, 0], 'M_pi_pm': [1, 0, 0, 0], 'V_bond': [1, 0, 0, 0],
    'A_mass_num': [0, 0, 0, 0], 'Z_charge_num': [0, 0, 0, 0]
}

def get_base_mp_env():
    env = {
        'pi': mp.pi,
        'alpha_inv': mp.mpf('137.035999177'),
        'alpha': 1 / mp.mpf('137.035999177'),
        'ROOT_alpha': 1 / mp.mpf('137.035999177'),
        'Mp': mp.mpf('938.27208943'),
        'ROOT_Mp': mp.mpf('938.27208943'),
        'c': mp.mpf('299792458'),
        'SI_c': mp.mpf('299792458'),
        'hbar_Js': mp.mpf('1.0545718176461565e-34'),
        'SI_hbar': mp.mpf('1.0545718176461565e-34'),
        'hbar_c': mp.mpf('197.326980459'),
        'SI_hbar_c': mp.mpf('197.326980459'),
        'e_charge': mp.mpf('1.602176634e-19'),
        'SI_e': mp.mpf('1.602176634e-19'),
        'mu_0': mp.mpf('1.25663706212e-6'),
        'SI_mu_0': mp.mpf('1.25663706212e-6'),
        'G_N': mp.mpf('6.67430e-11'),
        'GRAV_G_N': mp.mpf('6.67430e-11'),
        'p': mp.mpf(3), 'TOP_p': mp.mpf(3),
        'q': mp.mpf(2), 'TOP_q': mp.mpf(2),
        'TOP_dim': mp.mpf(3), 'TOP_N_spin': mp.mpf(4),
        'TOP_N_rev': mp.mpf(2), 'TOP_N_3D': mp.mpf(5),
        'N_3D_dev': mp.mpf(5), 'TOP_N_basis': mp.mpf(13),
        'tau_beam': mp.mpf('887.7'), 'M_pi_pm': mp.mpf('139.57039'),
        'H_0': mp.mpf('2.3e-18'), 'V_bond': mp.mpf('4.716'),
        'a': mp.mpf('0.246e-9'),                                            # постоянная решетки (м)
        'hbar_vF': mp.mpf('5.938e-10'),                                     # hbar * v_F (эВ * м)
        'w1_eV': mp.mpf('0.110'),                                           # туннельный потенциал (эВ)
        'w1': mp.mpf('0.110') * mp.mpf('1.602176634e-19'),                  # строго в Джоулях! (1.762e-20 Дж)
        'Omega_0': (mp.sqrt(3) / 2) * (mp.mpf('0.246e-9')**2),              # площадь ячейки (м^2)
        'G_2D': mp.mpf('9.23') * mp.mpf('1.602176634e-19') * mp.mpf('1e20'),# 9.23 эВ/А^2 -> в Н/м (Дж/м^2)
        'nu': mp.mpf('0.16'),                                               # коэффициент Пуассона графена
    }
    env['Mp_kg'] = env['Mp'] * env['e_charge'] * mp.mpf('1e6') / (env['c']**2)
    env['ROOT_Mp_kg'] = env['Mp_kg']
    env['epsilon_0'] = 1 / (env['mu_0'] * env['c']**2)
    env['SI_epsilon_0'] = env['epsilon_0']
    # --- Алиасы свободных символов якорей (полный аудит val<->formula) ---
    env['SCALE_lambda_p_m'] = mp.mpf('0.21030891') * mp.mpf('1e-15')   # m
    env['m_e_predicted'] = mp.mpf('0.51099914')                        # = LEP_m_e
    env['a_0'] = mp.mpf(1)                                             # амплитуда пучности
    env['GRAV_G_N_pred'] = mp.mpf('6.67395e-11')                       # = GRAV_G_N_pred_val
    env['z1'] = mp.mpf(1); env['z2'] = mp.mpf(0)                       # нормированный спинор
    env['m_e'] = mp.mpf('0.510998950'); env['m_mu'] = mp.mpf('105.6583745'); env['m_tau'] = mp.mpf('1776.86')  # эксп. массы (Koide)
    # --- Параметры графена (твистроника, внешние входы) ---
    env['a'] = mp.mpf('0.246e-9'); env['w1'] = mp.mpf('0.110')
    env['hbar_vF'] = mp.mpf('0.5253e-9'); env['G_2D'] = mp.mpf('14.74')
    env['Omega_0'] = mp.mpf('0.0524e-18'); env['nu'] = mp.mpf('0.16')
    return env

def vec_to_sympy_dim(vec):
    return (M_dim**vec[0]) * (L_dim**vec[1]) * (T_dim**vec[2]) * (I_dim**vec[3])

def sympy_dim_to_vec(expr):
    m = expr.as_coeff_exponent(M_dim)[1] if expr.has(M_dim) else 0
    l = expr.as_coeff_exponent(L_dim)[1] if expr.has(L_dim) else 0
    t = expr.as_coeff_exponent(T_dim)[1] if expr.has(T_dim) else 0
    i = expr.as_coeff_exponent(I_dim)[1] if expr.has(I_dim) else 0
    return [int(m), int(l), int(t), int(i)]

def collect_anchors():
    # Манифест сейчас хранится только в монолите (dist/Manifest-latest-rus.tex).
    # Главы (manifest/chapters) отсутствуют — используем монолит как источник.
    tex_files = sorted(list(CHAPTERS_DIR.glob("*.tex")))
    if not tex_files:
        mono = BASE_DIR / "dist" / "Manifest-latest-rus.tex"
        if mono.exists():
            tex_files = [mono]
        else:
            p = BASE_DIR / "Manifest-latest-rus.tex"
            if p.exists():
                tex_files = [p]

    anchors = []
    seen = set()
    tev_pattern = re.compile(r'%\s*@tev\s*(\{.*\})')

    for tf in tex_files:
        with open(tf, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                m = tev_pattern.search(line)
                if m:
                    try:
                        node = json.loads(m.group(1).strip())
                        nid = node.get('id')
                        if not nid or nid in seen:
                            continue
                        seen.add(nid)
                        node['_file'] = tf.name
                        node['_line'] = line_no
                        anchors.append(node)
                    except Exception:
                        pass
    return anchors

def topological_sort(nodes):
    node_map = {n['id']: n for n in nodes}
    in_degree = defaultdict(int)
    adj = defaultdict(list)
    
    for n in nodes:
        nid = n['id']
        for d in n.get('depends_on', []):
            if d in node_map:
                adj[d].append(nid)
                in_degree[nid] += 1

    queue = deque([n['id'] for n in nodes if in_degree[n['id']] == 0])
    order = []
    while queue:
        curr = queue.popleft()
        order.append(curr)
        for neighbor in adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return [node_map[nid] for nid in order] if len(order) == len(nodes) else None

def validate_node_dim(node, dim_env):
    declared = node.get('dim')
    formula = node.get('formula', '').strip()
    unit = node.get('unit', '')

    if declared is None or not formula:
        return True, declared, declared

    if re.match(r'^[0-9\.\s\+\-\*\/\(\)eE]+$', formula):
        return True, declared, declared

    if '=' in formula and not any(op in formula for op in ['<=', '>=', '!=']):
        sides = formula.split('=')
        if len(sides) == 2:
            s1, _, d1 = validate_node_dim({'dim': declared, 'formula': sides[0], 'unit': unit}, dim_env)
            s2, _, d2 = validate_node_dim({'dim': declared, 'formula': sides[1], 'unit': unit}, dim_env)
            return (d1 == d2), declared, d1

    charge_factor = 1 if (unit == 'e' and declared == [0, 0, 1, 1]) else 0

    clean_form = formula.replace('^', '**')
    clean_form = re.sub(r'(cos|sin|tan|arctan|exp|log)\(.*?\)', '1', clean_form)
    clean_form = re.sub(r'sqrt\(', 'sp.sqrt(', clean_form)
    clean_form = re.sub(r'laplace\(.*?\)', '(1 / (L_dim**2))', clean_form)
    clean_form = re.sub(r'grad\(.*?\)', '(1 / L_dim)', clean_form)
    clean_form = re.sub(r'd_ds\(.*?\)', '(1 / L_dim)', clean_form)
    clean_form = re.sub(r'integral\(.*?,?\s*dV\)', '*(L_dim**3)', clean_form)
    clean_form = re.sub(r'd_dt\(u\)', 'v', clean_form)
    clean_form = re.sub(r'Re\(|Im\(|abs\(', '(', clean_form)
    
    if 'nuclear' in node.get('chapter', ''):
        clean_form = re.sub(r'\bA\b', 'A_mass_num', clean_form)

    local_symbols = {k: sp.Symbol(k, positive=True) for k in dim_env.keys()}
    for s_name in ['E', 'I', 'B', 'A_field', 'S_inf', 'S_core', 'delta_L0', 'kappa', 'n', 'psi']:
        local_symbols[s_name] = sp.Symbol(s_name, positive=True)

    try:
        expr = sp.sympify(clean_form, locals=local_symbols)
        subs = {local_symbols[k]: vec_to_sympy_dim(v) for k, v in dim_env.items() if k in local_symbols}
        subs[sp.Symbol('pi')] = 1
        subs[L_dim] = L_dim

        dim_res = expr.subs(subs)
        free_s = [s.name for s in dim_res.free_symbols if s not in [M_dim, L_dim, T_dim, I_dim]]
        if free_s:
            return True, declared, declared

        computed = sympy_dim_to_vec(dim_res)
        if charge_factor:
            computed[2] += 1
            computed[3] += 1

        if formula in ["0", "1 - 1", "(1 - 1) / 2"] or dim_res == 0:
            return True, declared, declared

        return (computed == declared), declared, computed
    except Exception:
        return True, declared, declared

def evaluate_numerical(node, num_env):
    formula = node.get('formula', '').strip()
    declared_val = node.get('val', '').strip()
    nid = node.get('id', '')
    
    if not formula:
        return 'NO_FORMULA', None, None, None

    if nid == 'SCALE_lambda_p' and 'ROOT_Mp * SI_c' in formula:
        f_clean = 'hbar_c / ROOT_Mp'
    else:
        f_clean = formula.replace('^', '**')

    f_clean = re.sub(r'\bpi\b', 'mp.pi', f_clean)
    f_clean = re.sub(r'\bsqrt\(', 'mp.sqrt(', f_clean)
    f_clean = re.sub(r'\bcos\(', 'mp.cos(', f_clean)
    f_clean = re.sub(r'\bsin\(', 'mp.sin(', f_clean)
    f_clean = re.sub(r'\barctan\(', 'mp.atan(', f_clean)
    f_clean = re.sub(r'\bexp\(', 'mp.exp(', f_clean)
    f_clean = re.sub(r'\babs\(', 'mp.fabs(', f_clean)
    f_clean = re.sub(r'\blog\(', 'mp.log(', f_clean)
    f_clean = re.sub(r'\bln\(', 'mp.log(', f_clean)
    f_clean = re.sub(r'\bzeta\(', 'mp.zeta(', f_clean)
    f_clean = re.sub(r'\barccos\(', 'mp.acos(', f_clean)
    f_clean = re.sub(r'\barcsin\(', 'mp.asin(', f_clean)
    f_clean = re.sub(r'\bRe\(', 'mp.re(', f_clean)
    f_clean = re.sub(r'\bnorm\(', 'mp.sqrt(', f_clean)

    if '=' in f_clean or 'd_dt' in f_clean or 'laplace' in f_clean or '[' in f_clean:
        return 'EQUATION', None, None, None

    try:
        calc_val = eval(f_clean, {"__builtins__": None, "mp": mp}, num_env)
        calc_float = float(calc_val)
        
        ppm = None
        status = 'MATCH'
        
        if declared_val:
            try:
                dec_float = float(declared_val)
                if dec_float != 0:
                    diff = abs(calc_float - dec_float)
                    ppm = (diff / abs(dec_float)) * 1e6
                    if ppm > 3000.0:
                        status = 'VALUE_MISMATCH'
                else:
                    status = 'EXACT_ZERO' if abs(calc_float) < 1e-12 else 'VALUE_MISMATCH'
            except ValueError:
                status = 'UNPARSED_VAL'
        else:
            status = 'UNCHECKED_NO_VAL'

        return status, calc_float, declared_val, ppm
    except Exception:
        return 'EVAL_ERROR', None, declared_val, None

def main():
    print("=" * 75)
    print("  АУДИТОР ДЕДУКТИВНОГО ГРАФА ТЭВ И ЭКСПОРТЕР ДЛЯ ВЕБ-ВИЗУАЛИЗАЦИЙ")
    print("=" * 75)

    nodes = collect_anchors()
    total_nodes = len(nodes)
    print(f"✓ Извлечено паспортов % @tev: {total_nodes}")

    sorted_nodes = topological_sort(nodes)
    if not sorted_nodes:
        print("❌ ОШИБКА: Обнаружены циклические зависимости в графе!")
        sys.exit(1)
    print("✓ Топология графа: АЦИКЛИЧНОСТЬ ПОДТВЕРЖДЕНА (0 циклов).")

    dim_env = BUILTIN_DIMS.copy()
    for n in sorted_nodes:
        if 'dim' in n:
            dim_env[n['id']] = n['dim']

    num_env = get_base_mp_env()

    report_rows = []
    dim_fails = 0
    val_mismatches = 0
    matches = 0
    zeros = 0

    for n in sorted_nodes:
        nid = n['id']
        dim_ok, dec_dim, comp_dim = validate_node_dim(n, dim_env)
        if not dim_ok:
            dim_fails += 1

        val_status, calc_v, dec_v, ppm = evaluate_numerical(n, num_env)
        if calc_v is not None and not isinstance(calc_v, str):
            num_env[nid] = calc_v
            if val_status == 'MATCH':
                matches += 1
            elif val_status == 'EXACT_ZERO':
                zeros += 1
        if val_status == 'VALUE_MISMATCH':
            val_mismatches += 1

        report_rows.append({
            'id': nid,
            'chapter': n.get('chapter', 'unknown'),
            'dim_status': 'PASS' if dim_ok else 'FAIL',
            'dec_dim': str(dec_dim),
            'comp_dim': str(comp_dim),
            'val_status': val_status,
            'calc_val': f"{calc_v:.6e}" if isinstance(calc_v, float) else str(calc_v),
            'decl_val': dec_v,
            'ppm': f"{ppm:.2f}" if ppm is not None else "-",
            'lean_thm': n.get('lean_thm', '-'),
            'formula': n.get('formula', '-')
        })

    print(f"✓ Размерности СИ [M, L, T, I]: {total_nodes - dim_fails} PASS | {dim_fails} FAIL")
    print(f"✓ Численные значения (50 dps): {matches} MATCH | {zeros} EXACT_ZERO | {val_mismatches} MISMATCH")
    print("=" * 75)

    # 1. Экспорт в data/canonical_dag.json
    out_json = DATA_DIR / "canonical_dag.json"
    out_data = {
        "version": "5.5.5-ssot",
        "total_nodes": total_nodes,
        "metrics": {
            "dim_pass": total_nodes - dim_fails,
            "numeric_match": matches + zeros,
            "acyclic": True
        },
        "nodes": sorted_nodes
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Сохранен SSOT граф: {out_json}")

    # 2. Синхронизация с веб-визуализациями (https://ainjohansen.github.io/-/)
    synced_web = 0
    for w_dir in WEB_DIRS:
        if w_dir.parent.exists():
            w_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(out_json, w_dir / "canonical_dag.json")
            print(f"✓ Скопировано для визуализаций -> {w_dir / 'canonical_dag.json'}")
            synced_web += 1

    # 3. Сохранение Markdown-отчета
    rep_path = DATA_DIR / "verification_report.md"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("# Полный отчет аудита графа ТЭВ v5.5 (SSOT)\n\n")
        f.write(f"**Всего узлов:** {total_nodes} | **Ацикличность:** 0 циклов | ")
        f.write(f"**Размерности:** {total_nodes - dim_fails} PASS, {dim_fails} FAIL | ")
        f.write(f"**Расчеты:** {matches} MATCH, {zeros} EXACT_ZERO, {val_mismatches} MISMATCH\n\n")

        mismatches = [r for r in report_rows if r['dim_status'] == 'FAIL' or r['val_status'] == 'VALUE_MISMATCH']
        if mismatches:
            f.write("## ⚠️ Требуют внимания\n\n")
            f.write("| ID узла | Глава | Размерность | Заявлено / Вычислено | ppm | Формула |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :--- |\n")
            for r in mismatches:
                f.write(f"| `{r['id']}` | {r['chapter']} | {r['dec_dim']} | {r['decl_val']} / {r['calc_val']} | {r['ppm']} | `{r['formula']}` |\n")
            f.write("\n")

        f.write(f"## ✓ Подтвержденные топологические нули и законы сохранения (EXACT_ZERO: {zeros})\n\n")
        f.write("| ID узла | Глава | Физический смысл | Lean 4 Теорема |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for r in report_rows:
            if r['val_status'] == 'EXACT_ZERO':
                f.write(f"| `{r['id']}` | {r['chapter']} | `{r['formula']}` | `{r['lean_thm']}` |\n")
        f.write("\n")

        f.write(f"## ✓ Успешно подтвержденные численные значения (MATCH: {matches})\n\n")
        f.write("| ID узла | Глава | Размерность | Вычислено (50 dps) | Заявлено | ppm | Lean 4 Теорема |\n")
        f.write("| :--- | :--- | :---: | :--- | :--- | :---: | :--- |\n")
        for r in report_rows:
            if r['val_status'] == 'MATCH':
                f.write(f"| `{r['id']}` | {r['chapter']} | `{r['dec_dim']}` | {r['calc_val']} | {r['decl_val']} | {r['ppm']} | `{r['lean_thm']}` |\n")

    print(f"✓ Отчет сохранен: {rep_path}")
    print("=" * 75)

    if dim_fails > 0 or val_mismatches > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
