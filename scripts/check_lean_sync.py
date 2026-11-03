#!/usr/bin/env python3
"""
scripts/check_lean_sync.py
Zero-Trust Lean Auditor v4.2 (Direct Symbol Matcher):
- Автоматически находит корень репозитория.
- Сканирует ВСЕ .tex файлы (монолит и главы).
- Проверяет реальное наличие доказательств (детектор sorry / admit).
- Ищет имена теорем напрямую по тексту без зависимости от регулярки \\texttt.
"""

import os
import sys
import re
import json
import time
from pathlib import Path

def find_repo_root() -> Path:
    """Ищет корень репозитория по маркерам Lean 4."""
    curr = Path(__file__).resolve().parent
    for p in [curr, curr.parent, curr.parent.parent]:
        if (p / "lakefile.toml").exists() or (p / "lean-toolchain").exists() or (p / "lean").exists():
            return p
    return Path.cwd()

REPO_ROOT = find_repo_root()

# Пути к Lean-модулям
LEAN_DIR = REPO_ROOT / "lean" / "tev"
if not LEAN_DIR.exists():
    LEAN_DIR = REPO_ROOT / "tev"

DATA_DAG = REPO_ROOT / "data" / "canonical_dag.json"

def strip_lean_comments(code: str) -> str:
    """Удаляет строчные и блочные комментарии из Lean-кода."""
    code = re.sub(r'/-(?:.|\n)*?-/', '', code)
    code = re.sub(r'--.*', '', code)
    return code

def extract_lean_declarations(lean_path: Path):
    with open(lean_path, "r", encoding="utf-8") as f:
        clean_code = strip_lean_comments(f.read())

    decl_pattern = re.compile(
        r'^[ \t]*(?:protected\s+|private\s+|noncomputable\s+)*(theorem|lemma|def|axiom)\s+([A-Za-z0-9_]+)',
        re.MULTILINE
    )

    matches = list(decl_pattern.finditer(clean_code))
    declarations = {}

    for i, m in enumerate(matches):
        kind = m.group(1)
        name = m.group(2)
        start_idx = m.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(clean_code)
        body = clean_code[start_idx:end_idx]

        has_sorry = bool(re.search(r'\b(sorry|admit)\b', body))
        is_axiom = (kind == "axiom")

        declarations[name] = {
            "kind": kind,
            "has_sorry": has_sorry,
            "is_axiom": is_axiom,
            "is_proven": not has_sorry and not is_axiom
        }

    return declarations

def find_all_tex_files():
    """Находит абсолютно все .tex файлы в проекте."""
    tex_files = []
    for p in REPO_ROOT.rglob("*.tex"):
        # Исключаем временные файлы сборки
        if ".lake" not in p.parts and "build" not in p.parts:
            tex_files.append(p)
    return sorted(list(set(tex_files)))

def run_zero_trust_audit():
    print("=" * 85)
    print("🛡️   ZERO-TRUST LEAN AUDITOR v4.2: ПОЛНОТЕКСТОВЫЙ АУДИТ ТЭВ")
    print(f"📁  Корень репозитория: {REPO_ROOT}")
    print(f"📁  Каталог Lean 4:     {LEAN_DIR}")
    print("=" * 85)

    if not LEAN_DIR.exists():
        print(f"❌ Каталог Lean не найден: {LEAN_DIR}")
        sys.exit(1)

    lean_files = sorted(list(LEAN_DIR.glob("*.lean")))
    if not lean_files:
        print(f"❌ В {LEAN_DIR} нет файлов .lean!")
        sys.exit(1)

    # 1. Извлекаем все декларации Lean
    all_lean_data = {}
    all_proven_theorems = {} # name -> fname

    for lf in lean_files:
        decls = extract_lean_declarations(lf)
        all_lean_data[lf.name] = decls
        for dname, dinfo in decls.items():
            if dinfo["kind"] in ["theorem", "lemma"] and dinfo["is_proven"]:
                all_proven_theorems[dname] = lf.name

    # 2. Находим и читаем все файлы TeX
    tex_files = find_all_tex_files()
    print(f"\n📄 НАЙДЕНО ТЕХ-ФАЙЛОВ ДЛЯ АНАЛИЗА ({len(tex_files)}):")
    total_tex_content = ""
    for tf in tex_files:
        mtime = time.strftime('%H:%M:%S', time.localtime(tf.stat().st_mtime))
        print(f" • {tf.relative_to(REPO_ROOT)} (изменен: {mtime}, размер: {tf.stat().st_size} байт)")
        with open(tf, "r", encoding="utf-8") as f:
            total_tex_content += "\n" + f.read()

    # Также подтягиваем DAG, если есть
    dag_theorems = set()
    if DATA_DAG.exists():
        try:
            with open(DATA_DAG, "r", encoding="utf-8") as f:
                dag = json.load(f)
            for n in dag.get("nodes", []):
                thm = n.get("lean_thm")
                if thm and thm != "-":
                    dag_theorems.add(thm)
        except Exception:
            pass

    # 3. Проверяем цитируемость КАЖДОЙ доказанной теоремы в TeX
    cited_in_tex = set()
    uncited_theorems = []

    for thm_name, fname in all_proven_theorems.items():
        # Ищем строгое вхождение идентификатора как отдельного слова в TeX или в DAG
        pattern = rf'\b{re.escape(thm_name)}\b'
        if re.search(pattern, total_tex_content) or (thm_name in dag_theorems):
            cited_in_tex.add((fname, thm_name))
        else:
            uncited_theorems.append((fname, thm_name))

    # 4. Вывод сводной таблицы по модулям
    print("\n" + "-" * 85)
    print(f"{'Файл Lean 4':<24} | {'Всего':<5} | {'Доказано':<8} | {'В статье':<8} | {'Забыто':<6} | {'sorry':<5} | {'Статус'}")
    print("-" * 85)

    total_decls = 0
    total_proven = 0
    total_cited = 0
    total_sorry = 0
    critical_errors = 0

    for lf in lean_files:
        fname = lf.name
        decls = all_lean_data[fname]

        f_total = len(decls)
        f_proven = sum(1 for d in decls.values() if d["is_proven"] and d["kind"] in ["theorem", "lemma"])
        f_sorry = sum(1 for d in decls.values() if d["has_sorry"])

        f_cited = sum(1 for (fn, tn) in cited_in_tex if fn == fname)
        f_uncited = sum(1 for (fn, tn) in uncited_theorems if fn == fname)

        total_decls += f_total
        total_proven += f_proven
        total_cited += f_cited
        total_sorry += f_sorry

        # Проверка: не цитируется ли теорема с sorry?
        for dname, dinfo in decls.items():
            if dinfo["has_sorry"] and re.search(rf'\b{re.escape(dname)}\b', total_tex_content):
                critical_errors += 1
                print(f"\n❌ [CRITICAL] В статье цитируется НЕДОКАЗАННАЯ теорема с sorry: {fname}::{dname}")

        status = "✅ OK"
        if f_sorry > 0:
            status = "⚠️ SORRY"
        elif f_uncited == 0:
            status = "🎯 100%"
        elif f_cited == 0 and f_proven > 0:
            status = "💤 SLEEP"

        print(f"{fname:<24} | {f_total:<5} | {f_proven:<8} | {f_cited:<8} | {f_uncited:<6} | {f_sorry:<5} | {status}")

    print("-" * 85)
    print(f"ИТОГО ПО ВСЕМУ ЯДРУ:     | {total_decls:<5} | {total_proven:<8} | {total_cited:<8} | {len(uncited_theorems):<6} | {total_sorry:<5} |")
    print("=" * 85)

    # Вывод оставшихся "спящих" теорем
    if uncited_theorems:
        print(f"\n💡 [СПЯЩИЕ] Доказаны в Lean, но не найдены в TeX ({len(uncited_theorems)}):")
        for f, t in uncited_theorems[:20]:
            print(f"   • {f:<24} ➔ {t}")
        if len(uncited_theorems) > 20:
            print(f"   ... и еще {len(uncited_theorems) - 20} теорем.")
    else:
        print("\n🎉 ПОЛНОЕ ПОКРЫТИЕ! Все 100% доказанных теорем Lean 4 отражены в тексте статьи!")

    print("\n" + "=" * 85)
    if critical_errors == 0 and total_sorry == 0:
        print("🏆 ВЕРДИКТ ZERO-TRUST: ПРОВЕРКА ПРОЙДЕНА!")
        print(f" • Доказано теорем:   {total_proven}")
        print(f" • Отражено в статье: {total_cited} из {total_proven}")
        print(f" • Ошибок (sorry):    0")
        sys.exit(0)
    else:
        print(f"⛔ НАЙДЕНЫ КРИТИЧЕСКИЕ ОШИБКИ: {critical_errors}")
        sys.exit(1)

if __name__ == "__main__":
    run_zero_trust_audit()
