#!/usr/bin/env python3
"""
scripts/assemble_book.py (v5.5.6 - Полноценная структурная сборка LaTeX)
Схема сборки:
Preamble -> Chapters (01..18) -> \appendix -> Appendices -> Bibliography -> \end{document}
"""

import sys
import os
import re
import subprocess
import glob
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_DIR = BASE_DIR / "manifest"
CHAPTERS_DIR = MANIFEST_DIR / "chapters"
DIST_DIR = BASE_DIR / "dist"
DIST_DIR.mkdir(exist_ok=True)

PREAMBLE_FILE = MANIFEST_DIR / "preamble.tex"
OUTPUT_TEX = DIST_DIR / "Manifest-latest-rus.tex"

def clean_chunk(text: str) -> str:
    """Удаляет случайные команды закрытия документа внутри чанков."""
    text = re.sub(r'\\end\{document\}', '', text)
    return text.strip()

def find_tex_bin():
    """Находит каталог с рабочей TeX Live (pdflatex + latexmk).

    Системный TeX Live в этой среде установлен без скомпилированных форматов
    (пустые fmtutil.cnf), поэтому pdflatex падает с "can't find pdflatex.fmt".
    Рабочие форматы есть в flatpak-рантайме TeX Live — используем его бинарники.
    Возвращает путь к каталогу bin (или None, если ничего не найдено).
    """
    base = "/var/lib/flatpak/runtime/org.freedesktop.Sdk.Extension.texlive"
    candidates = []
    # 1) Стабильный путь 'active' flatpak-рантайма (предпочтительно, самый свежий).
    candidates += sorted(glob.glob(base + "/x86_64/*/active/files/bin/x86_64-linux"))
    # 2) Запасной вариант — хэш-каталоги той же версии.
    candidates += sorted(glob.glob(base + "/x86_64/*/*/files/bin/x86_64-linux"))
    for d in candidates:
        if os.path.exists(os.path.join(d, "pdflatex")) and os.path.exists(os.path.join(d, "latexmk")):
            return d
    return None


def assemble_book(build_pdf=False):
    print("=" * 75)
    print("  СТРУКТУРНАЯ СБОРКА МОНОЛИТА ТЭВ")
    print("=" * 75)

    if not PREAMBLE_FILE.exists():
        print(f"❌ Не найден файл преамбулы: {PREAMBLE_FILE}")
        sys.exit(1)

    with open(PREAMBLE_FILE, "r", encoding="utf-8") as f:
        preamble = clean_chunk(f.read())

    all_files = list(CHAPTERS_DIR.glob("*.tex"))
    if not all_files:
        print(f"❌ В {CHAPTERS_DIR} нет файлов!")
        sys.exit(1)

    # 1. Раскладываем файлы по структурным корзинам
    main_chapters = []
    appendix_files = []
    bib_files = []

    for f in all_files:
        name = f.name.lower()
        if "biblio" in name or name.startswith("99_"):
            bib_files.append(f)
        elif name.startswith("app_") or name.startswith("appendix") or "_app_" in name:
            appendix_files.append(f)
        else:
            main_chapters.append(f)

    main_chapters.sort(key=lambda x: x.name)
    appendix_files.sort(key=lambda x: x.name)
    bib_files.sort(key=lambda x: x.name)

    print(f"✓ Основных глав: {len(main_chapters)}")
    print(f"✓ Приложений:     {len(appendix_files)}")
    print(f"✓ Библиография:   {len(bib_files)}")

    # 2. Последовательная сборка
    doc_parts = [preamble, "\n\n% ==========================================\n% ОСНОВНЫЕ ГЛАВЫ МОДЕЛИ\n% ==========================================\n"]

    for ch in main_chapters:
        print(f"   [Глава]        {ch.name}")
        with open(ch, "r", encoding="utf-8") as cf:
            doc_parts.append(f"\n% === {ch.name} ===\n" + clean_chunk(cf.read()) + "\n")

    if appendix_files:
        doc_parts.append("\n\n% ==========================================\n\\appendix\n% ==========================================\n")
        for app in appendix_files:
            print(f"   [Приложение]   {app.name}")
            with open(app, "r", encoding="utf-8") as af:
                doc_parts.append(f"\n% === {app.name} ===\n" + clean_chunk(af.read()) + "\n")

    if bib_files:
        doc_parts.append("\n\n% ==========================================\n% БИБЛИОГРАФИЯ\n% ==========================================\n")
        for bf in bib_files:
            print(f"   [Библиография] {bf.name}")
            with open(bf, "r", encoding="utf-8") as bf_in:
                doc_parts.append(f"\n% === {bf.name} ===\n" + clean_chunk(bf_in.read()) + "\n")

    doc_parts.append("\n\\end{document}\n")

    full_text = "\n".join(doc_parts)
    with open(OUTPUT_TEX, "w", encoding="utf-8") as out:
        out.write(full_text)

    print(f"\n✓ Монолит собран: {OUTPUT_TEX}")

    if build_pdf:
        print("\nКомпиляция PDF через latexmk...")
        env = dict(os.environ)
        env["LC_ALL"] = "C.UTF-8"
        env["LANG"] = "C.UTF-8"
        tex_bin = find_tex_bin()
        if tex_bin:
            env["PATH"] = tex_bin + os.pathsep + env.get("PATH", "")
            print(f"   [TeX Live] {tex_bin}")
        else:
            print("   [TeX Live] flatpak-рантайм не найден, пробую системный pdflatex.")
        try:
            res = subprocess.run(
                ["latexmk", "-pdf", "-interaction=nonstopmode", f"-output-directory={DIST_DIR}", str(OUTPUT_TEX)],
                capture_output=True, env=env
            )
            if res.returncode == 0:
                print(f"🎉 PDF успешно скомпилирован: {DIST_DIR / 'Manifest-latest-rus.pdf'}")
            else:
                print("⚠️ Ошибка компиляции PDF (проверьте dist/Manifest-latest-rus.log).")
                out = (res.stdout or b"").decode("utf-8", "replace")[-3000:]
                err = (res.stderr or b"").decode("utf-8", "replace")[-2000:]
                sys.stderr.write(out + "\n" + err + "\n")
        except FileNotFoundError:
            print("ℹ latexmk не установлен, компиляция PDF пропущена.")

if __name__ == "__main__":
    make_pdf = "--pdf" in sys.argv
    assemble_book(build_pdf=make_pdf)
