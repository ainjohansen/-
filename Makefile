.PHONY: all audit audit-unit-test dag-audit zt-audit zt-agent-audit zero book pdf lean test help

help:
	@echo "КОМАНДЫ ПРОЕКТА ТЭВ v5.6:"
	@echo "  make audit    - Полный аудит: DAG (@tev) + Zero-Trust (сырой LaTeX)"
	@echo "  make dag-audit- Аудит DAG по тегам @tev (dag_compiler.py)"
	@echo "  make zt-audit - Zero-trust аудит формул монолита (scripts/run_audit.py)"
	@echo "  make zt-agent-audit - Дополнить аудит локальным LLM-агентом (sage-a1.5, до 3 потоков)"
	@echo "  make audit-unit-test - Модульные тесты парсера и интеграции агента"
	@echo "  make zero     - Alias для make zt-audit"
	@echo "  make lean     - Компиляция модулей ядра Lean 4"
	@echo "  make book     - Сборка монолита dist/Manifest-latest-rus.tex из глав"
	@echo "  make pdf      - Сборка монолита и компиляция PDF"
	@echo "  make sync     - Проверка синхронизации ссылок LaTeX <-> Lean 4"
	@echo "  make test     - Полный цикл проверок перед коммитом (Lean + Audit + Sync)"
	@echo "  make help     - Этот справочник"

# 1. Полный аудит: DAG-граф по тегам @tev + zero-trust аудит сырого LaTeX
audit: dag-audit zt-audit
	@echo "🎉 ПОЛНЫЙ АУДИТ ЗАВЕРШЁН (DAG + Zero-Trust)"

# 2. DAG-аудит: сверка @tev тегов манифеста с YAML-словарём
dag-audit:
	python3 scripts/dag_compiler.py

# 3. Zero-trust аудит: парсинг формул напрямую из .tex (без доверия к тегам)
zt-audit:
	@echo "Запуск zero-trust аудита на dist/Manifest-latest-rus.tex ..."
	python3 scripts/run_audit.py $(abspath dist/Manifest-latest-rus.tex) $(abspath data/anchors.yaml)
	@echo "Отчёт: scripts/audit_report_v3.json"

# Опциональный анализ только формул, отмеченных локальным аудитом.
zt-agent-audit:
	@echo "Zero-trust аудит с дополнительным анализом Sage (sage-a1.5) ..."
	python3 scripts/run_audit.py $(abspath dist/Manifest-latest-rus.tex) $(abspath data/anchors.yaml) --agent
	@echo "Отчёт: scripts/audit_report_v3.json"

# 4. Alias (обратная совместимость)
zero: zt-audit

# 5. Проверка формального ядра Lean 4
lean:
	cd lean && lake build

# 6. Проверка соответствия LaTeX <-> Lean
sync:
	python3 scripts/check_lean_sync.py

# 7. Сборка монолитного файла книги для публикации
book:
	python3 scripts/assemble_book.py

# 8. Сборка книги с компиляцией PDF
pdf:
	python3 scripts/assemble_book.py --pdf

# 9. Полный прогон всех тестов перед коммитом в репо
audit-unit-test:
	PYTHONPATH=scripts python3 -m unittest discover -s scripts/tests -v

test: audit-unit-test lean dag-audit zt-audit sync
	@echo "🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО! МОЖНО ДЕЛАТЬ КОММИТ."

# Разбивка монолита статьи на главы
split:
	python3 scripts/split_monolith.py
