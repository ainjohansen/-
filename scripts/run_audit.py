"""
Zero-trust audit runner for TEV project (v3 with type dispatcher).

Usage: python3 run_audit_v3.py <path_to_tex_or_dir> [anchors.yaml] [--strict] [--agent]

The optional --agent flag sends flagged formulas to a local OpenAI-compatible
LLM endpoint (http://127.0.0.1:1234/v1/chat/completions, model sage-a1.5).
Concurrency is capped at 3 parallel requests; failures are recorded, not raised,
so the deterministic audit always completes.

Uses formula_graph_v3 which classifies formulas into 5 types before processing:
- ALGEBRAIC: scalar arithmetic (full numeric eval + value comparison)
- DIFFERENTIAL: PDEs with \\nabla/\\partial operators (dimension check only)
- INTEGRAL: integral conservation laws (dimension check only)
- INEQUALITY: boolean conditions >, <, >=, <= (truth check if computable)
- LIMIT: asymptotic limits with \to (converted to equality for dim analysis)

Exit codes:
  0 - no parse/eval errors, no dimension MISMATCH
  1 - parse/eval errors or dimension mismatches present (or --strict: also
      value mismatches)
"""

import sys
import os
import json
from datetime import datetime

from latex_formula_parser import LaTeXFormulaParser, parse_all_files
from formula_graph_v3 import FormulaGraph, run_audit
from agent_review import review_flagged_formulas


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = {a for a in sys.argv[1:] if a.startswith('--')}
    strict = '--strict' in flags
    use_agent = '--agent' in flags

    if len(args) < 1:
        print("Usage: python3 run_audit.py <path_to_tex_or_dir> "
              "[anchors.yaml] [--strict] [--agent]")
        sys.exit(2)

    target = args[0]
    anchors_path = args[1] if len(args) > 1 else "anchors.yaml"

    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(target):
        target = os.path.join(script_dir, target)
    if not os.path.isabs(anchors_path):
        anchors_path = os.path.join(script_dir, anchors_path)

    # Collect .tex files
    tex_files = []
    if os.path.isfile(target) and target.endswith('.tex'):
        tex_files.append(target)
    elif os.path.isdir(target):
        for root, dirs, files in os.walk(target):
            for fname in sorted(files):
                if fname.endswith('.tex'):
                    tex_files.append(os.path.join(root, fname))
    else:
        print(f"Error: {target} is not a valid file or directory")
        sys.exit(2)

    print("=" * 70)
    print("ZERO-TRUST AUDIT v3 (type dispatcher): TEV Formula Verification System")
    print("=" * 70)
    print(f"Target files: {len(tex_files)} .tex files")
    for f in tex_files:
        print(f"  - {f}")
    print(f"Anchors: {anchors_path}")
    print()

    # Run audit
    result = run_audit(tex_files, anchors_path)
    # The agent is a *supplement* to the deterministic audit. It must never
    # crash the run when the local endpoint is unreachable, so failures are
    # recorded and reported, not raised.
    agent_error = None
    if use_agent:
        try:
            analysis = review_flagged_formulas(result['agent_review'])
            result['agent_review']['analysis'] = analysis
            if analysis['failed_batches']:
                agent_error = RuntimeError(
                    f"Agent review failed for {len(analysis['failed_batches'])} batch(es)"
                )
        except (RuntimeError, ValueError, KeyError) as error:
            # Endpoint unreachable / malformed response / bad queue shape.
            agent_error = RuntimeError(f'Agent review unavailable: {error}')
            result['agent_review']['analysis'] = {
                'provider': 'openai-compatible-local',
                'endpoint': 'http://127.0.0.1:1234/v1/chat/completions',
                'model': 'sage-a1.5',
                'reviewed': 0,
                'requested': len(result['agent_review'].get('items', [])),
                'results': [],
                'failed_batches': [{'batch': 0, 'ids': [i['id'] for i in result['agent_review']['items']], 'attempts': 1, 'error': str(error)}],
            }

    stats = result['stats']

    # Print statistics
    print("=" * 70)
    print("AUDIT STATISTICS")
    print("=" * 70)
    print(f"Total formulas parsed:   {stats['total_formulas']}")
    print(f"Resolved (scalar):       {stats['resolved']}")
    print(f"Resolved (matrix):       {stats.get('matrix_resolved', 0)}")
    print(f"Symbolic definitions:    {stats.get('symbolic', 0)}")
    print(f"Evaluation errors:       {stats['errors']}")
    print(f"Dimension checks OK:     {stats.get('dim_ok', 0)}")
    print(f"Dimension MISMATCH:      {stats.get('dim_mismatch', 0)}")
    print(f"Dimension UNKNOWN/PART:  {stats.get('dim_unknown', 0)}")
    print(f"Value mismatches (>1%):  {len(stats['value_mismatches'])}")
    print(f"Formula type counts:     {stats.get('type_counts', {})}")
    print()

    # Print dimension conflicts (cross-formula)
    if result.get('dim_conflicts'):
        print("-" * 70)
        print("DIMENSION CONFLICTS (same variable, different dims across formulas):")
        print("-" * 70)
        for err in result['dim_conflicts']:
            print(f"  {err}")
        print()

    # Print per-formula dimension mismatches
    dim_bad = [e for e in result.get('dimension_report', []) if e['status'] == 'MISMATCH']
    if dim_bad:
        print("-" * 70)
        print("DIMENSION MISMATCHES (LHS vs RHS):")
        print("-" * 70)
        for e in dim_bad:
            print(f"  {e['id']} ({e['lhs']}): LHS={e['lhs_dim']} RHS={e['rhs_dim']}")
        print()

    # Print value mismatches
    if stats['value_mismatches']:
        print("-" * 70)
        print("VALUE MISMATCHES (computed vs stated, >1%):")
        print("-" * 70)
        for mm in stats['value_mismatches']:
            print(f"  {mm}")
        print()

    # Print external params used
    if stats['external_params_used']:
        print("-" * 70)
        print("EXTERNAL PARAMETERS USED (not derivable from alpha/M_p):")
        print("-" * 70)
        for param in sorted(set(stats['external_params_used'])):
            print(f"  - {param}")
        print()

    # Print inferred dimensions table
    if result.get('inferred_dims'):
        print("-" * 70)
        print("INFERRED VARIABLE DIMENSIONS (from formula structure):")
        print("-" * 70)
        for var, dim in sorted(result['inferred_dims'].items()):
            print(f"  {var}: {dim}")
        print()

    # Print resolved formulas summary
    print("-" * 70)
    print("RESOLVED FORMULAS:")
    print("-" * 70)
    for nid, node in result['nodes'].items():
        if node['status'] == 'resolved':
            if node.get('matrix_value') is not None:
                mv = node['matrix_value']
                pretty = [[f"{c:.4g}" if c is not None else '?' for c in row] for row in mv]
                print(f"  {nid}: MATRIX {pretty}")
            elif node['value'] is not None:
                val = node['value']
                if abs(val) < 1e-6:
                    val_str = f"{val:.3g}"
                elif abs(val) < 1:
                    val_str = f"{val:.4f}"
                elif abs(val) < 1000:
                    val_str = f"{val:.4g}"
                else:
                    val_str = f"{val:.3e}"
                print(f"  {nid}: {val_str}")

    # Save full report (timestamped + latest)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(script_dir, "audit_report_v3.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    ts_path = os.path.join(script_dir, f"audit_report_v3_{ts}.json")
    with open(ts_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print(f"Full report saved to: {report_path}")
    print(f"Timestamped copy:     {ts_path}")

    if use_agent:
        analysis = result['agent_review'].get('analysis') or {}
        if agent_error is None and analysis.get('reviewed'):
            print(
                "Local agent reviewed "
                f"{analysis['reviewed']} of "
                f"{analysis['requested']} flagged formulas."
            )
        if agent_error is not None:
            print(f"Local agent review failed: {agent_error}", file=sys.stderr)
            for batch in analysis.get('failed_batches', []):
                print(
                    f"  {', '.join(batch['ids'])}: {batch['error']}",
                    file=sys.stderr,
                )

    # Exit code based on errors
    fail = stats['errors'] > 0 or stats.get('dim_mismatch', 0) > 0
    if strict and len(stats['value_mismatches']) > 0:
        fail = True
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
