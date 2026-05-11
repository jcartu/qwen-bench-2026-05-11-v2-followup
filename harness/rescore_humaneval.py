#!/usr/bin/env python3
"""rescore_humaneval.py — Re-score historical HumanEval jsonl results.

Applies the v2 smart_glue fix offline:
  - If model returned body-only code (no 'def' keyword), prepend HumanEval prompt's
    signature and re-indent all body lines to ensure consistent indentation
    under the def statement.

Usage:
    python3 rescore_humaneval.py PROBLEMS_FILE INPUT_JSONL [INPUT_JSONL ...]

Prints corrected pass rates. Does NOT modify input files. Generates per-file
side-car summary files: <input>_rescored.json
"""
import json, re, subprocess, sys, os
from pathlib import Path


CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)(?:```|$)", re.DOTALL)


def smart_glue(code: str, prompt: str) -> str:
    """Glue body-only response to the HumanEval prompt signature.

    Re-indents to ensure all body lines are uniformly indented under the def.
    """
    if 'def ' in code:
        return code
    indent = '    '
    fixed = []
    for line in code.split('\n'):
        if line.strip() == '':
            fixed.append('')
        elif line.startswith('    ') or line.startswith('\t'):
            fixed.append(line)
        else:
            fixed.append(indent + line)
    return prompt.rstrip() + '\n' + '\n'.join(fixed)


def run_test(code: str, test_code: str, entry_point: str, timeout_s: int = 10) -> bool:
    full = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    try:
        r = subprocess.run(
            [sys.executable, "-c", full],
            capture_output=True, text=True, timeout=timeout_s
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def rescore(jsonl_path: Path, problems: dict) -> dict:
    orig_pass = 0
    recovered = 0
    total = 0
    failures_remaining = 0
    truncated = 0
    other_empty = 0

    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            total += 1
            mode = r.get('failure_mode')
            if mode == 'ok':
                orig_pass += 1
                continue
            if mode == 'empty_response':
                if r.get('finish_reason') == 'length':
                    truncated += 1
                else:
                    other_empty += 1
                continue
            if mode != 'test_fail' or not r.get('code_extracted'):
                failures_remaining += 1
                continue
            content = r.get('content_head', '')
            m = CODE_FENCE_RE.search(content)
            if not m:
                failures_remaining += 1
                continue
            code = m.group(1).strip()
            prob = problems.get(r['task_id'])
            if not prob:
                failures_remaining += 1
                continue
            glued = smart_glue(code, prob['prompt'])
            if run_test(glued, prob['test'], prob['entry_point']):
                recovered += 1
            else:
                failures_remaining += 1

    new_pass = orig_pass + recovered
    return {
        'jsonl': str(jsonl_path),
        'total': total,
        'orig_pass': orig_pass,
        'orig_pass_rate': orig_pass / total if total else 0,
        'recovered_via_smart_glue': recovered,
        'corrected_pass': new_pass,
        'corrected_pass_rate': new_pass / total if total else 0,
        'remaining_failures': failures_remaining,
        'truncated_at_max_tokens': truncated,
        'other_empty_response': other_empty,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: rescore_humaneval.py PROBLEMS_FILE INPUT_JSONL [INPUT_JSONL ...]")
        sys.exit(2)
    problems_file = Path(sys.argv[1])
    if not problems_file.is_file():
        print(f"ERROR: problems file not found: {problems_file}")
        sys.exit(2)

    problems = {}
    with open(problems_file) as f:
        for line in f:
            r = json.loads(line)
            problems[r['task_id']] = r

    print(f"Loaded {len(problems)} HumanEval problems.\n")
    print(f"{'Run':<55} {'Reported':>10} {'Recovered':>10} {'Corrected':>10} {'Δ':>6}")
    print("-" * 95)

    for jsonl_path in sys.argv[2:]:
        p = Path(jsonl_path)
        if not p.is_file():
            print(f"{jsonl_path:<55} (not found)")
            continue
        s = rescore(p, problems)
        delta = (s['corrected_pass_rate'] - s['orig_pass_rate']) * 100
        label = p.parent.name + '/' + p.name
        if len(label) > 53:
            label = '...' + label[-50:]
        print(f"{label:<55} {s['orig_pass_rate']*100:>9.1f}% {s['recovered_via_smart_glue']:>10} {s['corrected_pass_rate']*100:>9.1f}% {delta:>+5.1f}")

        sidecar = p.with_name(p.stem + '_rescored.json')
        with open(sidecar, 'w') as f:
            json.dump(s, f, indent=2)


if __name__ == '__main__':
    main()
