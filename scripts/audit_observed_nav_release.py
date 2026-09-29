"""Compare release files against immutable r29 runtime manifests."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def audit(repo, evidence):
    manifests = [json.loads((evidence / f'world{w}_r29/manifest.json').read_text()) for w in range(1, 6)]
    result = {'baseline_commit': '27359f7', 'revision': 'r29', 'worlds': 5,
              'same_runtime_sources': True, 'same_runtime_harness': True,
              'source_files_checked': 0, 'harness_files_checked': 0,
              'line_ending_only_differences': [], 'mismatches': [], 'syntax_errors': []}
    for field, prefix, count in [('source_sha256', '', 'source_files_checked'),
                                 ('harness_sha256', 'scripts/', 'harness_files_checked')]:
        reference = manifests[0][field]
        result['same_runtime_sources' if not prefix else 'same_runtime_harness'] = all(m[field] == reference for m in manifests)
        for relative, expected in reference.items():
            filename = prefix + relative
            file = repo / filename
            data = file.read_bytes() if file.is_file() else b''
            digest = hashlib.sha256(data).hexdigest()
            normalized = hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()
            if digest != expected:
                if normalized == expected:
                    result['line_ending_only_differences'].append(filename)
                else:
                    result['mismatches'].append({'file': filename, 'expected': expected, 'actual': digest})
            if file.suffix == '.py':
                try:
                    ast.parse(data.decode('utf-8-sig'), filename=filename)
                except (SyntaxError, UnicodeError) as exc:
                    result['syntax_errors'].append({'file': filename, 'error': str(exc)})
            result[count] += 1
    result['pass'] = (result['same_runtime_sources'] and result['same_runtime_harness']
                      and not result['mismatches'] and not result['syntax_errors'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.repo, args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(0 if result['pass'] else 1)
