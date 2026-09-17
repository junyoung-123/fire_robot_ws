#!/usr/bin/env python3
"""Keep source and input fingerprints beside a new validation, without assets."""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def snapshot(workspace, output):
    output.mkdir(parents=True, exist_ok=False)
    files = sorted(path for root in ('src', 'scripts', 'tools')
                   for path in (workspace / root).rglob('*')
                   if path.is_file() and '__pycache__' not in path.parts)
    inputs = {}
    source_suffixes = {'.py', '.yaml', '.yml', '.xml', '.world', '.xacro',
                       '.urdf', '.sh', '.cfg', '.txt', '.md', '.srv', '.msg',
                       '.cpp', '.cc', '.hh', '.h', '.hpp', '.cmake'}
    with tarfile.open(output / 'source_without_weights_meshes.tar.gz', 'w:gz') as archive:
        for path in files:
            if path.suffix.lower() in source_suffixes | {'.pt', '.pgm'}:
                name = path.relative_to(workspace).as_posix()
                inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                if path.suffix.lower() in source_suffixes:
                    archive.add(path, arcname=name)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=workspace, text=True)
    metadata = dict(commit=git('rev-parse', 'HEAD').strip(),
                    branch=git('branch', '--show-current').strip(),
                    status=git('status', '--short'), inputs_sha256=inputs,
                    scope='Source snapshot; weights, meshes and images not archived')
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (output / 'source.patch').write_text(git('diff', '--binary', 'HEAD'), encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = snapshot(args.workspace.resolve(), args.output)
    print(f"Archived {len(result['inputs_sha256'])} input hashes in {args.output}")
