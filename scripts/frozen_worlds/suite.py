"""One frozen-code attempt per world; retain every result without overwriting."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from build_world import build
from audit import check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--baseline-label', default='R243')
    parser.add_argument('--worlds', type=int, nargs='+', choices=range(1,6), default=[2,3,4,5])
    parser.add_argument('--stop-on-failure', action='store_true')
    parser.add_argument('--timeout', type=float, default=21600.)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    harness = Path(__file__).resolve().parent
    fixtures = root / 'fixtures'
    for number in args.worlds:
        fixture = fixtures / f'world{number}_full_contact.world'
        if not fixture.exists():
            build(number, fixture)
    if args.prepare_only:
        return
    suite = dict(policy=args.baseline_label+' frozen per sequence; no edits during a run',
        requested_order=args.worlds,
        baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        harness_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in harness.glob('*.py')},
        active_world=None, completed=[])
    status = root / 'suite_status.json'
    for number in args.worlds:
        output = root / f'world{number}'
        if output.exists():
            raise FileExistsError('No automatic retry or overwritten evidence: ' + str(output))
        suite['active_world'] = number
        status.write_text(json.dumps(suite, indent=2))
        command = [sys.executable, str(harness / 'run.py'), '--world', str(number),
            '--fixture', str(fixtures / f'world{number}_full_contact.world'),
            '--baseline', str(args.baseline.resolve()), '--output', str(output),
            '--baseline-label', args.baseline_label,
            '--timeout', str(args.timeout), '--full-reference-validation',
            '--contact-motion-policy', 'r34_reference', '--contact-reference-frame', 'contact_scan', '--dds-udp-only']
        with (root / f'world{number}_runner.log').open('w') as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            report = check(output)
            result = json.loads((output / 'result.json').read_text())
            summary = dict(world=number, status=result['status'], pass_all=report['pass_all'],
                physical_jobs=result['successful_physical_jobs'],
                expected_blue=len(report['evaluation_expected_blue_models']),
                failing_checks=[key for key, passed in report['checks'].items() if not passed])
        except Exception as error:
            summary = dict(world=number, status='INCOMPLETE_EVIDENCE', pass_all=False, error=repr(error),
                           runner_exit=process.returncode)
        suite['completed'].append(summary)
        status.write_text(json.dumps(suite, indent=2))
        print(json.dumps(summary), flush=True)
        if not summary['pass_all'] and args.stop_on_failure:
            suite['stopped_on_failure'] = number
            break
        if not all(hashlib.sha256((harness/name).read_bytes()).hexdigest() == digest
                   for name, digest in suite['harness_sha256'].items()):
            raise RuntimeError('Test harness changed during frozen sequence')
    suite['active_world'] = None
    suite['closed'] = True
    status.write_text(json.dumps(suite, indent=2))


if __name__ == '__main__':
    main()
