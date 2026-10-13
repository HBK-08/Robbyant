"""Audit explicit official pins and save inventories without changing packages."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement

assert sys.prefix == '/root/miniconda3/envs/lingbotvla', sys.prefix
artifact_dir = Path(__file__).parent
repo_dir = Path('/home/lmy/Robbyant/lingbot-vla-v2')
pins = []
for line in (artifact_dir / 'official-final-pins.txt').read_text().splitlines():
    if not line or line.startswith('#'):
        continue
    requirement = Requirement(line)
    actual = metadata.version(requirement.name)
    pins.append({'requirement': line, 'installed': actual,
                 'passed': actual in requirement.specifier})

pip_check = subprocess.run([sys.executable, '-m', 'pip', 'check'],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(artifact_dir / 'pip-check.txt').write_text(pip_check.stdout)
(artifact_dir / 'packages.txt').write_text(subprocess.check_output(
    [sys.executable, '-m', 'pip', 'list'], text=True))
(artifact_dir / 'pip-freeze.txt').write_text(subprocess.check_output(
    [sys.executable, '-m', 'pip', 'freeze'], text=True))
(artifact_dir / 'packages-after-official-alignment.json').write_text(subprocess.check_output(
    [sys.executable, '-m', 'pip', 'list', '--format=json'], text=True))
base_after = subprocess.check_output(
    ['/root/miniconda3/bin/conda', 'list', '-n', 'base', '--explicit'], text=True)
(artifact_dir / 'base-after-official-alignment.txt').write_text(base_after)
base_unchanged = base_after == (artifact_dir / 'base-before-official-alignment.txt').read_text()
base_history_hash = hashlib.sha256(Path('/root/miniconda3/conda-meta/history').read_bytes()).hexdigest()
expected_history_hash = (artifact_dir / 'base-history-before.sha256').read_text().split()[0]
git_status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo_dir, text=True)
verification = json.loads((artifact_dir / 'verification.json').read_text())
report = {
    'environment': 'lingbotvla', 'prefix': sys.prefix,
    'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_dir, text=True).strip(),
    'official_explicit_pin_checks': pins,
    'official_explicit_pins_passed': all(p['passed'] for p in pins),
    'hub_precedence': 'tools/create_train_env.sh:175 overrides requirements.txt:0.34.3 with 0.34.0',
    'installation_adjustment': 'MoGe installed with dependency resolution enabled, constrained by explicit repository pins to prevent unpinned transitive upgrades.',
    'unchanged_official_script_completed': False,
    'pip_check_passed': pip_check.returncode == 0,
    'pip_check_messages': len(pip_check.stdout.splitlines()),
    'base_package_inventory_unchanged': base_unchanged,
    'base_history_unchanged': base_history_hash == expected_history_hash,
    'repository_clean': not git_status,
    'repository_status': git_status,
    'runtime_verification_versions': verification['versions'],
    'runtime_validation_passed': verification['passed'],
    'runtime_checks_passed': sum(c['passed'] for c in verification['checks'].values()),
}
(artifact_dir / 'official-alignment-report.json').write_text(json.dumps(report, indent=2))
summary = json.loads((artifact_dir / 'installation-summary.json').read_text())
summary.update({key: report[key] for key in (
    'pip_check_passed', 'pip_check_messages', 'base_package_inventory_unchanged',
    'base_history_unchanged', 'repository_clean', 'runtime_validation_passed', 'runtime_checks_passed')})
summary['versions'] = {name: metadata.version(name) for name in summary['versions']}
summary['official_alignment_report'] = str(artifact_dir / 'official-alignment-report.json')
summary['current_installation_policy'] = report['installation_adjustment']
summary['hub_precedence'] = report['hub_precedence']
summary['repairs_are_historical'] = True
(artifact_dir / 'installation-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({key: report[key] for key in (
    'official_explicit_pins_passed', 'pip_check_messages', 'base_package_inventory_unchanged',
    'base_history_unchanged', 'repository_clean')}, indent=2))
assert report['official_explicit_pins_passed'], [p for p in pins if not p['passed']]
assert base_unchanged and report['base_history_unchanged']
