"""Save official dependency checks and prove the two existing environments unchanged."""
import difflib
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from packaging.requirements import Requirement

assert sys.prefix == '/root/miniconda3/envs/RoboTwin'
out = Path(__file__).parent
repo = Path('/home/lmy/Robbyant/RoboTwin')
env = dict(os.environ, PIP_CACHE_DIR=str(out / 'pip-cache'))
checks = []
for line in (repo / 'script/requirements.txt').read_text().splitlines() + ['numpy==1.26.4', 'warp-lang==1.12.0', 'setuptools==69.5.1', 'nvidia-curobo==0.7.8']:
    line = line.split('#')[0].strip()
    if not line:
        continue
    requirement = Requirement(line)
    installed = metadata.version(requirement.name)
    checks.append({'requirement': line, 'installed': installed, 'passed': installed in requirement.specifier})
pip_check = subprocess.run([sys.executable, '-m', 'pip', 'check'], env=env, text=True, capture_output=True)
(out / 'pip-check.txt').write_text(pip_check.stdout + pip_check.stderr)
for filename, options in [('packages.json', ['list', '--format=json']), ('pip-freeze.txt', ['freeze'])]:
    (out / filename).write_text(subprocess.check_output([sys.executable, '-m', 'pip'] + options, env=env, text=True))
(out / 'conda-explicit.txt').write_text(subprocess.check_output(['/root/miniconda3/bin/conda', 'list', '-n', 'RoboTwin', '--explicit'], text=True))
base_after = subprocess.check_output(['/root/miniconda3/bin/conda', 'list', '-n', 'base', '--explicit'], text=True)
(out / 'base-after.txt').write_text(base_after)
model_after = subprocess.check_output(['/root/miniconda3/envs/lingbotvla/bin/python', '-m', 'pip', 'list', '--format=json'], env=env, text=True)
(out / 'lingbotvla-after.json').write_text(model_after)
patches = {}
for package, filename in [('sapien', 'sapien/wrapper/urdf_loader.py'), ('mplib', 'mplib/planner.py')]:
    source = metadata.distribution(package).locate_file(filename)
    before = (out / (package + '-before-official-patch.py')).read_text()
    after = source.read_text()
    difference = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='before', tofile=str(source)))
    (out / (package + '-official-patch.diff')).write_text(difference)
    patches[package] = {'source': str(source), 'changed': before != after, 'diff': str(out / (package + '-official-patch.diff'))}
verification = json.loads((out / 'verification-native.json').read_text())
report = {
    'environment': 'RoboTwin', 'prefix': sys.prefix, 'python': verification['python'],
    'repository': str(repo),
    'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    'official_requirements': checks,
    'official_requirements_passed': all(c['passed'] for c in checks),
    'pip_check_passed': pip_check.returncode == 0,
    'base_inventory_unchanged': base_after == (out / 'base-before.txt').read_text(),
    'base_history_unchanged': hashlib.sha256(Path('/root/miniconda3/conda-meta/history').read_bytes()).hexdigest() == (out / 'base-history-before.sha256').read_text().strip(),
    'lingbotvla_packages_unchanged': json.loads(model_after) == json.loads((out / 'lingbotvla-before.json').read_text()),
    'robotwin_repository_clean': not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True),
    'lingbot_repository_clean': not subprocess.check_output(['git', 'status', '--porcelain'], cwd='/home/lmy/Robbyant/lingbot-vla-v2', text=True),
    'official_dependency_source_patches': patches,
    'runtime_validation_passed': verification['passed'],
    'runtime_checks_passed': sum(c['passed'] for c in verification['checks'].values()),
    'runtime_checks_total': len(verification['checks']),
    'verification_runs': ['verification-native.log', 'verification-kinematics-recheck.log', 'verification-project-import-final.log'],
    'verification_harness_repairs': ['Passed tensor_args by keyword to the CuRobo kinematics API.', 'Added the RoboTwin repository root to sys.path.'],
    'vulkan_probe': json.loads((out / 'vulkan-probe.json').read_text()),
    'versions': verification['versions'],
    'cuda_visible_devices': '0', 'models_downloaded': False, 'robotwin_dataset_downloaded': False,
    'robotwin_asset_bundles_downloaded': False, 'closed_loop_policy_evaluation_run': False,
    'official_asset_metadata_download': json.loads((out / 'asset-metadata-download.json').read_text()),
    'curobo_build_dependency_repair': 'Rebuilt the identical v0.7.8 source after its declared setuptools_scm build dependency was installed; corrected initial 0.0.0 distribution version to 0.7.8.',
    'graphics_library_source': 'NVIDIA official libnvidia-gl-580_580.178.04-1ubuntu1_amd64.deb, extracted only into project directory',
    'compilation_cuda_toolkit': '12.8',
}
for name in ['pytorch3d', 'nvidia-curobo']:
    distribution = metadata.distribution(name)
    direct_url = distribution.read_text('direct_url.json')
    report[name + '_source'] = json.loads(direct_url) if direct_url else None
report['curobo_checkout_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo / 'envs/curobo', text=True).strip()
(out / 'installation-summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps({key: report[key] for key in ['official_requirements_passed', 'pip_check_passed', 'runtime_validation_passed',
                    'runtime_checks_passed', 'base_inventory_unchanged', 'base_history_unchanged', 'lingbotvla_packages_unchanged',
                    'robotwin_repository_clean', 'lingbot_repository_clean']}, indent=2))
assert report['official_requirements_passed'] and report['runtime_validation_passed']
assert report['base_inventory_unchanged'] and report['base_history_unchanged'] and report['lingbotvla_packages_unchanged']
