"""Load only the official argument dataclasses, excluding GPU operator imports and main()."""
import ast
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import sys
import types
import typing

def load_argument_classes():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    from lingbotvla.utils.arguments import DataArguments, EvalArguments, ModelArguments, TrainingArguments
    path = Path(__file__).resolve().parents[2] / 'lingbot-vla-v2/tasks/vla/train_lingbotvla.py'
    source = path.read_text()
    names = {'MyTrainingArguments', 'MyDataArguments', 'Arguments'}
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name in names]
    assert {node.name for node in nodes} == names
    mod = types.ModuleType('tasks.vla.train_lingbotvla')
    mod.__file__ = str(path)
    mod.__dict__.update(vars(typing))
    mod.__name__ = 'tasks.vla.train_lingbotvla'
    mod.__dict__.update(dataclass=dataclass, field=field, DataArguments=DataArguments,
        EvalArguments=EvalArguments, ModelArguments=ModelArguments, TrainingArguments=TrainingArguments)
    sys.modules[mod.__name__] = mod
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec')
    exec(code, mod.__dict__)
    mod.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return mod

if __name__ == '__main__':
    import runpy
    root = Path(__file__).resolve().parents[2]
    load_argument_classes()
    assert len(sys.argv) == 2 and Path(sys.argv[1]).name.startswith('norm-')
    # Only the official normalization utility is allowed here.
    runpy.run_path(str(root / 'lingbot-vla-v2/scripts/compute_norm_stats.py'), run_name='__main__')
