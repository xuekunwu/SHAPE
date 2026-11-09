"""List maintained implementations without importing scientific modules."""
import argparse
import ast
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GROUPS=("dataset","runtime","workflows","prediction","evaluation")

def scripts():
    paths=[ROOT/'run.py'] + [p for group in GROUPS for p in (ROOT/group).rglob('*.py') if p.name!='__init__.py']
    rows=[]
    for path in sorted(paths):
        text=path.read_text(encoding='utf-8');tree=ast.parse(text)
        cli=any(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
                and '__name__' in ast.unparse(n.test) for n in tree.body)
        rows.append(dict(path=path.relative_to(ROOT).as_posix(),category=path.relative_to(ROOT).parts[0] if path.parent!=ROOT else 'entrypoints',
            kind='CLI' if cli else 'library',description=(ast.get_docstring(tree) or '').splitlines()[0],
            help_supported=cli and path.stem not in ('import_archive','six_arm_report')))
    return rows



def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category",choices=(*GROUPS,"entrypoints"))
    parser.add_argument("--json",action="store_true")
    args=parser.parse_args(argv)
    rows=[r for r in scripts() if not args.category or r["category"]==args.category]
    if args.json:print(json.dumps(rows,indent=2))
    else:
        for r in rows:print(f"{r['category']:12} {r['kind']:7} {r['path']:48} {r['description']}")
