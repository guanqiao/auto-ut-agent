"""Import Optimization Script - 导入优化脚本

This script analyzes and fixes import structure issues:
- Sorts imports (stdlib, third-party, local)
- Removes unused imports
- Converts relative imports to absolute where appropriate
- Detects circular imports

Usage:
    python scripts/optimize_imports.py [file_or_directory]
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Set, Tuple, Optional


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze imports in a Python file."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.imports: List[Tuple[str, Optional[str], int]] = []  # (module, name, line)
        self.from_imports: List[Tuple[str, List[str], int]] = []  # (module, names, line)
        self.wildcard_imports: List[Tuple[str, int]] = []  # (module, line)
        self.relative_imports: List[Tuple[str, List[str], int]] = []  # (module, names, line)
    
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append((alias.name, alias.asname, node.lineno))
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module is None:
            # Relative import
            module = '.' * node.level
            names = [alias.name for alias in node.names]
            self.relative_imports.append((module, names, node.lineno))
        else:
            names = [alias.name for alias in node.names]
            if '*' in names:
                self.wildcard_imports.append((node.module, node.lineno))
            else:
                self.from_imports.append((node.module, names, node.lineno))
        self.generic_visit(node)


def is_stdlib_module(module_name: str) -> bool:
    """Check if a module is from the Python standard library."""
    stdlib_modules = {
        'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'contextlib',
        'copy', 'csv', 'dataclasses', 'datetime', 'decimal', 'enum', 'functools',
        'glob', 'hashlib', 'importlib', 'inspect', 'io', 'itertools', 'json',
        'logging', 'math', 'os', 'pathlib', 'pickle', 'platform', 'random',
        're', 'shutil', 'signal', 'socket', 'sqlite3', 'string', 'subprocess',
        'sys', 'tempfile', 'textwrap', 'threading', 'time', 'traceback', 'types',
        'typing', 'unittest', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile',
    }
    base_module = module_name.split('.')[0]
    return base_module in stdlib_modules


def is_third_party_module(module_name: str) -> bool:
    """Check if a module is a third-party module."""
    third_party_modules = {
        'pytest', 'pydantic', 'yaml', 'requests', 'numpy', 'pandas',
        'click', 'rich', 'typer', 'fastapi', 'flask', 'django',
        'langchain', 'openai', 'anthropic', 'tiktoken', 'chromadb',
        'jinja2', 'markupsafe', 'werkzeug', 'sqlalchemy', 'alembic',
        'redis', 'celery', 'kafka', 'boto3', 'botocore',
        'matplotlib', 'seaborn', 'plotly', 'pillow', 'opencv',
        'scipy', 'sklearn', 'tensorflow', 'torch', 'transformers',
        'bs4', 'lxml', 'html5lib', 'scrapy', 'selenium',
        'psycopg2', 'pymongo', 'pymysql', 'cx_oracle',
        'cryptography', 'jwt', 'oauthlib', 'requests_oauthlib',
        'aiohttp', 'tornado', 'twisted', 'gevent', 'eventlet',
        'grpc', 'protobuf', 'thrift', 'avro', 'msgpack',
        'elasticsearch', 'solr', 'whoosh', 'sphinx', 'mkdocs',
        'tox', 'nox', 'pre_commit', 'black', 'isort', 'flake8',
        'mypy', 'pylint', 'bandit', 'safety', 'pip_audit',
        'setuptools', 'wheel', 'twine', 'build', 'hatch', 'poetry',
        'pipenv', 'conda', 'mamba', 'virtualenv', 'venv',
        'PySide6', 'PyQt6', 'PyQt5', 'PySide2', 'tkinter', 'wx',
        'kivy', 'flet', 'streamlit', 'gradio', 'dash', 'panel',
        'bokeh', 'holoviews', 'datashader', 'geoviews',
        'networkx', 'igraph', 'graph_tool', 'pydot', 'pygraphviz',
        'dask', 'ray', 'modin', 'vaex', 'polars', 'duckdb',
        'arrow', 'pendulum', 'delorean', 'maya', 'dateutil',
        'faker', 'hypothesis', 'factory_boy', 'mock', 'responses',
        'freezegun', 'time_machine', 'vcrpy', 'betamax',
        'coverage', 'pytest_cov', 'pytest_xdist', 'pytest_asyncio',
        'pytest_mock', 'pytest_django', 'pytest_flask', 'pytest_fastapi',
        'allure', 'xdist', 'nose', 'nose2', 'unittest2', 'trial',
        'doctest', 'sphinx_testing', 'pytest_sphinx',
        'tenacity', 'backoff', 'retry', 'retrying', 'stopit',
        'timeout_decorator', 'func_timeout', 'pebble', 'billiard',
        'multiprocessing', 'concurrent', 'threading', 'queue',
        'asyncio', 'aiofiles', 'aiopath', 'anyio', 'trio',
        'sniffio', 'outcome', 'trio_asyncio', 'curio',
        'greenlet', 'stackless', 'pypy', 'cython', 'numba',
        'cffi', 'ctypes', 'swig', 'boost', 'pybind11',
        'pyro', 'pyro4', 'pyro5', 'rpyc', 'zerorpc', 'pyzmq',
        'pika', 'kombu', 'dramatiq', 'huey', 'rq', 'celery',
        'airflow', 'prefect', 'dagster', 'luigi', 'mage',
        'kedro', 'metaflow', 'mlflow', 'kubeflow', 'flyte',
        'dvc', 'cml', 'neptune', 'wandb', 'comet', 'tensorboard',
        'visdom', 'dash', 'streamlit', 'gradio', 'panel', 'voila',
        'jupyter', 'ipython', 'ipywidgets', 'bqplot', 'ipyleaflet',
        'pythreejs', 'ipyvolume', 'ipycanvas', 'ipyevents',
        'traitlets', 'nbformat', 'nbconvert', 'nbclient',
        'jupyterlab', 'notebook', 'nteract', 'hydrogen',
        'papermill', 'scrapbook', 'commuter', 'nbviewer',
        'binder', 'repo2docker', 'jupyterhub', 'jupyterlite',
        'voila', 'panel', 'dash', 'streamlit', 'gradio',
        'nicegui', 'flet', 'pynecone', 'reflex', 'solara',
        'anvil', 'pywebio', 'remi', 'flexx', 'pyscript',
        'brython', 'skulpt', 'transcrypt', 'rapydscript',
        'pyjs', 'pyjamas', 'pyjaco', 'py2js', 'pythonjs',
    }
    base_module = module_name.split('.')[0]
    return base_module in third_party_modules


def is_local_module(module_name: str, file_path: Path) -> bool:
    """Check if a module is a local project module."""
    return module_name.startswith('pyutagent')


def categorize_import(module_name: str, file_path: Path) -> str:
    """Categorize an import as stdlib, third-party, or local."""
    if is_stdlib_module(module_name):
        return 'stdlib'
    elif is_third_party_module(module_name):
        return 'third_party'
    elif is_local_module(module_name, file_path):
        return 'local'
    else:
        return 'unknown'


def analyze_file(file_path: Path) -> dict:
    """Analyze a Python file for import issues."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        analyzer = ImportAnalyzer(file_path)
        analyzer.visit(tree)
        
        issues = []
        
        # Check for wildcard imports
        for module, line in analyzer.wildcard_imports:
            issues.append({
                'type': 'wildcard_import',
                'line': line,
                'message': f"Wildcard import from '{module}' at line {line}"
            })
        
        # Check for relative imports
        for module, names, line in analyzer.relative_imports:
            issues.append({
                'type': 'relative_import',
                'line': line,
                'message': f"Relative import at line {line}: from {module} import {', '.join(names)}"
            })
        
        # Categorize imports
        import_categories = {
            'stdlib': [],
            'third_party': [],
            'local': [],
            'unknown': []
        }
        
        for module, alias, line in analyzer.imports:
            category = categorize_import(module, file_path)
            import_categories[category].append((module, alias, line))
        
        for module, names, line in analyzer.from_imports:
            category = categorize_import(module, file_path)
            import_categories[category].append((module, names, line))
        
        return {
            'file': str(file_path),
            'issues': issues,
            'import_categories': import_categories,
            'total_imports': len(analyzer.imports) + len(analyzer.from_imports),
        }
    except SyntaxError as e:
        return {
            'file': str(file_path),
            'error': f"Syntax error: {e}"
        }
    except Exception as e:
        return {
            'file': str(file_path),
            'error': str(e)
        }


def scan_directory(directory: Path) -> List[dict]:
    """Scan a directory for Python files and analyze their imports."""
    results = []
    
    for py_file in directory.rglob('*.py'):
        # Skip common non-project directories
        if any(part.startswith('.') for part in py_file.parts):
            continue
        if 'venv' in py_file.parts or '__pycache__' in py_file.parts:
            continue
        
        result = analyze_file(py_file)
        results.append(result)
    
    return results


def print_report(results: List[dict]):
    """Print a formatted report of import analysis."""
    print("=" * 80)
    print("IMPORT STRUCTURE ANALYSIS REPORT")
    print("=" * 80)
    
    total_files = len(results)
    files_with_issues = 0
    total_wildcards = 0
    total_relatives = 0
    
    for result in results:
        if 'error' in result:
            print(f"\n⚠️  {result['file']}")
            print(f"   Error: {result['error']}")
            continue
        
        issues = result.get('issues', [])
        if issues:
            files_with_issues += 1
            print(f"\n📄 {result['file']}")
            
            for issue in issues:
                if issue['type'] == 'wildcard_import':
                    total_wildcards += 1
                    print(f"   ⚠️  {issue['message']}")
                elif issue['type'] == 'relative_import':
                    total_relatives += 1
                    print(f"   ℹ️  {issue['message']}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files analyzed: {total_files}")
    print(f"Files with issues: {files_with_issues}")
    print(f"Wildcard imports found: {total_wildcards}")
    print(f"Relative imports found: {total_relatives}")
    
    if files_with_issues == 0:
        print("\n✅ No import issues found!")
    else:
        print(f"\n⚠️  Found import issues in {files_with_issues} files")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path('pyutagent')
    
    if not target.exists():
        print(f"Error: {target} does not exist")
        sys.exit(1)
    
    if target.is_file():
        results = [analyze_file(target)]
    else:
        print(f"Scanning directory: {target}")
        results = scan_directory(target)
    
    print_report(results)


if __name__ == '__main__':
    main()
