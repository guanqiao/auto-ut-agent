methods = [
    {"name": "add", "parameters": [("int", "a"), ("int", "b")]},
    {"name": "subtract", "parameters": [("int", "x")]},
]

method_sig = ";".join(sorted([
    f"{m.get('name', '')}({','.join(f'{p[0]} {p[1]}' for p in m.get('parameters', []))})"
    for m in methods
]))
print(f"method_sig: {method_sig}")
