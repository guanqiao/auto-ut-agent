"""
Smart Refactoring Engine for Test Code Improvement
Provides automated refactoring suggestions and transformations
"""

import re
import ast
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Set
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class RefactoringType(Enum):
    EXTRACT_METHOD = "extract_method"
    EXTRACT_CONSTANT = "extract_constant"
    RENAME_VARIABLE = "rename_variable"
    SIMPLIFY_CONDITIONAL = "simplify_conditional"
    REMOVE_DUPLICATION = "remove_duplication"
    IMPROVE_NAMING = "improve_naming"
    ADD_MISSING_ASSERTIONS = "add_missing_assertions"
    ORGANIZE_IMPORTS = "organize_imports"
    REMOVE_DEAD_CODE = "remove_dead_code"
    EXTRACT_TEST_DATA = "extract_test_data"
    PARAMETERIZE_TEST = "parameterize_test"
    SPLIT_TEST_METHOD = "split_test_method"


@dataclass
class RefactoringSuggestion:
    refactoring_type: RefactoringType
    description: str
    location: Tuple[int, int]
    suggested_code: Optional[str] = None
    original_code: Optional[str] = None
    priority: int = 1
    confidence: float = 0.8
    impact: str = "medium"
    rationale: str = ""


@dataclass
class RefactoringResult:
    success: bool
    refactoring_type: RefactoringType
    original_code: str
    refactored_code: str
    changes_made: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class JavaTestAnalyzer:
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.lines = source_code.split('\n')
        self.methods: Dict[str, Dict[str, Any]] = {}
        self.imports: List[str] = []
        self.fields: Dict[str, str] = {}
        self.class_name: str = ""
        self._parse()
    
    def _parse(self):
        self._extract_imports()
        self._extract_class_name()
        self._extract_methods()
        self._extract_fields()
    
    def _extract_imports(self):
        import_pattern = r'^import\s+[\w.]+;'
        for line in self.lines:
            match = re.match(import_pattern, line.strip())
            if match:
                self.imports.append(match.group())
    
    def _extract_class_name(self):
        class_pattern = r'public\s+class\s+(\w+)'
        for line in self.lines:
            match = re.search(class_pattern, line)
            if match:
                self.class_name = match.group(1)
                break
    
    def _extract_methods(self):
        method_pattern = r'@(Test|Before|After|BeforeEach|AfterEach|BeforeClass|AfterClass)\s*(?:\n\s*)?(?:public|private|protected)?\s*(?:static\s+)?(?:\w+(?:<[\w\s,<>]+>)?)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[\w,\s]+)?\s*\{'
        
        content = self.source_code
        for match in re.finditer(method_pattern, content):
            method_name = match.group(2)
            params = match.group(3)
            start_pos = match.start()
            
            brace_count = 0
            method_start = match.end()
            method_end = method_start
            
            for i in range(method_start, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        method_end = i + 1
                        break
            
            method_body = content[method_start:method_end]
            
            self.methods[method_name] = {
                'annotation': match.group(1),
                'params': params,
                'body': method_body,
                'start': start_pos,
                'end': method_end,
                'start_line': content[:start_pos].count('\n') + 1,
                'end_line': content[:method_end].count('\n') + 1
            }
    
    def _extract_fields(self):
        field_pattern = r'(?:private|protected|public)?\s*(?:static\s+)?(?:final\s+)?(\w+(?:<[\w\s,<>]+>)?)\s+(\w+)\s*(?:=\s*[^;]+)?;'
        
        for match in re.finditer(field_pattern, self.source_code):
            field_type = match.group(1)
            field_name = match.group(2)
            
            in_method = False
            for method_info in self.methods.values():
                if method_info['start'] < match.start() < method_info['end']:
                    in_method = True
                    break
            
            if not in_method:
                self.fields[field_name] = field_type


class CodeDuplicationDetector:
    
    def __init__(self, min_lines: int = 3, min_tokens: int = 20):
        self.min_lines = min_lines
        self.min_tokens = min_tokens
    
    def detect_duplication(self, methods: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        duplications = []
        method_names = list(methods.keys())
        
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                method1 = methods[method_names[i]]
                method2 = methods[method_names[j]]
                
                similarity = self._calculate_similarity(
                    method1['body'], method2['body']
                )
                
                if similarity > 0.7:
                    duplications.append({
                        'method1': method_names[i],
                        'method2': method_names[j],
                        'similarity': similarity,
                        'suggestion': self._generate_duplication_suggestion(
                            method_names[i], method_names[j], similarity
                        )
                    })
        
        return duplications
    
    def _calculate_similarity(self, code1: str, code2: str) -> float:
        tokens1 = self._tokenize(code1)
        tokens2 = self._tokenize(code2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _tokenize(self, code: str) -> List[str]:
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'"[^"]*"', '""', code)
        code = re.sub(r'\d+', 'NUM', code)
        
        tokens = re.findall(r'\b\w+\b|[{}()\[\];,.]', code)
        return [t for t in tokens if len(t) > 1 or t in '{}()[];']
    
    def _generate_duplication_suggestion(self, method1: str, method2: str, similarity: float) -> str:
        return f"Methods '{method1}' and '{method2}' have {similarity:.0%} similar code. Consider extracting common logic into a helper method."


class NamingAnalyzer:
    
    BAD_PATTERNS = {
        'test': [r'^test\d+$', r'^testMethod\d*$', r'^test[A-Z]{1,3}$'],
        'variable': [r'^[a-z]$', r'^temp\d*$', r'^var\d+$', r'^x\d+$', r'^_'],
        'method': [r'^method\d+$', r'^helper\d*$', r'^do[A-Z]'],
        'constant': [r'^[a-z]+$', r'^[A-Z]{1,2}$']
    }
    
    GOOD_PATTERNS = {
        'test': [r'^should[A-Z]', r'^when[A-Z]', r'^given[A-Z]', r'^test[A-Z][a-zA-Z]+'],
        'variable': [r'^[a-z][a-zA-Z0-9]*$'],
        'constant': [r'^[A-Z][A-Z0-9_]*$']
    }
    
    def analyze_naming(self, analyzer: JavaTestAnalyzer) -> List[RefactoringSuggestion]:
        suggestions = []
        
        for method_name, method_info in analyzer.methods.items():
            if method_info['annotation'] == 'Test':
                if self._is_bad_name(method_name, 'test'):
                    suggestions.append(RefactoringSuggestion(
                        refactoring_type=RefactoringType.IMPROVE_NAMING,
                        description=f"Test method '{method_name}' has unclear naming",
                        location=(method_info['start_line'], method_info['end_line']),
                        original_code=method_name,
                        suggested_code=self._suggest_better_name(method_name, method_info['body']),
                        priority=2,
                        confidence=0.9,
                        impact="medium",
                        rationale="Clear test names improve readability and documentation"
                    ))
        
        for field_name in analyzer.fields:
            if self._is_bad_name(field_name, 'variable'):
                suggestions.append(RefactoringSuggestion(
                    refactoring_type=RefactoringType.IMPROVE_NAMING,
                    description=f"Field '{field_name}' has unclear naming",
                    location=(1, 1),
                    original_code=field_name,
                    suggested_code=self._suggest_better_variable_name(field_name),
                    priority=3,
                    confidence=0.7,
                    impact="low"
                ))
        
        return suggestions
    
    def _is_bad_name(self, name: str, name_type: str) -> bool:
        patterns = self.BAD_PATTERNS.get(name_type, [])
        return any(re.match(p, name) for p in patterns)
    
    def _suggest_better_name(self, current_name: str, method_body: str) -> str:
        assertions = re.findall(r'assert(?:Equals|True|False|NotNull|Null)\s*\(\s*(?:[^,]+,)?\s*([^,)]+)', method_body)
        conditions = re.findall(r'(?:when|given|if)\s*\(\s*(\w+)', method_body, re.IGNORECASE)
        
        if conditions:
            return f"should{self._capitalize(conditions[0])}When{self._capitalize(assertions[0]) if assertions else 'Expected'}"
        
        if assertions:
            return f"shouldReturn{self._capitalize(assertions[0])}"
        
        return f"should{self._capitalize(current_name.replace('test', ''))}"
    
    def _suggest_better_variable_name(self, current_name: str) -> str:
        return "meaningfulName"
    
    def _capitalize(self, s: str) -> str:
        return s[0].upper() + s[1:] if s else ""


class AssertionAnalyzer:
    
    ASSERTION_PATTERNS = [
        r'assert(?:Equals|True|False|NotNull|Null|ArrayEquals|Same)\s*\(',
        r'expect(?:ed)?\s*\(',
        r'verify\s*\(',
        r'should\s*\(',
    ]
    
    def analyze_assertions(self, methods: Dict[str, Dict[str, Any]]) -> List[RefactoringSuggestion]:
        suggestions = []
        
        for method_name, method_info in methods.items():
            if method_info['annotation'] != 'Test':
                continue
            
            body = method_info['body']
            assertion_count = self._count_assertions(body)
            
            if assertion_count == 0:
                suggestions.append(RefactoringSuggestion(
                    refactoring_type=RefactoringType.ADD_MISSING_ASSERTIONS,
                    description=f"Test method '{method_name}' has no assertions",
                    location=(method_info['start_line'], method_info['end_line']),
                    priority=1,
                    confidence=0.95,
                    impact="high",
                    rationale="Tests without assertions don't verify behavior"
                ))
            elif assertion_count > 5:
                suggestions.append(RefactoringSuggestion(
                    refactoring_type=RefactoringType.SPLIT_TEST_METHOD,
                    description=f"Test method '{method_name}' has {assertion_count} assertions, consider splitting",
                    location=(method_info['start_line'], method_info['end_line']),
                    priority=3,
                    confidence=0.7,
                    impact="medium",
                    rationale="Too many assertions in one test make failures hard to diagnose"
                ))
            
            weak_assertions = self._find_weak_assertions(body)
            for weak in weak_assertions:
                suggestions.append(RefactoringSuggestion(
                    refactoring_type=RefactoringType.ADD_MISSING_ASSERTIONS,
                    description=f"Weak assertion in '{method_name}': {weak}",
                    location=(method_info['start_line'], method_info['end_line']),
                    original_code=weak,
                    suggested_code=self._suggest_stronger_assertion(weak),
                    priority=2,
                    confidence=0.8,
                    impact="medium"
                ))
        
        return suggestions
    
    def _count_assertions(self, code: str) -> int:
        count = 0
        for pattern in self.ASSERTION_PATTERNS:
            count += len(re.findall(pattern, code))
        return count
    
    def _find_weak_assertions(self, code: str) -> List[str]:
        weak = []
        
        assertTrue_matches = re.findall(r'assertTrue\s*\(\s*([^)]+)\s*\)', code)
        for match in assertTrue_matches:
            if '==' in match or '!=' in match:
                weak.append(f"assertTrue({match})")
        
        assertFalse_matches = re.findall(r'assertFalse\s*\(\s*([^)]+)\s*\)', code)
        for match in assertFalse_matches:
            if '==' in match or '!=' in match:
                weak.append(f"assertFalse({match})")
        
        return weak
    
    def _suggest_stronger_assertion(self, weak_assertion: str) -> str:
        if 'assertTrue' in weak_assertion:
            match = re.search(r'assertTrue\s*\(\s*(\w+)\s*==\s*([^)]+)\s*\)', weak_assertion)
            if match:
                return f"assertEquals({match.group(2)}, {match.group(1)})"
        
        if 'assertFalse' in weak_assertion:
            match = re.search(r'assertFalse\s*\(\s*(\w+)\s*==\s*([^)]+)\s*\)', weak_assertion)
            if match:
                return f"assertNotEquals({match.group(2)}, {match.group(1)})"
        
        return weak_assertion


class RefactoringEngine:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analyzer: Optional[JavaTestAnalyzer] = None
        self.duplication_detector = CodeDuplicationDetector()
        self.naming_analyzer = NamingAnalyzer()
        self.assertion_analyzer = AssertionAnalyzer()
        self._refactoring_history: List[RefactoringResult] = []
    
    def analyze(self, source_code: str) -> List[RefactoringSuggestion]:
        self.analyzer = JavaTestAnalyzer(source_code)
        suggestions = []
        
        suggestions.extend(self._analyze_duplication())
        suggestions.extend(self._analyze_naming())
        suggestions.extend(self._analyze_assertions())
        suggestions.extend(self._analyze_structure())
        suggestions.extend(self._analyze_test_data())
        
        suggestions.sort(key=lambda s: (s.priority, -s.confidence))
        
        return suggestions
    
    def _analyze_duplication(self) -> List[RefactoringSuggestion]:
        suggestions = []
        duplications = self.duplication_detector.detect_duplication(self.analyzer.methods)
        
        for dup in duplications:
            suggestions.append(RefactoringSuggestion(
                refactoring_type=RefactoringType.REMOVE_DUPLICATION,
                description=dup['suggestion'],
                location=(
                    self.analyzer.methods[dup['method1']]['start_line'],
                    self.analyzer.methods[dup['method2']]['end_line']
                ),
                priority=2,
                confidence=dup['similarity'],
                impact="high",
                rationale="Removing duplication improves maintainability"
            ))
        
        return suggestions
    
    def _analyze_naming(self) -> List[RefactoringSuggestion]:
        return self.naming_analyzer.analyze_naming(self.analyzer)
    
    def _analyze_assertions(self) -> List[RefactoringSuggestion]:
        return self.assertion_analyzer.analyze_assertions(self.analyzer.methods)
    
    def _analyze_structure(self) -> List[RefactoringSuggestion]:
        suggestions = []
        
        imports_to_organize = self._check_import_organization()
        if imports_to_organize:
            suggestions.append(RefactoringSuggestion(
                refactoring_type=RefactoringType.ORGANIZE_IMPORTS,
                description="Imports need organization",
                location=(1, len(self.analyzer.imports) + 1),
                suggested_code=imports_to_organize,
                priority=3,
                confidence=0.9,
                impact="low"
            ))
        
        for method_name, method_info in self.analyzer.methods.items():
            if method_info['annotation'] == 'Test':
                complexity = self._calculate_complexity(method_info['body'])
                if complexity > 10:
                    suggestions.append(RefactoringSuggestion(
                        refactoring_type=RefactoringType.EXTRACT_METHOD,
                        description=f"Method '{method_name}' has high complexity ({complexity})",
                        location=(method_info['start_line'], method_info['end_line']),
                        priority=2,
                        confidence=0.8,
                        impact="medium",
                        rationale="High complexity makes tests hard to understand and maintain"
                    ))
        
        return suggestions
    
    def _analyze_test_data(self) -> List[RefactoringSuggestion]:
        suggestions = []
        
        for method_name, method_info in self.analyzer.methods.items():
            if method_info['annotation'] != 'Test':
                continue
            
            body = method_info['body']
            test_data_patterns = self._find_test_data_patterns(body)
            
            if test_data_patterns:
                suggestions.append(RefactoringSuggestion(
                    refactoring_type=RefactoringType.EXTRACT_TEST_DATA,
                    description=f"Test data in '{method_name}' could be extracted",
                    location=(method_info['start_line'], method_info['end_line']),
                    original_code=test_data_patterns,
                    priority=3,
                    confidence=0.7,
                    impact="medium",
                    rationale="Extracted test data improves readability and reusability"
                ))
        
        return suggestions
    
    def _check_import_organization(self) -> Optional[str]:
        if not self.analyzer.imports:
            return None
        
        imports = sorted(set(self.analyzer.imports))
        
        java_imports = [i for i in imports if i.startswith('import java.')]
        javax_imports = [i for i in imports if i.startswith('import javax.')]
        third_party = [i for i in imports if not i.startswith('import java.') and not i.startswith('import javax.')]
        
        organized = []
        if java_imports:
            organized.extend(sorted(java_imports))
            organized.append('')
        if javax_imports:
            organized.extend(sorted(javax_imports))
            organized.append('')
        if third_party:
            organized.extend(sorted(third_party))
        
        return '\n'.join(organized).strip()
    
    def _calculate_complexity(self, code: str) -> int:
        complexity = 1
        
        patterns = [
            r'\bif\s*\(',
            r'\belse\s+if\s*\(',
            r'\bfor\s*\(',
            r'\bwhile\s*\(',
            r'\bswitch\s*\(',
            r'\bcase\s+',
            r'\bcatch\s*\(',
            r'\b\?\s*:',  # ternary
            r'&&',
            r'\|\|',
        ]
        
        for pattern in patterns:
            complexity += len(re.findall(pattern, code))
        
        return complexity
    
    def _find_test_data_patterns(self, code: str) -> List[str]:
        patterns = []
        
        list_initializations = re.findall(r'(?:List|ArrayList|Arrays)\s*\.\s*asList\s*\([^)]{50,}\)', code)
        patterns.extend(list_initializations)
        
        string_arrays = re.findall(r'new\s+String\s*\[\]\s*\{[^}]{30,}\}', code)
        patterns.extend(string_arrays)
        
        return patterns
    
    def apply_refactoring(
        self,
        source_code: str,
        suggestion: RefactoringSuggestion
    ) -> RefactoringResult:
        try:
            if suggestion.refactoring_type == RefactoringType.ORGANIZE_IMPORTS:
                return self._apply_organize_imports(source_code, suggestion)
            elif suggestion.refactoring_type == RefactoringType.IMPROVE_NAMING:
                return self._apply_rename(source_code, suggestion)
            elif suggestion.refactoring_type == RefactoringType.ADD_MISSING_ASSERTIONS:
                return self._apply_add_assertion(source_code, suggestion)
            elif suggestion.refactoring_type == RefactoringType.EXTRACT_METHOD:
                return self._apply_extract_method(source_code, suggestion)
            elif suggestion.refactoring_type == RefactoringType.EXTRACT_TEST_DATA:
                return self._apply_extract_test_data(source_code, suggestion)
            else:
                return RefactoringResult(
                    success=False,
                    refactoring_type=suggestion.refactoring_type,
                    original_code=source_code,
                    refactored_code=source_code,
                    errors=[f"Refactoring type {suggestion.refactoring_type} not yet implemented"]
                )
        except Exception as e:
            logger.error(f"Error applying refactoring: {e}")
            return RefactoringResult(
                success=False,
                refactoring_type=suggestion.refactoring_type,
                original_code=source_code,
                refactored_code=source_code,
                errors=[str(e)]
            )
    
    def _apply_organize_imports(
        self,
        source_code: str,
        suggestion: RefactoringSuggestion
    ) -> RefactoringResult:
        lines = source_code.split('\n')
        
        import_lines = []
        non_import_lines = []
        import_end = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('import\t'):
                import_lines.append(line)
                import_end = i + 1
            elif stripped == '' and import_lines and not non_import_lines:
                continue
            else:
                non_import_lines.append(line)
        
        if not import_lines:
            return RefactoringResult(
                success=False,
                refactoring_type=RefactoringType.ORGANIZE_IMPORTS,
                original_code=source_code,
                refactored_code=source_code,
                errors=["No imports found to organize"]
            )
        
        organized_imports = sorted(set(line.strip() for line in import_lines))
        
        refactored_code = '\n'.join(organized_imports) + '\n\n' + '\n'.join(non_import_lines[import_end:])
        
        return RefactoringResult(
            success=True,
            refactoring_type=RefactoringType.ORGANIZE_IMPORTS,
            original_code=source_code,
            refactored_code=refactored_code,
            changes_made=["Organized imports alphabetically", "Removed duplicate imports"]
        )
    
    def _apply_rename(
        self,
        source_code: str,
        suggestion: RefactoringSuggestion
    ) -> RefactoringResult:
        if not suggestion.original_code or not suggestion.suggested_code:
            return RefactoringResult(
                success=False,
                refactoring_type=RefactoringType.IMPROVE_NAMING,
                original_code=source_code,
                refactored_code=source_code,
                errors=["Missing original or suggested name"]
            )
        
        old_name = suggestion.original_code
        new_name = suggestion.suggested_code
        
        pattern = r'\b' + re.escape(old_name) + r'\b'
        refactored_code = re.sub(pattern, new_name, source_code)
        
        changes_made = []
        if refactored_code != source_code:
            count = len(re.findall(pattern, source_code))
            changes_made.append(f"Renamed '{old_name}' to '{new_name}' ({count} occurrences)")
        
        return RefactoringResult(
            success=len(changes_made) > 0,
            refactoring_type=RefactoringType.IMPROVE_NAMING,
            original_code=source_code,
            refactored_code=refactored_code,
            changes_made=changes_made
        )
    
    def _apply_add_assertion(
        self,
        source_code: str,
        suggestion: RefactoringSuggestion
    ) -> RefactoringResult:
        if suggestion.original_code and suggestion.suggested_code:
            old_assertion = suggestion.original_code
            new_assertion = suggestion.suggested_code
            
            refactored_code = source_code.replace(old_assertion, new_assertion)
            
            return RefactoringResult(
                success=True,
                refactoring_type=RefactoringType.ADD_MISSING_ASSERTIONS,
                original_code=source_code,
                refactored_code=refactored_code,
                changes_made=[f"Strengthened assertion: {old_assertion} -> {new_assertion}"]
            )
        
        lines = source_code.split('\n')
        modified = False
        changes = []
        
        for i, line in enumerate(lines):
            if '@Test' in line:
                j = i + 1
                while j < len(lines) and '{' not in lines[j]:
                    j += 1
                
                if j < len(lines):
                    brace_count = lines[j].count('{') - lines[j].count('}')
                    k = j + 1
                    while k < len(lines) and brace_count > 0:
                        brace_count += lines[k].count('{') - lines[k].count('}')
                        k += 1
                    
                    if k < len(lines):
                        indent = '        '
                        assertion = f'\n{indent}// TODO: Add assertion\n{indent}fail("Test not implemented");\n'
                        lines.insert(k, assertion)
                        modified = True
                        changes.append("Added placeholder assertion")
                        break
        
        if modified:
            return RefactoringResult(
                success=True,
                refactoring_type=RefactoringType.ADD_MISSING_ASSERTIONS,
                original_code=source_code,
                refactored_code='\n'.join(lines),
                changes_made=changes
            )
        
        return RefactoringResult(
            success=False,
            refactoring_type=RefactoringType.ADD_MISSING_ASSERTIONS,
            original_code=source_code,
            refactored_code=source_code,
            errors=["Could not add assertion"]
        )
    
    def _apply_extract_method(
        self,
        source_code: str,
        suggestion: RefactoringSuggestion
    ) -> RefactoringResult:
        return RefactoringResult(
            success=False,
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            original_code=source_code,
            refactored_code=source_code,
            errors=["Extract method refactoring requires manual intervention for complex cases"]
        )
    
    def _apply_extract_test_data(
        self,
        source_code: str,
        suggestion: RefactoringSuggestion
    ) -> RefactoringResult:
        return RefactoringResult(
            success=False,
            refactoring_type=RefactoringType.EXTRACT_TEST_DATA,
            original_code=source_code,
            refactored_code=source_code,
            errors=["Extract test data refactoring requires manual intervention"]
        )
    
    def auto_refactor(
        self,
        source_code: str,
        max_refactorings: int = 5,
        min_confidence: float = 0.8
    ) -> Tuple[str, List[RefactoringResult]]:
        suggestions = self.analyze(source_code)
        
        high_confidence_suggestions = [
            s for s in suggestions
            if s.confidence >= min_confidence and s.priority <= 2
        ][:max_refactorings]
        
        current_code = source_code
        results = []
        
        for suggestion in high_confidence_suggestions:
            result = self.apply_refactoring(current_code, suggestion)
            if result.success:
                current_code = result.refactored_code
                results.append(result)
                self._refactoring_history.append(result)
        
        return current_code, results
    
    def get_refactoring_summary(self) -> Dict[str, Any]:
        if not self._refactoring_history:
            return {"total_refactorings": 0}
        
        by_type = defaultdict(int)
        for result in self._refactoring_history:
            by_type[result.refactoring_type.value] += 1
        
        return {
            "total_refactorings": len(self._refactoring_history),
            "by_type": dict(by_type),
            "success_rate": sum(1 for r in self._refactoring_history if r.success) / len(self._refactoring_history)
        }
