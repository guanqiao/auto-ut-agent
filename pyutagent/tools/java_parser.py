"""Java code parser using tree-sitter."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

from tree_sitter import Language, Parser, Node
import tree_sitter_java as tsjava


@dataclass
class JavaMethod:
    """Represents a Java method."""
    name: str
    return_type: Optional[str]
    parameters: List[Tuple[str, str]]  # [(type, name), ...]
    modifiers: List[str]
    annotations: List[str]
    start_line: int
    end_line: int


@dataclass
class JavaClass:
    """Represents a Java class."""
    package: str
    name: str
    methods: List[JavaMethod]
    fields: List[Tuple[str, str, str]]  # [(type, name, modifiers), ...]
    imports: List[str]
    annotations: List[str]


class JavaCodeParser:
    """Parser for Java source code using tree-sitter."""
    
    def __init__(self):
        """Initialize the Java parser."""
        self.parser = Parser(Language(tsjava.language()))
    
    def parse(self, code: bytes) -> JavaClass:
        """Parse Java code and extract class information.
        
        Args:
            code: Java source code as bytes
            
        Returns:
            JavaClass with extracted information
        """
        tree = self.parser.parse(code)
        root = tree.root_node
        
        # Extract package
        package = self._extract_package(root)
        
        # Extract imports
        imports = self._extract_imports(root)
        
        # Find class declaration
        class_node = self._find_class_node(root)
        
        if class_node is None:
            return JavaClass(
                package=package,
                name="",
                methods=[],
                fields=[],
                imports=imports,
                annotations=[]
            )
        
        # Extract class name
        name = self._extract_class_name(class_node)
        
        # Extract class annotations
        annotations = self._extract_annotations(class_node)
        
        # Extract methods
        methods = self._extract_methods(class_node)
        
        # Extract fields
        fields = self._extract_fields(class_node)
        
        return JavaClass(
            package=package,
            name=name,
            methods=methods,
            fields=fields,
            imports=imports,
            annotations=annotations
        )
    
    def parse_file(self, file_path: str) -> JavaClass:
        """Parse a Java file.
        
        Args:
            file_path: Path to Java file
            
        Returns:
            JavaClass with extracted information
        """
        with open(file_path, 'rb') as f:
            code = f.read()
        return self.parse(code)
    
    def _extract_package(self, root: Node) -> str:
        """Extract package declaration."""
        for child in root.children:
            if child.type == 'package_declaration':
                # Extract identifier from package declaration
                for subchild in child.children:
                    if subchild.type == 'scoped_identifier' or subchild.type == 'identifier':
                        return subchild.text.decode()
        return ""
    
    def _extract_imports(self, root: Node) -> List[str]:
        """Extract import statements."""
        imports = []
        for child in root.children:
            if child.type == 'import_declaration':
                # Extract the import path
                for subchild in child.children:
                    if subchild.type == 'scoped_identifier' or subchild.type == 'identifier':
                        imports.append(subchild.text.decode())
        return imports
    
    def _find_class_node(self, root: Node) -> Optional[Node]:
        """Find the class declaration node."""
        for child in root.children:
            if child.type == 'class_declaration':
                return child
        return None
    
    def _extract_class_name(self, class_node: Node) -> str:
        """Extract class name from class declaration."""
        for child in class_node.children:
            if child.type == 'identifier':
                return child.text.decode()
        return ""
    
    def _extract_annotations(self, node: Node) -> List[str]:
        """Extract annotations from a node."""
        annotations = []
        
        # Look for modifiers node which contains annotations
        for child in node.children:
            if child.type == 'modifiers':
                for modifier in child.children:
                    if modifier.type == 'annotation':
                        # Extract annotation text
                        annotation_text = modifier.text.decode()
                        annotations.append(annotation_text)
        
        return annotations
    
    def _extract_methods(self, class_node: Node) -> List[JavaMethod]:
        """Extract methods from class declaration."""
        methods = []
        
        # Find class body
        class_body = None
        for child in class_node.children:
            if child.type == 'class_body':
                class_body = child
                break
        
        if class_body is None:
            return methods
        
        # Find method declarations in class body
        for child in class_body.children:
            if child.type == 'method_declaration':
                method = self._parse_method(child)
                if method:
                    methods.append(method)
            elif child.type == 'constructor_declaration':
                method = self._parse_method(child, is_constructor=True)
                if method:
                    methods.append(method)
        
        return methods
    
    def _parse_method(self, method_node: Node, is_constructor: bool = False) -> Optional[JavaMethod]:
        """Parse a method declaration node."""
        name = None
        return_type = None
        parameters = []
        modifiers = []
        annotations = []
        
        for child in method_node.children:
            if child.type == 'modifiers':
                modifiers, annotations = self._parse_modifiers(child)
            elif child.type == 'type_identifier' or child.type == 'integral_type' or child.type == 'floating_point_type':
                return_type = child.text.decode()
            elif child.type == 'identifier' and name is None:
                name = child.text.decode()
            elif child.type == 'formal_parameters':
                parameters = self._parse_parameters(child)
            elif child.type == 'constructor_body' or child.type == 'block':
                # This is the method body, skip
                pass
        
        # For constructors, name is the class name and return type is None
        if is_constructor:
            # Find the identifier before formal_parameters
            for i, child in enumerate(method_node.children):
                if child.type == 'formal_parameters' and i > 0:
                    prev = method_node.children[i-1]
                    if prev.type == 'identifier':
                        name = prev.text.decode()
                        break
            return_type = None
        
        if name is None:
            return None
        
        return JavaMethod(
            name=name,
            return_type=return_type,
            parameters=parameters,
            modifiers=modifiers,
            annotations=annotations,
            start_line=method_node.start_point[0],
            end_line=method_node.end_point[0]
        )
    
    def _parse_modifiers(self, modifiers_node: Node) -> Tuple[List[str], List[str]]:
        """Parse modifiers node to extract modifiers and annotations."""
        modifiers = []
        annotations = []
        
        for child in modifiers_node.children:
            if child.type == 'modifier':
                modifiers.append(child.text.decode())
            elif child.type == 'annotation':
                annotations.append(child.text.decode())
        
        return modifiers, annotations
    
    def _parse_parameters(self, params_node: Node) -> List[Tuple[str, str]]:
        """Parse formal parameters."""
        parameters = []
        
        for child in params_node.children:
            if child.type == 'formal_parameter':
                param_type = None
                param_name = None
                
                for subchild in child.children:
                    if subchild.type in ('type_identifier', 'integral_type', 'floating_point_type'):
                        param_type = subchild.text.decode()
                    elif subchild.type == 'identifier':
                        param_name = subchild.text.decode()
                
                if param_type and param_name:
                    parameters.append((param_type, param_name))
        
        return parameters
    
    def _extract_fields(self, class_node: Node) -> List[Tuple[str, str, str]]:
        """Extract fields from class declaration."""
        fields = []
        
        # Find class body
        class_body = None
        for child in class_node.children:
            if child.type == 'class_body':
                class_body = child
                break
        
        if class_body is None:
            return fields
        
        # Find field declarations
        for child in class_body.children:
            if child.type == 'field_declaration':
                field = self._parse_field(child)
                if field:
                    fields.append(field)
        
        return fields
    
    def _parse_field(self, field_node: Node) -> Optional[Tuple[str, str, str]]:
        """Parse a field declaration."""
        field_type = None
        field_name = None
        modifiers_str = ""
        
        for child in field_node.children:
            if child.type == 'modifiers':
                modifiers_str = child.text.decode()
            elif child.type in ('type_identifier', 'integral_type', 'floating_point_type'):
                field_type = child.text.decode()
            elif child.type == 'variable_declarator':
                # Extract name from variable declarator
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        field_name = subchild.text.decode()
                        break
        
        if field_type and field_name:
            return (field_type, field_name, modifiers_str)
        
        return None
