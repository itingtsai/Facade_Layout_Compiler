import json
import sys
import re

# --- 1. FRONT-END: Lexer & Parser ---

class Token:
    def __init__(self, type, value, line=0, col=0):
        self.type = type
        self.value = value
        self.line = line
        self.col = col
    
    def __repr__(self):
        return f"Token({self.type}, {self.value})"

class Lexer:
    """Proper tokenizer for the facade DSL"""
    TOKEN_PATTERNS = [
        ('GRID_DECL', r'grid\s+(\d+)x(\d+)'),
        ('ROW_DECL', r'row\s+(\d+):'),
        ('RULE_DECL', r'rule\s+(\w+):'),
        ('SYMBOL', r'[EWSD]'),
        ('NUMBER', r'\d+'),
        ('IDENTIFIER', r'\w+'),
        ('WHITESPACE', r'\s+'),
        ('COMMENT', r'#.*'),
    ]
    
    def __init__(self, source):
        self.source = source
        self.tokens = []
        self.line = 1
        self.col = 1
    
    def tokenize(self):
        lines = self.source.strip().split('\n')
        for line_num, line in enumerate(lines, 1):
            col = 0
            while col < len(line):
                matched = False
                for token_type, pattern in self.TOKEN_PATTERNS:
                    regex = re.compile(pattern)
                    match = regex.match(line, col)
                    if match:
                        value = match.group(0)
                        if token_type not in ['WHITESPACE', 'COMMENT']:
                            if token_type == 'GRID_DECL':
                                self.tokens.append(Token('GRID_DECL', 
                                    (int(match.group(1)), int(match.group(2))), 
                                    line_num, col))
                            elif token_type == 'ROW_DECL':
                                self.tokens.append(Token('ROW_DECL', 
                                    int(match.group(1)), line_num, col))
                            elif token_type == 'RULE_DECL':
                                self.tokens.append(Token('RULE_DECL', 
                                    match.group(1), line_num, col))
                            else:
                                self.tokens.append(Token(token_type, value, line_num, col))
                        col = match.end()
                        matched = True
                        break
                
                if not matched:
                    print(f"Lexical error at line {line_num}, col {col}: '{line[col]}'")
                    col += 1
        
        return self.tokens

class FacadeParser:
    def __init__(self):
        self.grid = []
        self.grid_size = None
        self.rules = {}
        self.tokens = []
        self.pos = 0
    
    def parse(self, input_text):
        """Parse DSL into IR with grid and rules"""
        lexer = Lexer(input_text)
        self.tokens = lexer.tokenize()
        self.pos = 0
        
        # Parse grid declaration if present
        if self.pos < len(self.tokens) and self.tokens[self.pos].type == 'GRID_DECL':
            self.grid_size = self.tokens[self.pos].value
            self.pos += 1
        
        # Parse rules if present
        while self.pos < len(self.tokens) and self.tokens[self.pos].type == 'RULE_DECL':
            rule_name = self.tokens[self.pos].value
            self.pos += 1
            # Simple rule value parsing
            rule_value = []
            while self.pos < len(self.tokens) and self.tokens[self.pos].type in ['IDENTIFIER', 'NUMBER']:
                rule_value.append(self.tokens[self.pos].value)
                self.pos += 1
            self.rules[rule_name] = ' '.join(rule_value)
        
        # Parse rows
        ir_grid = []
        while self.pos < len(self.tokens):
            if self.tokens[self.pos].type == 'ROW_DECL':
                row_num = self.tokens[self.pos].value
                self.pos += 1
                
                # Collect symbols for this row
                row_symbols = []
                while self.pos < len(self.tokens) and self.tokens[self.pos].type == 'SYMBOL':
                    row_symbols.append(self.tokens[self.pos].value)
                    self.pos += 1
                
                ir_grid.append(row_symbols)
            else:
                self.pos += 1
        
        self.grid = ir_grid
        return {
            'grid': ir_grid,
            'grid_size': self.grid_size,
            'rules': self.rules
        }

# --- 2. MIDDLE-END: Semantic Analysis & Transformations ---

class FacadeAnalyzer:
    def __init__(self, ir):
        self.grid = ir['grid']
        self.grid_size = ir.get('grid_size')
        self.rules = ir.get('rules', {})
        self.errors = []
        self.warnings = []

    def analyze(self):
        self._check_grid_consistency()
        self._check_grid_size_match()
        self._check_doors_ground_floor()
        self._check_window_spacing()
        self._check_column_alignment()
        self._check_symmetry()
        self._check_no_window_above_door()  # Extensibility example
        
        return {
            "is_valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings
        }

    def _check_grid_consistency(self):
        if not self.grid:
            self.errors.append("Grid is empty.")
            return
        
        width = len(self.grid[0])
        for i, row in enumerate(self.grid):
            if len(row) != width:
                self.errors.append(f"Row {i+1} length mismatch. Expected {width}, got {len(row)}.")

    def _check_grid_size_match(self):
        if self.grid_size:
            expected_cols, expected_rows = self.grid_size
            if len(self.grid) != expected_rows:
                self.errors.append(f"Grid size mismatch: expected {expected_rows} rows, got {len(self.grid)}.")
            if self.grid and len(self.grid[0]) != expected_cols:
                self.errors.append(f"Grid size mismatch: expected {expected_cols} columns, got {len(self.grid[0])}.")

    def _check_doors_ground_floor(self):
        """Rule: Doors (D) only allowed on the last row"""
        total_rows = len(self.grid)
        for r_idx, row in enumerate(self.grid):
            if r_idx == total_rows - 1:
                continue
            
            if 'D' in row:
                self.errors.append(f"Structural Violation: Door found on upper floor (Row {r_idx+1}).")

    def _check_window_spacing(self):
        """Rule: Windows should have at least 1 cell spacing"""
        min_spacing = int(self.rules.get('min_window_spacing', 1))
        
        for r_idx, row in enumerate(self.grid):
            last_window_pos = -10
            for c_idx, cell in enumerate(row):
                if cell == 'W':
                    if c_idx - last_window_pos - 1 < min_spacing:
                        self.warnings.append(
                            f"Window spacing violation at Row {r_idx+1}, Col {c_idx+1}: "
                            f"spacing is {c_idx - last_window_pos - 1}, minimum is {min_spacing}."
                        )
                    last_window_pos = c_idx

    def _check_column_alignment(self):
        """Rule: Check if windows align vertically"""
        if not self.grid or len(self.grid) < 2:
            return
        
        cols = len(self.grid[0])
        for c in range(cols):
            window_rows = [r for r in range(len(self.grid)) if self.grid[r][c] == 'W']
            if len(window_rows) >= 2:
                # Check if windows are consistently aligned
                gaps = [window_rows[i+1] - window_rows[i] for i in range(len(window_rows)-1)]
                if len(set(gaps)) > 1:
                    self.warnings.append(f"Column {c+1}: Windows not evenly spaced vertically.")

    def _check_symmetry(self):
        """Rule: Rows should be palindromes"""
        for r_idx, row in enumerate(self.grid):
            if row != row[::-1]:
                self.warnings.append(f"Asymmetry detected on Row {r_idx+1}.")

    def _check_no_window_above_door(self):
        """Extensibility example: No window directly above door"""
        if not self.grid or len(self.grid) < 2:
            return
        
        # Check last row for doors
        last_row = len(self.grid) - 1
        for c_idx, cell in enumerate(self.grid[last_row]):
            if cell == 'D':
                # Check if there's a window above
                for r_idx in range(last_row):
                    if self.grid[r_idx][c_idx] == 'W':
                        self.warnings.append(
                            f"Design warning: Window at Row {r_idx+1}, Col {c_idx+1} "
                            f"is above door at Row {last_row+1}, Col {c_idx+1}."
                        )

class FacadeTransformer:
    """Transform passes for the IR"""
    
    @staticmethod
    def apply_symmetry(grid):
        """Auto-expand symmetry: mirror each row"""
        transformed = []
        for row in grid:
            mid = len(row) // 2
            left_half = row[:mid]
            # Mirror to create symmetry
            mirrored_row = left_half + left_half[::-1]
            if len(row) % 2 == 1:
                mirrored_row = left_half + [row[mid]] + left_half[::-1]
            transformed.append(mirrored_row)
        return transformed
    
    @staticmethod
    def run_length_encode(grid):
        """Compact consecutive empty cells using RLE"""
        compressed = []
        for row in grid:
            compressed_row = []
            i = 0
            while i < len(row):
                if row[i] == 'E':
                    count = 1
                    while i + count < len(row) and row[i + count] == 'E':
                        count += 1
                    compressed_row.append(f'E{count}' if count > 1 else 'E')
                    i += count
                else:
                    compressed_row.append(row[i])
                    i += 1
            compressed.append(compressed_row)
        return compressed
    
    @staticmethod
    def run_length_decode(compressed_grid):
        """Expand RLE back to normal grid"""
        expanded = []
        for row in compressed_grid:
            expanded_row = []
            for token in row:
                if token.startswith('E') and len(token) > 1:
                    count = int(token[1:])
                    expanded_row.extend(['E'] * count)
                else:
                    expanded_row.append(token)
            expanded.append(expanded_row)
        return expanded

# --- 3. BACK-END: Code Generation ---

class FacadeGenerator:
    def __init__(self, ir):
        self.grid = ir['grid']
        self.grid_size = ir.get('grid_size')
        self.rules = ir.get('rules', {})
        
        self.style_map = {
            'E': {'fill': 'white', 'stroke': '#ddd'},
            'W': {'fill': '#89CFF0', 'stroke': '#333'},
            'S': {'fill': '#E6E6FA', 'stroke': '#333'},
            'D': {'fill': '#8B4513', 'stroke': '#000'}
        }
        self.cell_size = 50

    def generate_json(self, analysis_result=None):
        """Enhanced JSON with metadata and validation results"""
        output = {
            "meta": {
                "rows": len(self.grid),
                "cols": len(self.grid[0]) if self.grid else 0,
                "declared_size": self.grid_size,
                "rules": self.rules
            },
            "layout": self.grid
        }
        
        if analysis_result:
            output["validation"] = {
                "is_valid": analysis_result['is_valid'],
                "errors": analysis_result['errors'],
                "warnings": analysis_result['warnings']
            }
        
        return json.dumps(output, indent=2)

    def generate_svg(self):
        """Generate optimized SVG with grouping"""
        if not self.grid:
            return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        
        rows = len(self.grid)
        cols = len(self.grid[0])
        width = cols * self.cell_size
        height = rows * self.cell_size
        
        svg_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" version="1.1">'
        ]
        
        # Group by cell type for better organization
        for cell_type in ['E', 'S', 'W', 'D']:
            style = self.style_map[cell_type]
            svg_lines.append(f'  <g id="{cell_type}-group">')
            
            for r, row in enumerate(self.grid):
                for c, token in enumerate(row):
                    if token == cell_type:
                        x = c * self.cell_size
                        y = r * self.cell_size
                        
                        rect = (
                            f'    <rect x="{x}" y="{y}" width="{self.cell_size}" height="{self.cell_size}" '
                            f'fill="{style["fill"]}" stroke="{style["stroke"]}" stroke-width="2"/>'
                        )
                        text = (
                            f'    <text x="{x + self.cell_size/2}" y="{y + self.cell_size/2 + 4}" '
                            f'font-family="Arial" font-size="14" text-anchor="middle" '
                            f'dominant-baseline="middle" fill="#555">{token}</text>'
                        )
                        svg_lines.append(rect)
                        svg_lines.append(text)
            
            svg_lines.append('  </g>')
        
        svg_lines.append('</svg>')
        return "\n".join(svg_lines)

# --- 4. COMPILER DRIVER ---

def compile_facade(source_code, apply_transforms=False):
    print("=" * 60)
    print("--- FACADE LAYOUT COMPILER ---")
    print("=" * 60)
    
    # 1. FRONT-END: Parse
    print("\n[PHASE 1: PARSING]")
    parser = FacadeParser()
    ir = parser.parse(source_code)
    print(f"✓ Parsed {len(ir['grid'])} rows")
    if ir['grid_size']:
        print(f"✓ Grid size: {ir['grid_size'][0]}x{ir['grid_size'][1]}")
    if ir['rules']:
        print(f"✓ Rules: {ir['rules']}")
    
    # 2. MIDDLE-END: Transform (optional)
    if apply_transforms:
        print("\n[PHASE 2: TRANSFORMATIONS]")
        original_grid = ir['grid']
        
        # Show RLE compression
        compressed = FacadeTransformer.run_length_encode(ir['grid'])
        print(f"✓ RLE Compression: {original_grid[0]} -> {compressed[0]}")
        
        # Verify round-trip
        decoded = FacadeTransformer.run_length_decode(compressed)
        assert decoded == original_grid, "RLE round-trip failed!"
        print("✓ RLE round-trip verified")
    
    # 3. MIDDLE-END: Analyze
    print("\n[PHASE 3: SEMANTIC ANALYSIS]")
    analyzer = FacadeAnalyzer(ir)
    result = analyzer.analyze()
    
    if not result['is_valid']:
        print("✗ Validation FAILED:")
        for err in result['errors']:
            print(f"   [ERROR] {err}")
        return None, None, result
    
    print("✓ Validation PASSED")
    for warn in result['warnings']:
        print(f"   [WARN] {warn}")

    # 4. BACK-END: Generate
    print("\n[PHASE 4: CODE GENERATION]")
    generator = FacadeGenerator(ir)
    json_out = generator.generate_json(result)
    svg_out = generator.generate_svg()
    print("✓ JSON generated")
    print("✓ SVG generated")
    
    return json_out, svg_out, result

# --- 5. TEST SUITE ---

def run_test_suite():
    """Run 5+ test cases as per proposal"""
    
    test_cases = [
        # Test 1: Valid symmetric facade
        {
            "name": "Valid Symmetric Facade",
            "code": """
            grid 8x5
            row 1: E E E E E E E E
            row 2: E E E E E E E E
            row 3: E W S S S S W E
            row 4: E W S S S S W E
            row 5: E E E D D E E E
            """,
            "should_pass": True
        },
        
        # Test 2: Invalid - door on upper floor
        {
            "name": "Invalid - Door on Upper Floor",
            "code": """
            row 1: E D E
            row 2: E E E
            """,
            "should_pass": False
        },
        
        # Test 3: Invalid - grid size mismatch
        {
            "name": "Invalid - Grid Size Mismatch",
            "code": """
            grid 4x2
            row 1: E E E
            row 2: E E E
            """,
            "should_pass": False
        },
        
        # Test 4: Valid with rules
        {
            "name": "Valid with Window Spacing Rule",
            "code": """
            rule min_window_spacing: 2
            row 1: E W E E W E
            row 2: E E E E E E
            """,
            "should_pass": True
        },
        
        # Test 5: Asymmetric (warning only)
        {
            "name": "Asymmetric Facade (Warning)",
            "code": """
            row 1: E W S E
            row 2: E E E E
            """,
            "should_pass": True  # Warnings don't fail
        },
        
        # Test 6: Window above door (extensibility)
        {
            "name": "Window Above Door (Extensibility Demo)",
            "code": """
            row 1: E W E
            row 2: E D E
            """,
            "should_pass": True  # Warning only
        }
    ]
    
    print("\n" + "=" * 60)
    print("RUNNING TEST SUITE")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test['name']} ---")
        json_out, svg_out, result = compile_facade(test['code'])
        
        actual_pass = result['is_valid'] if result else False
        expected_pass = test['should_pass']
        
        if actual_pass == expected_pass:
            print(f"✓ Test {i} PASSED")
            passed += 1
        else:
            print(f"✗ Test {i} FAILED (expected {'pass' if expected_pass else 'fail'}, got {'pass' if actual_pass else 'fail'})")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} total")
    print("=" * 60)

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # Run test suite first
    # run_test_suite()
    
    # Then compile the main example
    # print("\n\n")

    # source_code = """
    # grid 8x5
    # rule min_window_spacing: 1
    # row 1: E E E E E E E E
    # row 2: E E E E E E E E
    # row 3: E W S S S S W E
    # row 4: E W S S S S W E
    # row 5: E E E D D E E E
    # """

    # test7
    source_code = """
    row 1: E E W E S E E W E E
    row 2: E S W W S E W S E E
    row 3: E W S S S W S E E E
    row 4: S S W S E S W S S E
    row 5: E E S W S S E W S E
    row 6: D E E D S E E D E E
    """

    
    json_result, svg_result, analysis = compile_facade(source_code, apply_transforms=True)
    
    if json_result:
        with open('/Users/it84/Desktop/GitHub/facade_layout_compiler/enhanced_facade.json', 'w') as f:
            f.write(json_result)
        
        with open('/Users/it84/Desktop/GitHub/facade_layout_compiler/enhanced_facade.svg', 'w') as f:
            f.write(svg_result)
        
        print("\n" + "=" * 60)
        print("FILES SAVED:")
        print("  - enhanced_facade.json")
        print("  - enhanced_facade.svg")
        print("=" * 60)