import json
import sys
import re
import cairosvg

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
        ('SYMBOL', r'[EWSDC]'),
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
        self._check_chimney_placement()
        self._check_no_window_above_door()
        
        return {
            "is_valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings
        }

    def _check_grid_consistency(self):
        """Check that all rows have the same length"""
        if not self.grid:
            self.errors.append("Grid is empty.")
            return
        
        width = len(self.grid[0])
        for i, row in enumerate(self.grid):
            if len(row) != width:
                self.errors.append(f"Row {i+1} length mismatch. Expected {width}, got {len(row)}.")

    def _check_grid_size_match(self):
        """Check that grid matches declared size"""
        if self.grid_size:
            expected_cols, expected_rows = self.grid_size
            if len(self.grid) != expected_rows:
                self.errors.append(f"Grid size mismatch: expected {expected_rows} rows, got {len(self.grid)}.")
            if self.grid and len(self.grid[0]) != expected_cols:
                self.errors.append(f"Grid size mismatch: expected {expected_cols} columns, got {len(self.grid[0])}.")

    def _check_doors_ground_floor(self):
        """
        Rule: Doors must be continuous from ground floor, same column, no gaps.
        ERROR if door doesn't reach ground or has gaps.
        """
        if not self.grid:
            return
        
        total_rows = len(self.grid)
        
        # Track door columns and their extent
        door_columns = {}
        
        # Scan grid to find all doors
        for r_idx, row in enumerate(self.grid):
            for c_idx, cell in enumerate(row):
                if cell == 'D':
                    if c_idx not in door_columns:
                        door_columns[c_idx] = []
                    door_columns[c_idx].append(r_idx)
        
        # Validate each door column
        for col, rows in door_columns.items():
            rows_sorted = sorted(rows)
            
            # Rule 1: Doors must reach ground floor (last row)
            if total_rows - 1 not in rows_sorted:
                self.errors.append(
                    f"Structural Violation: Door at column {col+1} does not reach ground floor (Row {total_rows})."
                )
            
            # Rule 2: Doors must be continuous (no gaps)
            for i in range(len(rows_sorted) - 1):
                if rows_sorted[i+1] - rows_sorted[i] != 1:
                    self.errors.append(
                        f"Structural Violation: Door at column {col+1} has a gap between Row {rows_sorted[i]+1} and Row {rows_sorted[i+1]+1}."
                    )

    def _check_window_spacing(self):
        """Rule: Windows should have at least minimum spacing (configurable via rules)"""
        min_spacing = int(self.rules.get('min_window_spacing', 0))
        
        for r_idx, row in enumerate(self.grid):
            last_window_pos = -min_spacing - 1  # Initialize to allow first window
            for c_idx, cell in enumerate(row):
                if cell == 'W':
                    actual_spacing = c_idx - last_window_pos - 1
                    if actual_spacing < min_spacing:
                        self.warnings.append(
                            f"Window spacing violation at Row {r_idx+1}, Col {c_idx+1}: "
                            f"spacing is {actual_spacing}, minimum is {min_spacing}."
                        )
                    last_window_pos = c_idx

    def _check_column_alignment(self):
        """Rule: Check if windows align vertically (warning only)"""
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
        """
        Rule: Rows should be palindromes (symmetric).
        WARNING only if asymmetric.
        """
        for r_idx, row in enumerate(self.grid):
            if row != row[::-1]:
                self.warnings.append(f"Asymmetry detected on Row {r_idx+1}.")

    def _check_chimney_placement(self):
        """
        Rule: Chimneys must form connected regions that start from the top row.
        Chimneys can be diagonal/staircase (horizontally adjacent between rows) as long as:
        1. The chimney region touches the top row (row 0)
        2. The chimney region is continuous (no disconnected chimney cells)
        ERROR if chimney doesn't connect to top or is disconnected.
        """
        if not self.grid:
            return
        
        # Find all chimney cells
        chimney_cells = set()
        for r_idx, row in enumerate(self.grid):
            for c_idx, cell in enumerate(row):
                if cell == 'C':
                    chimney_cells.add((r_idx, c_idx))
        
        if not chimney_cells:
            return  # No chimneys to check
        
        # Find connected components using flood fill
        def get_neighbors(cell):
            """Get adjacent cells (up, down, left, right)"""
            r, c = cell
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in chimney_cells:
                    neighbors.append((nr, nc))
            return neighbors
        
        def flood_fill(start, visited):
            """Find all cells in the connected component containing start"""
            component = set()
            stack = [start]
            while stack:
                cell = stack.pop()
                if cell in visited:
                    continue
                visited.add(cell)
                component.add(cell)
                for neighbor in get_neighbors(cell):
                    if neighbor not in visited:
                        stack.append(neighbor)
            return component
        
        # Find all connected components
        visited = set()
        components = []
        for cell in chimney_cells:
            if cell not in visited:
                component = flood_fill(cell, visited)
                components.append(component)
        
        # Validate each connected component
        for comp_idx, component in enumerate(components):
            # Get the rows and columns in this component
            rows_in_component = {cell[0] for cell in component}
            cols_in_component = {cell[1] for cell in component}
            
            # Rule 1: Component must touch top row (row 0)
            if 0 not in rows_in_component:
                min_col = min(cols_in_component) + 1
                max_col = max(cols_in_component) + 1
                col_desc = f"column {min_col}" if min_col == max_col else f"columns {min_col}-{max_col}"
                self.errors.append(
                    f"Structural Violation: Chimney region at {col_desc} does not start from top row (Row 1)."
                )

    def _check_no_window_above_door(self):
        """
        Rule: No window directly adjacent above door.
        WARNING only (design suggestion, not structural error).
        """
        if not self.grid or len(self.grid) < 2:
            return
        
        # Find all door columns and their topmost door row
        for c_idx in range(len(self.grid[0])):
            # Find the topmost door in this column
            topmost_door_row = None
            for r_idx in range(len(self.grid)):
                if self.grid[r_idx][c_idx] == 'D':
                    topmost_door_row = r_idx
                    break  # First door found is topmost
            
            # If there's a door and a row above it, check for adjacent window
            if topmost_door_row is not None and topmost_door_row > 0:
                if self.grid[topmost_door_row - 1][c_idx] == 'W':
                    self.warnings.append(
                        f"Design warning: Window at Row {topmost_door_row}, Col {c_idx+1} "
                        f"is directly above door at Row {topmost_door_row+1}, Col {c_idx+1}."
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
            'D': {'fill': '#8B4513', 'stroke': '#000'},
            'C': {'fill': '#DC143C', 'stroke': '#8B0000'}  # Chimney (Crimson red with dark red border)
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
        for cell_type in ['E', 'S', 'W', 'D', 'C']:
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

    def generate_png(self, svg_content, output_path, scale=2):
        """
        Generate PNG from SVG content.
        
        Args:
            svg_content: The SVG string to convert
            output_path: Path to save the PNG file
            scale: Scale factor for higher resolution (default 2x)
        """
        cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=output_path,
            scale=scale
        )


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
    
    # 2. MIDDLE-END: Transform
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
    else:
        print("✓ Validation PASSED")
    
    for warn in result['warnings']:
        print(f"   [WARN] {warn}")
    
    if not result['is_valid']:
        return None, None, result, None

    # 4. BACK-END: Generate
    print("\n[PHASE 4: CODE GENERATION]")
    generator = FacadeGenerator(ir)
    json_out = generator.generate_json(result)
    svg_out = generator.generate_svg()
    print("✓ JSON generated")
    print("✓ SVG generated")
    
    return json_out, svg_out, result, generator


# --- 5. TEST SUITE ---

def run_test_suite():
    """Run comprehensive test cases"""
    
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
            "should_pass": True,
            "expected_warnings": 0,
            "expected_errors": 0
        },
        
        # Test 2: Invalid - door on upper floor (ERROR)
        {
            "name": "Invalid - Door on Upper Floor",
            "code": """
            row 1: E D E
            row 2: E E E
            """,
            "should_pass": False,
            "expected_errors": 1
        },
        
        # Test 3: Invalid - grid size mismatch (ERROR)
        {
            "name": "Invalid - Grid Size Mismatch",
            "code": """
            grid 4x2
            row 1: E E E
            row 2: E E E
            """,
            "should_pass": False,
            "expected_errors": 1
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
        
        # Test 5: Asymmetric (WARNING only, should pass)
        {
            "name": "Asymmetric Facade (Warning Only)",
            "code": """
            row 1: E W S E
            row 2: E E E E
            """,
            "should_pass": True,
            "expected_warnings": 1  # Asymmetry warning
        },
        
        # Test 6: Window above door (WARNING only, should pass)
        {
            "name": "Window Above Door (Warning Only)",
            "code": """
            row 1: E W E
            row 2: E D E
            """,
            "should_pass": True,
            "expected_warnings": 2  # Window above door + asymmetry
        },
        
        # Test 7: Invalid - chimney not starting from top (ERROR)
        {
            "name": "Invalid - Chimney Not From Top",
            "code": """
            row 1: E E E
            row 2: E C E
            row 3: E E E
            """,
            "should_pass": False,
            "expected_errors": 1
        },
        
        # Test 8: Invalid - chimney with gap (ERROR) - disconnected chimney regions
        {
            "name": "Invalid - Chimney With Gap (Disconnected)",
            "code": """
            row 1: E C E
            row 2: E E E
            row 3: E C E
            """,
            "should_pass": False,
            "expected_errors": 1  # The bottom C is disconnected and doesn't reach top
        },
        
        # Test 9: Valid chimney from top
        {
            "name": "Valid - Chimney From Top",
            "code": """
            row 1: E C E
            row 2: E C E
            row 3: E E E
            """,
            "should_pass": True
        },
        
        # Test 10: Invalid - door with gap (ERROR)
        {
            "name": "Invalid - Door With Gap",
            "code": """
            row 1: E E E
            row 2: E D E
            row 3: E E E
            row 4: E D E
            """,
            "should_pass": False,
            "expected_errors": 1  # Gap error
        },
        
        # Test 11: Valid multi-row door
        {
            "name": "Valid - Multi-row Door",
            "code": """
            row 1: E E E
            row 2: E D E
            row 3: E D E
            """,
            "should_pass": True
        },
        
        # Test 12: Complex valid facade with chimney and door
        {
            "name": "Valid - Complex Facade",
            "code": """
            row 1: E E C E E
            row 2: E W C W E
            row 3: E W C W E
            row 4: E E D E E
            """,
            "should_pass": True
        },
        
        # Test 13: Valid - Diagonal/Staircase Chimney (NEW TEST)
        {
            "name": "Valid - Diagonal Chimney",
            "code": """
            row 1: E E C E E E E E E E E
            row 2: W W C C W W W W W W W
            row 3: W E S C C S S S W E W
            row 4: W E S S C C S W W E W
            row 5: E E S S S C C E E E E
            row 6: E D D E E E C C D D E
            """,
            "should_pass": True
        },
        
        # Test 14: Valid - Another diagonal chimney pattern
        {
            "name": "Valid - Diagonal Chimney Left to Right",
            "code": """
            row 1: C E E E E
            row 2: C C E E E
            row 3: E C C E E
            row 4: E E C C E
            row 5: E E E C E
            """,
            "should_pass": True
        },
        
        # Test 15: Invalid - Disconnected diagonal chimney (gap in the middle)
        {
            "name": "Invalid - Disconnected Diagonal Chimney",
            "code": """
            row 1: C E E E E
            row 2: C E E E E
            row 3: E E E E E
            row 4: E E C C E
            row 5: E E E C E
            """,
            "should_pass": False,
            "expected_errors": 1  # Bottom chimney region doesn't connect to top
        },
    ]
    
    print("\n" + "=" * 60)
    print("RUNNING TEST SUITE")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test['name']} ---")
        json_out, svg_out, result, generator = compile_facade(test['code'])
        
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
    
    return passed, failed


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Facade Layout Compiler')
    parser.add_argument('--test', action='store_true', help='Run test suite')
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    args = parser.parse_args()
    
    if args.test:
        run_test_suite()
    else:
        # # facade_1
        # source_code = """ 
        # row 1: E E E E E E E E C E E 
        # row 2: S S S S S S S S C S S 
        # row 3: S S W W S S S W W S S 
        # row 4: S S W W S S S W W S S 
        # row 5: S S S S S S S S S S S 
        # row 6: S S D D S S W W W W S 
        # row 7: S S D D S S W W W W S 
        # """

        # facade_2
        # source_code = """
        # row 1: E W S E E W W E S W E
        # row 2: W E W S E E S W E E E
        # row 3: E S E W W S E E W S E
        # row 4: S W S S E W E S E W E
        # row 5: E E W E S S W E E S W
        # row 6: W S E E W E S W S E E
        # row 7: D D E D E E D E D D E
        # """

        # # facade_3
        # source_code = """
        # row 1: E C E S E S E C E
        # row 2: E C E W E W E C E
        # row 3: E C E W E W E C E
        # row 4: E C E W W W E C E
        # row 5: E C E W E W E C E
        # row 6: E C E W E W E C E
        # row 7: E C E S S S E C E
        # row 8: E E E S S S E E E
        # row 9: E D D E E E D D E
        # """

        # # facade_4
        # source_code = """
        # row 1: E E C E E E E E E E E
        # row 2: W W C C W W W W W W W
        # row 3: W S S C C S S S W S W
        # row 4: W S S S C C S W W S W
        # row 5: E S S S S C C S S S E
        # row 6: E D D S S S C S D D E
        # """

        # facade_5
        source_code = """
        row 1: E C E E E E C E E E E C E
        row 2: E C S W W W C W W W S C E
        row 3: E C S W S W C W S W S C E
        row 4: E C S W S W C W S W S C E
        row 5: E C S W W W C W W W S C E
        row 6: E S S S S S S S S S S S E
        row 7: E W W S S S W S S S W W E
        row 8: D D S E S D D D S E S D D
        """

        json_result, svg_result, analysis, generator = compile_facade(source_code, apply_transforms=True)
        
        if json_result:
            output_dir = args.output_dir
            
            json_path = f'{output_dir}/facade_5.json'
            svg_path = f'{output_dir}/facade_5.svg'
            png_path = f'{output_dir}/facade_5.png'
            
            with open(json_path, 'w') as f:
                f.write(json_result)
            
            with open(svg_path, 'w') as f:
                f.write(svg_result)
            
            # Generate PNG from SVG
            generator.generate_png(svg_result, png_path, scale=2)
            
            print("\n" + "=" * 60)
            print("FILES SAVED:")
            print(f"  - {json_path}")
            print(f"  - {svg_path}")
            print(f"  - {png_path}")
            print("=" * 60)