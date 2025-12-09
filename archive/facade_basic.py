import json
import sys

# --- 1. FRONT-END: Lexer & Parser ---
class FacadeParser:
    def __init__(self):
        self.grid = []
    
    def parse(self, input_text):
        """
        Parses raw text into a 2D grid Intermediate Representation (IR).
        Expected format: "row X: A B C ..."
        """
        lines = input_text.strip().split('\n')
        ir_grid = []
        
        for line in lines:
            if not line.strip():
                continue
            
            # Basic Lexing: Split "row 1: E E..." into parts
            try:
                # Remove "row X:" prefix to get tokens
                parts = line.split(':')
                if len(parts) < 2:
                    continue # Skip malformed lines
                
                token_string = parts[1].strip()
                tokens = token_string.split()
                ir_grid.append(tokens)
            except Exception as e:
                print(f"Syntax Error parsing line: {line} -> {e}")
                
        self.grid = ir_grid
        return ir_grid

# --- 2. MIDDLE-END: Semantic Analysis ---
class FacadeAnalyzer:
    def __init__(self, ir_grid):
        self.grid = ir_grid
        self.errors = []
        self.warnings = []

    def analyze(self):
        self._check_grid_consistency()
        self._check_doors_ground_floor()
        self._check_symmetry()
        
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

    def _check_doors_ground_floor(self):
        # Rule: Doors (D) only allowed on the last row
        total_rows = len(self.grid)
        for r_idx, row in enumerate(self.grid):
            if r_idx == total_rows - 1:
                continue # Ground floor is allowed to have doors
            
            if 'D' in row:
                self.errors.append(f"Structural Violation: Door found on upper floor (Row {r_idx+1}).")

    def _check_symmetry(self):
        # Rule: Row should be a palindrome
        for r_idx, row in enumerate(self.grid):
            if row != row[::-1]:
                self.warnings.append(f"Asymmetry detected on Row {r_idx+1}.")

# --- 3. BACK-END: Code Generation ---
class FacadeGenerator:
    def __init__(self, ir_grid):
        self.grid = ir_grid
        # DSL to Color Mapping
        self.style_map = {
            'E': {'fill': 'white', 'stroke': '#ddd'},    # Empty
            'W': {'fill': '#89CFF0', 'stroke': '#333'},  # Window (Blue)
            'S': {'fill': '#E6E6FA', 'stroke': '#333'},  # Stone/Surface (Grey)
            'D': {'fill': '#8B4513', 'stroke': '#000'}   # Door (Brown)
        }
        self.cell_size = 50

    def generate_json(self):
        return json.dumps({
            "meta": {"rows": len(self.grid), "cols": len(self.grid[0]) if self.grid else 0},
            "layout": self.grid
        }, indent=2)

    def generate_svg(self):
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
        
        for r, row in enumerate(self.grid):
            for c, token in enumerate(row):
                x = c * self.cell_size
                y = r * self.cell_size
                style = self.style_map.get(token, {'fill': 'black', 'stroke': 'red'})
                
                rect = (
                    f'  <rect x="{x}" y="{y}" width="{self.cell_size}" height="{self.cell_size}" '
                    f'fill="{style["fill"]}" stroke="{style["stroke"]}" stroke-width="2"/>'
                )
                
                # Add text label for clarity
                text = (
                    f'  <text x="{x + self.cell_size/2}" y="{y + self.cell_size/2 + 4}" '
                    f'font-family="Arial" font-size="14" text-anchor="middle" '
                    f'dominant-baseline="middle" fill="#555">{token}</text>'
                )
                svg_lines.append(rect)
                svg_lines.append(text)
                
        svg_lines.append('</svg>')
        return "\n".join(svg_lines)

# --- 4. COMPILER DRIVER ---
def compile_facade(source_code):
    print("--- COMPILING FACADE ---")
    
    # 1. Parse
    parser = FacadeParser()
    ir = parser.parse(source_code)
    print("1. Parsing Complete.")
    
    # 2. Analyze
    analyzer = FacadeAnalyzer(ir)
    result = analyzer.analyze()
    
    if not result['is_valid']:
        print("2. Semantic Analysis Failed:")
        for err in result['errors']:
            print(f"   [ERROR] {err}")
        return None, None
    
    print("2. Semantic Analysis Passed.")
    for warn in result['warnings']:
        print(f"   [WARNING] {warn}")

    # 3. Generate
    generator = FacadeGenerator(ir)
    json_out = generator.generate_json()
    svg_out = generator.generate_svg()
    print("3. Code Generation Complete.")
    
    return json_out, svg_out

# --- INPUT DATA ---
# source_code = """
# row 1: E E E E E E E E
# row 2: E E E E E E E E
# row 3: E W S S S S W E
# row 4: E W S S S S W E
# row 5: E E E D D E E E
# """

source_code = """
row 1: E E E E E E E E
row 2: E E D D E E E E   
row 3: E W S S S S       
row 4: E W S S S S W E
row 5: E E E E E E E E
"""
# <-- ERROR: Doors on 2nd floor!
# <-- ERROR: Length is 6 (should be 8)

# --- EXECUTION ---
if __name__ == "__main__":
    result = compile_facade(source_code)
    
    if result[0] is not None:
        json_result, svg_result = result
        
        print("\n--- OUTPUT: JSON ---")
        # print(json_result)
        
        print("\n--- OUTPUT: SVG ---")
        # print(svg_result)
        
        # Write to files
        with open('/Users/it84/Desktop/GitHub/facade_layout_compiler/facade.json', 'w') as f:
            f.write(json_result)
        
        with open('/Users/it84/Desktop/GitHub/facade_layout_compiler/facade.svg', 'w') as f:
            f.write(svg_result)
        
        print("\n--- FILES SAVED ---")
        # print("JSON: facade.json")
        # print("SVG: facade.svg")