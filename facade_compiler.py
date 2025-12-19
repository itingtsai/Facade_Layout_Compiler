"""
Facade Layout Compiler

Features:
- Input syntax (single line or multi-line)
- Automatic grid size detection
- Automatic missing cell filling with 'E' (empty)
- Flexible whitespace handling (spaces optional between symbols)
- Support for repeat expressions (W*5)

Grammar (EBNF):
    program     ::= rule_decl* row_decl+
    rule_decl   ::= 'rule' IDENTIFIER ':' rule_value+
    rule_value  ::= INTEGER | IDENTIFIER
    row_decl    ::= 'row' INTEGER ':' cell+
    cell        ::= SYMBOL | repeat_expr
    repeat_expr ::= SYMBOL '*' INTEGER
    SYMBOL      ::= 'E' | 'W' | 'S' | 'D' | 'C'

Example inputs:
    "row 1: E E C E row 2: W W C W row 3: E D D E"
    "row 1: EEC row 2: WWC row 3: EDD"
    "row 1: E*3 C E*3 row 2: W*7"
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Tuple, Union, Set

try:
    import cairosvg
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False


# =============================================================================
# 1. SOURCE LOCATION & ERROR HANDLING
# =============================================================================

@dataclass(frozen=True)
class SourceLocation:
    line: int
    column: int
    
    def __str__(self) -> str:
        return f"line {self.line}, col {self.column}"


@dataclass
class SourceFile:
    name: str
    content: str
    
    def __post_init__(self):
        self.lines = self.content.split('\n')
    
    def get_line(self, n: int) -> str:
        return self.lines[n - 1] if 1 <= n <= len(self.lines) else ""


class ErrorKind(Enum):
    LEXICAL = auto()
    SYNTAX = auto()
    SEMANTIC = auto()
    WARNING = auto()
    INFO = auto()


@dataclass
class Message:
    kind: ErrorKind
    text: str
    location: Optional[SourceLocation] = None
    hint: Optional[str] = None


class MessageCollector:
    def __init__(self, source: Optional[SourceFile] = None):
        self.source = source
        self.errors: List[Message] = []
        self.warnings: List[Message] = []
        self.infos: List[Message] = []
    
    def error(self, text: str, loc: Optional[SourceLocation] = None, 
              hint: str = None, kind: ErrorKind = ErrorKind.SEMANTIC):
        self.errors.append(Message(kind, text, loc, hint))
    
    def warn(self, text: str, loc: Optional[SourceLocation] = None):
        self.warnings.append(Message(ErrorKind.WARNING, text, loc))
    
    def info(self, text: str, loc: Optional[SourceLocation] = None):
        self.infos.append(Message(ErrorKind.INFO, text, loc))
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0


# =============================================================================
# 2. LEXER
# =============================================================================

class TokenType(Enum):
    ROW = auto()
    RULE = auto()
    SYMBOL = auto()
    INTEGER = auto()
    IDENTIFIER = auto()
    COLON = auto()
    MULTIPLY = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    location: SourceLocation
    raw: str
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r})"


class Lexer:
    KEYWORDS = {'row': TokenType.ROW, 'rule': TokenType.RULE}
    SYMBOLS = {'E', 'W', 'S', 'D', 'C'}
    
    def __init__(self, source: SourceFile, messages: MessageCollector):
        self.source = source
        self.messages = messages
        self.content = source.content
        self.pos = 0
        self.line = 1
        self.column = 1
    
    def loc(self) -> SourceLocation:
        return SourceLocation(self.line, self.column)
    
    def peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        return self.content[idx] if idx < len(self.content) else '\0'
    
    def advance(self) -> str:
        if self.pos >= len(self.content):
            return '\0'
        ch = self.content[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch
    
    def skip_ws(self):
        while True:
            while self.peek() in ' \t\n\r':
                self.advance()
            if self.peek() == '#':
                while self.peek() not in ('\n', '\0'):
                    self.advance()
                continue
            break
    
    def scan_number(self) -> Token:
        start = self.loc()
        start_pos = self.pos
        while self.peek().isdigit():
            self.advance()
        raw = self.content[start_pos:self.pos]
        return Token(TokenType.INTEGER, int(raw), start, raw)
    
    def scan_word(self) -> Token:
        start = self.loc()
        start_pos = self.pos
        while self.peek().isalnum() or self.peek() == '_':
            self.advance()
        raw = self.content[start_pos:self.pos]
        raw_upper = raw.upper()
        
        # Check for keywords (case-insensitive)
        if raw.lower() in self.KEYWORDS:
            return Token(self.KEYWORDS[raw.lower()], raw.lower(), start, raw)
        
        # Check if it's a single valid symbol
        if raw_upper in self.SYMBOLS and len(raw) == 1:
            return Token(TokenType.SYMBOL, raw_upper, start, raw)
        
        # Check if it's a sequence of valid symbols only
        if len(raw) > 0 and all(c in self.SYMBOLS for c in raw_upper):
            return Token(TokenType.IDENTIFIER, raw_upper, start, raw)
        
        # It's an identifier - but check if it contains invalid symbol chars
        # This catches cases like "EWX" or "window" or "X"
        return Token(TokenType.IDENTIFIER, raw, start, raw)
    
    def tokenize(self) -> List[Token]:
        tokens = []
        while self.pos < len(self.content):
            self.skip_ws()
            if self.pos >= len(self.content):
                break
            ch = self.peek()
            if ch == '\0':
                break
            
            # Numbers (but not negative - that's handled as separate tokens)
            if ch.isdigit():
                tokens.append(self.scan_number())
                continue
            
            # Words (keywords, identifiers, symbols, symbol sequences)
            if ch.isalpha() or ch == '_':
                tokens.append(self.scan_word())
                continue
            
            # Punctuation
            if ch == ':':
                tokens.append(Token(TokenType.COLON, ':', self.loc(), ':'))
                self.advance()
                continue
            
            if ch == '*':
                tokens.append(Token(TokenType.MULTIPLY, '*', self.loc(), '*'))
                self.advance()
                continue
            
            # Minus sign (for negative numbers - but we reject these)
            if ch == '-':
                loc = self.loc()
                self.advance()
                if self.peek().isdigit():
                    self.messages.error("Negative numbers not allowed", loc, kind=ErrorKind.LEXICAL)
                    # Skip the number
                    while self.peek().isdigit():
                        self.advance()
                else:
                    self.messages.error(f"Unexpected character '-'", loc, kind=ErrorKind.LEXICAL)
                continue
            
            # Unknown/invalid character
            loc = self.loc()
            self.messages.error(f"Unexpected character '{ch}'", loc, kind=ErrorKind.LEXICAL)
            self.advance()
        
        tokens.append(Token(TokenType.EOF, None, self.loc(), ''))
        return tokens


# =============================================================================
# 3. AST
# =============================================================================

@dataclass
class Cell:
    symbol: str
    location: SourceLocation


@dataclass
class RepeatExpr:
    symbol: str
    count: int
    location: SourceLocation
    
    def expand(self) -> List[str]:
        return [self.symbol] * self.count


@dataclass
class RowDecl:
    row_number: int
    cells: List[Union[Cell, RepeatExpr]]
    location: SourceLocation
    
    def get_symbols(self) -> List[str]:
        result = []
        for cell in self.cells:
            if isinstance(cell, RepeatExpr):
                result.extend(cell.expand())
            else:
                result.append(cell.symbol)
        return result


@dataclass
class RuleDecl:
    name: str
    value: str
    location: SourceLocation


@dataclass
class Program:
    rules: List[RuleDecl]
    rows: List[RowDecl]
    location: SourceLocation


# =============================================================================
# 4. PARSER
# =============================================================================

class Parser:
    SYMBOLS = {'E', 'W', 'S', 'D', 'C'}
    
    def __init__(self, tokens: List[Token], messages: MessageCollector):
        self.tokens = tokens
        self.messages = messages
        self.pos = 0
    
    def current(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else self.tokens[-1]
    
    def advance(self) -> Token:
        token = self.current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token
    
    def check(self, *types: TokenType) -> bool:
        return self.current().type in types
    
    def match(self, *types: TokenType) -> Optional[Token]:
        if self.check(*types):
            return self.advance()
        return None
    
    def expect(self, tt: TokenType, msg: str) -> Optional[Token]:
        if self.check(tt):
            return self.advance()
        self.messages.error(msg, self.current().location, kind=ErrorKind.SYNTAX)
        return None
    
    def sync_to_row(self):
        while not self.check(TokenType.EOF, TokenType.ROW):
            self.advance()
    
    def parse(self) -> Optional[Program]:
        start = self.current().location
        rules, rows = [], []
        
        while self.check(TokenType.RULE):
            if rule := self.parse_rule():
                rules.append(rule)
        
        while self.check(TokenType.ROW):
            if row := self.parse_row():
                rows.append(row)
        
        if not self.check(TokenType.EOF):
            self.messages.error(f"Unexpected: '{self.current().raw}'", self.current().location, kind=ErrorKind.SYNTAX)
        
        if not rows:
            self.messages.error("No rows found", self.current().location, hint="Example: row 1: E W S", kind=ErrorKind.SYNTAX)
        
        return Program(rules, rows, start)
    
    def parse_rule(self) -> Optional[RuleDecl]:
        start = self.advance()  # consume 'rule'
        name = self.expect(TokenType.IDENTIFIER, "Expected rule name")
        if not name:
            self.sync_to_row()
            return None
        if not self.match(TokenType.COLON):
            self.messages.error("Expected ':'", self.current().location, kind=ErrorKind.SYNTAX)
            self.sync_to_row()
            return None
        
        parts = []
        # Collect rule values - can be integers or identifiers that aren't symbol sequences
        while True:
            if self.check(TokenType.INTEGER):
                parts.append(str(self.advance().value))
            elif self.check(TokenType.IDENTIFIER):
                token = self.current()
                upper_val = token.value.upper() if isinstance(token.value, str) else str(token.value)
                # Stop if this looks like a symbol sequence (would be cells in next row)
                if all(c in self.SYMBOLS for c in upper_val):
                    break
                parts.append(str(self.advance().value))
            else:
                break
        
        if not parts:
            self.messages.error("Expected rule value", self.current().location, kind=ErrorKind.SYNTAX)
            self.sync_to_row()
            return None
        
        return RuleDecl(name.value, ' '.join(parts), start.location)
    
    def parse_row(self) -> Optional[RowDecl]:
        start = self.advance()  # consume 'row'
        num = self.expect(TokenType.INTEGER, "Expected row number")
        if not num:
            self.sync_to_row()
            return None
        if not self.match(TokenType.COLON):
            self.messages.error("Expected ':'", self.current().location, hint=f"row {num.value}: E W S", kind=ErrorKind.SYNTAX)
            self.sync_to_row()
            return None
        
        cells = self.parse_cells()
        if not cells:
            self.messages.error(f"Row {num.value} has no cells", self.current().location, kind=ErrorKind.SYNTAX)
            return None
        
        return RowDecl(num.value, cells, start.location)
    
    def parse_cells(self) -> List[Union[Cell, RepeatExpr]]:
        cells = []
        while not self.check(TokenType.ROW, TokenType.RULE, TokenType.EOF):
            if self.check(TokenType.SYMBOL):
                token = self.advance()
                if self.match(TokenType.MULTIPLY):
                    count_token = self.current()
                    if not self.check(TokenType.INTEGER):
                        self.messages.error("Expected repeat count after '*'", self.current().location, kind=ErrorKind.SYNTAX)
                        # Try to recover - treat as single cell
                        cells.append(Cell(token.value, token.location))
                        continue
                    count = self.advance()
                    n = count.value if count.value >= 0 else 0  # Allow 0 to mean "no cells"
                    # Only add if count > 0 (skip if count is 0)
                    if n > 0:
                        cells.append(RepeatExpr(token.value, n, token.location))
                    # If n == 0, we simply don't add anything (element doesn't exist)
                else:
                    cells.append(Cell(token.value, token.location))
            elif self.check(TokenType.IDENTIFIER):
                token = self.current()
                upper_val = token.value.upper() if isinstance(token.value, str) else str(token.value)
                # Check if it's a valid symbol sequence
                if all(c in self.SYMBOLS for c in upper_val):
                    self.advance()
                    for i, ch in enumerate(upper_val):
                        cells.append(Cell(ch, SourceLocation(token.location.line, token.location.column + i)))
                else:
                    # Invalid symbol - report error
                    self.messages.error(
                        f"Invalid symbol(s) in '{token.value}'. Valid symbols: E, W, S, D, C",
                        token.location,
                        kind=ErrorKind.SEMANTIC
                    )
                    self.advance()  # Skip the invalid token
            elif self.check(TokenType.INTEGER):
                # Standalone number in cell context - error
                token = self.current()
                self.messages.error(
                    f"Unexpected number '{token.value}' in cell position",
                    token.location,
                    hint="Use E*N for repeat (e.g., E*5)",
                    kind=ErrorKind.SYNTAX
                )
                self.advance()
            else:
                break
        return cells


# =============================================================================
# 5. SEMANTIC ANALYSIS
# =============================================================================

class Analyzer:
    def __init__(self, ast: Program, messages: MessageCollector):
        self.ast = ast
        self.messages = messages
        self.grid: List[List[str]] = []
        self.rules: Dict[str, str] = {}
        self.row_locs: List[SourceLocation] = []
        self.auto_filled: List[Tuple[int, int]] = []
    
    def analyze(self) -> Dict[str, Any]:
        for rule in self.ast.rules:
            self.rules[rule.name] = rule.value
        
        self._build_grid()
        self._normalize_grid()
        self._check_doors()
        self._check_chimneys()
        self._check_windows()
        self._check_symmetry()
        
        h = len(self.grid)
        w = len(self.grid[0]) if self.grid else 0
        
        return {
            'is_valid': not self.messages.has_errors(),
            'grid': self.grid,
            'grid_size': (w, h),
            'rules': self.rules,
            'auto_filled': self.auto_filled,
            'errors': [e.text for e in self.messages.errors],
            'warnings': [w.text for w in self.messages.warnings]
        }
    
    def _build_grid(self):
        if not self.ast.rows:
            return
        
        sorted_rows = sorted(self.ast.rows, key=lambda r: r.row_number)
        seen = set()
        for row in sorted_rows:
            if row.row_number in seen:
                self.messages.warn(f"Duplicate row {row.row_number}", row.location)
            seen.add(row.row_number)
        
        max_row = max(r.row_number for r in sorted_rows)
        self.grid = [[] for _ in range(max_row)]
        self.row_locs = [SourceLocation(1, 1) for _ in range(max_row)]
        
        for row in sorted_rows:
            idx = row.row_number - 1
            if 0 <= idx < len(self.grid):
                self.grid[idx] = row.get_symbols()
                self.row_locs[idx] = row.location
    
    def _normalize_grid(self):
        if not self.grid:
            return
        
        max_w = max((len(r) for r in self.grid), default=0)
        if max_w == 0:
            self.messages.error("Grid has no cells")
            return
        
        for i, row in enumerate(self.grid):
            diff = max_w - len(row)
            if diff > 0:
                self.grid[i].extend(['E'] * diff)
                self.auto_filled.append((i + 1, diff))
                self.messages.info(f"Row {i + 1}: Added {diff} empty cell(s)")
    
    def _check_doors(self):
        if not self.grid:
            return
        total = len(self.grid)
        door_cols: Dict[int, List[int]] = {}
        
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == 'D':
                    door_cols.setdefault(c, []).append(r)
        
        for col, rows in door_cols.items():
            rows.sort()
            if total - 1 not in rows:
                loc = self.row_locs[rows[-1]] if rows[-1] < len(self.row_locs) else None
                self.messages.error(f"Door at col {col+1} doesn't reach ground floor", loc)
            for i in range(len(rows) - 1):
                if rows[i + 1] - rows[i] != 1:
                    loc = self.row_locs[rows[i]] if rows[i] < len(self.row_locs) else None
                    self.messages.error(f"Door at col {col+1} has gap", loc)
    
    def _check_chimneys(self):
        if not self.grid:
            return
        chimney: Set[Tuple[int, int]] = set()
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == 'C':
                    chimney.add((r, c))
        
        if not chimney:
            return
        
        def flood(start, visited):
            comp = set()
            stack = [start]
            while stack:
                cell = stack.pop()
                if cell in visited:
                    continue
                visited.add(cell)
                comp.add(cell)
                r, c = cell
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nb = (r+dr, c+dc)
                    if nb in chimney and nb not in visited:
                        stack.append(nb)
            return comp
        
        visited = set()
        for cell in chimney:
            if cell not in visited:
                comp = flood(cell, visited)
                if 0 not in {c[0] for c in comp}:
                    cols = sorted({c[1] for c in comp})
                    self.messages.error(f"Chimney at col {cols[0]+1} doesn't reach roof")
    
    def _check_windows(self):
        if not self.grid:
            return
        min_sp = int(self.rules.get('min_window_spacing', 0))
        if min_sp > 0:
            for r, row in enumerate(self.grid):
                last = -min_sp - 1
                for c, cell in enumerate(row):
                    if cell == 'W':
                        if c - last - 1 < min_sp:
                            self.messages.warn(f"Window spacing violation row {r+1} col {c+1}")
                        last = c
    
    def _check_symmetry(self):
        for r, row in enumerate(self.grid):
            if row != row[::-1]:
                self.messages.warn(f"Row {r+1} asymmetric")


# =============================================================================
# 6. CODE GENERATION
# =============================================================================

class Generator:
    STYLES = {
        'E': {'fill': '#FFFFFF', 'stroke': '#DDDDDD'},
        'W': {'fill': '#89CFF0', 'stroke': '#333333'},
        'S': {'fill': '#E6E6FA', 'stroke': '#333333'},
        'D': {'fill': '#8B4513', 'stroke': '#000000'},
        'C': {'fill': '#DC143C', 'stroke': '#8B0000'}
    }
    
    def __init__(self, grid, grid_size, rules, auto_filled):
        self.grid = grid
        self.grid_size = grid_size
        self.rules = rules
        self.auto_filled = auto_filled
        self.cell_size = 50
    
    def json(self, analysis) -> str:
        return json.dumps({
            'meta': {'width': self.grid_size[0], 'height': self.grid_size[1], 
                     'rules': self.rules, 'auto_filled': self.auto_filled},
            'grid': self.grid,
            'validation': {'is_valid': analysis['is_valid'], 
                          'errors': analysis['errors'], 
                          'warnings': analysis['warnings']}
        }, indent=2)
    
    def svg(self) -> str:
        if not self.grid:
            return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        w = self.grid_size[0] * self.cell_size
        h = self.grid_size[1] * self.cell_size
        
        lines = [f'<?xml version="1.0"?>\n<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">',
                 '<style>text{font:14px Arial}</style>']
        
        for ct, st in self.STYLES.items():
            lines.append(f'<g id="{ct}">')
            for r, row in enumerate(self.grid):
                for c, cell in enumerate(row):
                    if cell == ct:
                        x, y = c * self.cell_size, r * self.cell_size
                        lines.append(f'<rect x="{x}" y="{y}" width="{self.cell_size}" height="{self.cell_size}" fill="{st["fill"]}" stroke="{st["stroke"]}" stroke-width="2"/>')
                        lines.append(f'<text x="{x+25}" y="{y+30}" text-anchor="middle" fill="#555">{cell}</text>')
            lines.append('</g>')
        lines.append('</svg>')
        return '\n'.join(lines)
    
    def png(self, svg_content: str, path: str, scale: int = 2):
        if not HAS_CAIROSVG:
            raise RuntimeError("cairosvg not installed")
        cairosvg.svg2png(bytestring=svg_content.encode(), write_to=path, scale=scale)


# =============================================================================
# 7. COMPILER DRIVER
# =============================================================================

class FacadeCompiler:
    def __init__(self, name: str = "<input>"):
        self.name = name
        self.ast = None
        self.messages = None
    
    def compile(self, code: str, verbose: bool = True):
        if verbose:
            print("=" * 60)
            print("FACADE LAYOUT COMPILER")
            print("=" * 60)
        
        source = SourceFile(self.name, code)
        self.messages = MessageCollector(source)
        
        # Lex
        if verbose:
            print("\n[PHASE 1: LEXICAL ANALYSIS]")
        tokens = Lexer(source, self.messages).tokenize()
        if verbose:
            print(f"  ✓ {len([t for t in tokens if t.type != TokenType.EOF])} tokens")
        
        if self.messages.has_errors():
            return None, None, {'is_valid': False, 'errors': [e.text for e in self.messages.errors], 'warnings': []}, None
        
        # Parse
        if verbose:
            print("\n[PHASE 2: PARSING]")
        self.ast = Parser(tokens, self.messages).parse()
        if verbose and self.ast:
            print(f"  ✓ {len(self.ast.rules)} rules, {len(self.ast.rows)} rows")
        
        if self.messages.has_errors():
            return None, None, {'is_valid': False, 'errors': [e.text for e in self.messages.errors], 'warnings': []}, None
        
        # Analyze
        if verbose:
            print("\n[PHASE 3: SEMANTIC ANALYSIS]")
        analysis = Analyzer(self.ast, self.messages).analyze()
        if verbose:
            w, h = analysis['grid_size']
            print(f"  ✓ Grid: {w}x{h} (auto-detected)")
            if analysis['auto_filled']:
                print(f"  ✓ Auto-filled {len(analysis['auto_filled'])} row(s)")
            print(f"  {'✓' if analysis['is_valid'] else '✗'} Validation {'PASSED' if analysis['is_valid'] else 'FAILED'}")
            for e in analysis['errors']:
                print(f"    [ERROR] {e}")
            for w in analysis['warnings']:
                print(f"    [WARN] {w}")
            for i in self.messages.infos:
                print(f"    [INFO] {i.text}")
        
        if not analysis['is_valid']:
            return None, None, analysis, None
        
        # Generate
        if verbose:
            print("\n[PHASE 4: CODE GENERATION]")
        gen = Generator(analysis['grid'], analysis['grid_size'], analysis['rules'], analysis['auto_filled'])
        if verbose:
            print("  ✓ JSON & SVG generated")
        
        return gen.json(analysis), gen.svg(), analysis, gen


# =============================================================================
# 8. TESTS & MAIN
# =============================================================================

def run_tests():
    """Comprehensive test suite with edge cases and error handling."""
    
    tests = [
        # =================================================================
        # CATEGORY 1: Basic Valid Inputs
        # =================================================================
        ("Basic: Simple row", "row 1: E E E", True, None),
        ("Basic: Multiple rows", "row 1: EEE row 2: WWW row 3: EDE", True, None),
        ("Basic: All element types", "row 1: E W S D C", True, None),
        ("Basic: Single cell", "row 1: E", True, None),
        ("Basic: Single row single cell", "row 1: W", True, None),
        
        # =================================================================
        # CATEGORY 2: Syntax Variations
        # =================================================================
        ("Syntax: Compact symbols", "row 1: EEWWSSDDCC", True, None),
        ("Syntax: Mixed compact/spaced", "row 1: EE W W SS", True, None),
        ("Syntax: Repeat expression", "row 1: E*5", True, {'grid_size': (5, 1)}),
        ("Syntax: Multiple repeats", "row 1: E*3 W*2 S*3", True, {'grid_size': (8, 1)}),
        ("Syntax: Repeat with compact", "row 1: E*2 WW E*2", True, {'grid_size': (6, 1)}),
        ("Syntax: Single line multiple rows", "row 1: EEE row 2: WWW row 3: SSS", True, None),
        ("Syntax: Lowercase symbols", "row 1: e w s d c", True, None),
        ("Syntax: Mixed case", "row 1: E w S d C", True, None),
        ("Syntax: Extra whitespace", "row   1:   E   E   E", True, None),
        ("Syntax: Newlines between rows", "row 1: EEE\nrow 2: WWW\nrow 3: SSS", True, None),
        ("Syntax: Tabs as whitespace", "row\t1:\tE\tE\tE", True, None),
        
        # =================================================================
        # CATEGORY 3: Comments
        # =================================================================
        ("Comments: End of line", "row 1: EEE # this is a comment", True, None),
        ("Comments: Full line comment", "# comment\nrow 1: EEE", True, None),
        ("Comments: Multiple comments", "# c1\nrow 1: EEE # c2\n# c3\nrow 2: WWW", True, None),
        ("Comments: Empty comment", "row 1: EEE #", True, None),
        
        # =================================================================
        # CATEGORY 4: Auto-fill Feature
        # =================================================================
        ("AutoFill: Short second row", "row 1: EEEEE row 2: WW", True, {'auto_filled': [(2, 3)]}),
        ("AutoFill: Multiple short rows", "row 1: EEEEE row 2: WW row 3: S", True, {'auto_filled': [(2, 3), (3, 4)]}),
        ("AutoFill: First row shorter", "row 1: EE row 2: WWWWW", True, {'auto_filled': [(1, 3)]}),
        ("AutoFill: All different lengths", "row 1: E row 2: WW row 3: SSS", True, {'grid_size': (3, 3)}),
        ("AutoFill: Empty row (all auto)", "row 1: EEE row 3: SSS", True, {'auto_filled': [(2, 3)]}),
        
        # =================================================================
        # CATEGORY 5: Row Number Handling
        # =================================================================
        ("RowNum: Out of order", "row 3: SSS row 1: EEE row 2: WWW", True, None),
        ("RowNum: Gaps filled", "row 1: EEE row 5: SSS", True, {'grid_size': (3, 5)}),
        ("RowNum: Duplicate (last wins)", "row 1: EEE row 1: WWW", True, None),  # Warning but passes
        ("RowNum: Large number", "row 100: EEE", True, {'grid_size': (3, 100)}),
        ("RowNum: Non-sequential", "row 2: WWW row 5: SSS row 1: EEE", True, None),
        
        # =================================================================
        # CATEGORY 6: Rules
        # =================================================================
        ("Rules: Single rule", "rule min_window_spacing: 2 row 1: E W E E W E", True, None),
        ("Rules: Multiple rules", "rule a: 1 rule b: 2 row 1: EEE", True, None),
        ("Rules: Rule with multiple values", "rule test: value1 value2 123 row 1: EEE", True, None),
        ("Rules: Numeric rule value", "rule spacing: 5 row 1: EEE", True, None),
        
        # =================================================================
        # CATEGORY 7: Structural Validation - Doors
        # =================================================================
        ("Door: Valid ground floor", "row 1: EEE row 2: EDE", True, None),
        ("Door: Valid multi-row", "row 1: EEE row 2: EDE row 3: EDE", True, None),
        ("Door: Valid double door", "row 1: EEEE row 2: EDDE", True, None),
        ("Door: Invalid not ground", "row 1: EDE row 2: EEE", False, None),
        ("Door: Invalid gap", "row 1: EDE row 2: EEE row 3: EDE", False, None),
        ("Door: Invalid floating", "row 1: EEE row 2: EDE row 3: EEE row 4: EEE", False, None),
        ("Door: Multiple columns valid", "row 1: EEEEE row 2: DEEED", True, None),
        ("Door: Multiple columns one invalid", "row 1: DEEEE row 2: EEEED", False, None),  # Left door doesn't reach ground
        
        # =================================================================
        # CATEGORY 8: Structural Validation - Chimneys
        # =================================================================
        ("Chimney: Valid from top", "row 1: ECE row 2: ECE row 3: EEE", True, None),
        ("Chimney: Valid single row", "row 1: ECE row 2: EEE", True, None),
        ("Chimney: Valid diagonal", "row 1: CEE row 2: CCE row 3: ECC row 4: EEC", True, None),
        ("Chimney: Invalid not from top", "row 1: EEE row 2: ECE row 3: EEE", False, None),
        ("Chimney: Invalid disconnected", "row 1: CEE row 2: EEE row 3: EEC", False, None),
        ("Chimney: Invalid gap vertical", "row 1: ECE row 2: EEE row 3: ECE", False, None),
        ("Chimney: Multiple valid", "row 1: CECEC row 2: EEEEE", True, None),
        ("Chimney: One valid one invalid", "row 1: CEEEE row 2: CEEEE row 3: EECEE", False, None),
        
        # =================================================================
        # CATEGORY 9: Window Spacing Rule
        # =================================================================
        ("WinSpace: Valid spacing", "rule min_window_spacing: 2 row 1: W E E W E E W", True, None),
        ("WinSpace: Invalid spacing", "rule min_window_spacing: 2 row 1: W W E E E E E", True, None),  # Warning only
        ("WinSpace: Adjacent windows no rule", "row 1: W W W W W", True, None),
        ("WinSpace: Zero spacing rule", "rule min_window_spacing: 0 row 1: WWWWW", True, None),
        
        # =================================================================
        # CATEGORY 10: Invalid Characters (Lexer Errors)
        # =================================================================
        ("BadChar: At sign", "row 1: E @ E", False, None),
        ("BadChar: Dollar sign", "row 1: E $ E", False, None),
        ("BadChar: Ampersand", "row 1: E & E", False, None),
        ("BadChar: Percent", "row 1: E % E", False, None),
        ("BadChar: Exclamation", "row 1: E ! E", False, None),
        ("BadChar: Question mark", "row 1: E ? E", False, None),
        ("BadChar: Semicolon", "row 1: E ; E", False, None),
        ("BadChar: Comma", "row 1: E , E", False, None),
        ("BadChar: Period", "row 1: E . E", False, None),
        ("BadChar: Brackets", "row 1: [E E E]", False, None),
        ("BadChar: Braces", "row 1: {E E E}", False, None),
        ("BadChar: Parentheses", "row 1: (E E E)", False, None),
        ("BadChar: Pipe", "row 1: E | E", False, None),
        ("BadChar: Backslash", "row 1: E \\ E", False, None),
        ("BadChar: Quote", "row 1: E \" E", False, None),
        ("BadChar: Single quote", "row 1: E ' E", False, None),
        ("BadChar: Unicode", "row 1: E é E", False, None),
        ("BadChar: Emoji", "row 1: E 🏠 E", False, None),
        
        # =================================================================
        # CATEGORY 11: Invalid Symbols (Not E/W/S/D/C)
        # =================================================================
        ("BadSymbol: Letter X", "row 1: E X E", False, None),
        ("BadSymbol: Letter A", "row 1: E A E", False, None),
        ("BadSymbol: Letter Z", "row 1: ZZZ", False, None),
        ("BadSymbol: Mixed valid/invalid", "row 1: EWX", False, None),
        ("BadSymbol: Number as symbol", "row 1: E 1 E", False, None),
        ("BadSymbol: Word instead of symbol", "row 1: window", False, None),
        
        # =================================================================
        # CATEGORY 12: Syntax Errors
        # =================================================================
        ("SyntaxErr: Missing colon", "row 1 E E E", False, None),
        ("SyntaxErr: Missing row number", "row : E E E", False, None),
        ("SyntaxErr: Missing row keyword", "1: E E E", False, None),
        ("SyntaxErr: Empty input", "", False, None),
        ("SyntaxErr: Only whitespace", "   \n\t  ", False, None),
        ("SyntaxErr: Only comments", "# just a comment", False, None),
        ("SyntaxErr: No cells after colon", "row 1:", False, None),
        ("SyntaxErr: Double colon", "row 1:: E E E", False, None),
        ("SyntaxErr: Negative row number", "row -1: E E E", False, None),
        ("SyntaxErr: Zero row number", "row 0: E E E", True, None),  # Parser accepts, grid handles
        ("SyntaxErr: Float row number", "row 1.5: E E E", False, None),
        ("SyntaxErr: Incomplete repeat", "row 1: E* E", False, None),
        ("SyntaxErr: Repeat with zero", "row 1: E*0 W W W", True, {'grid_size': (3, 1)}),  # E*0 means no E cells
        ("SyntaxErr: Repeat with negative", "row 1: E*-1 E", False, None),
        ("SyntaxErr: Repeat no number", "row 1: E* row 2: EEE", False, None),
        ("SyntaxErr: Rule without value", "rule test: row 1: EEE", False, None),
        ("SyntaxErr: Rule without name", "rule : value row 1: EEE", False, None),
        ("SyntaxErr: Rule without colon", "rule test value row 1: EEE", False, None),
        
        # =================================================================
        # CATEGORY 13: Edge Cases
        # =================================================================
        ("Edge: Very long row", "row 1: " + "E " * 100, True, {'grid_size': (100, 1)}),
        ("Edge: Many rows", "\n".join([f"row {i}: EEE" for i in range(1, 51)]), True, {'grid_size': (3, 50)}),
        ("Edge: Large repeat", "row 1: E*1000", True, {'grid_size': (1000, 1)}),
        ("Edge: All doors (valid single row)", "row 1: DDD", True, None),  # Single row doors at ground are valid
        ("Edge: All chimneys (valid)", "row 1: CCC", True, None),
        ("Edge: All windows", "row 1: WWW", True, None),
        ("Edge: All stone", "row 1: SSS", True, None),
        ("Edge: All empty", "row 1: EEE row 2: EEE row 3: EEE", True, None),
        ("Edge: Alternating pattern", "row 1: EWEWEW row 2: WEWEW", True, None),
        ("Edge: Checkerboard", "row 1: EWEW row 2: WEWE row 3: EWEW row 4: WEWE", True, None),
        
        # =================================================================
        # CATEGORY 14: Complex Valid Facades
        # =================================================================
        ("Complex: House with chimney and door", 
         "row 1: EECEE row 2: EWCWE row 3: EWCWE row 4: EEDEE", True, None),
        ("Complex: Symmetric building",
         "row 1: EEECEEE row 2: EWWCWWE row 3: EWWCWWE row 4: EEEDDEEE", True, None),
        ("Complex: Multi-door building",
         "row 1: EEEEEEE row 2: EWWEWWE row 3: EDEEDEE", True, None),
        ("Complex: Tall building",
         "row 1: ECE row 2: ECE row 3: EWE row 4: EWE row 5: EWE row 6: EDE", True, None),
        
        # =================================================================
        # CATEGORY 15: Boundary Conditions
        # =================================================================
        ("Boundary: Row number 0", "row 0: EEE row 1: WWW", True, None),
        ("Boundary: Single char repeat", "row 1: E*1", True, {'grid_size': (1, 1)}),
        ("Boundary: Max practical repeat", "row 1: E*9999", True, None),
        ("Boundary: Unicode in comment", "row 1: EEE # 你好世界 🏠", True, None),
        
        # =================================================================
        # CATEGORY 16: Whitespace Edge Cases
        # =================================================================
        ("WS: Leading whitespace", "   row 1: EEE", True, None),
        ("WS: Trailing whitespace", "row 1: EEE   ", True, None),
        ("WS: Multiple blank lines", "\n\n\nrow 1: EEE\n\n\n", True, None),
        ("WS: Carriage return", "row 1: EEE\r\nrow 2: WWW", True, None),
        ("WS: Mixed line endings", "row 1: EEE\nrow 2: WWW\r\nrow 3: SSS", True, None),
        
        # =================================================================
        # CATEGORY 17: More Invalid Symbol Tests
        # =================================================================
        ("BadSymbol2: Lowercase invalid", "row 1: E x E", False, None),
        ("BadSymbol2: Mixed with valid", "row 1: EWFWS", False, None),
        ("BadSymbol2: Only invalid", "row 1: XYZ", False, None),
        ("BadSymbol2: Invalid in compact", "row 1: EAWBE", False, None),
        ("BadSymbol2: Reserved word", "row 1: E row E", False, None),  # 'row' in middle
        ("BadSymbol2: Rule as symbol", "row 1: E rule E", False, None),  # 'rule' in middle
        
        # =================================================================
        # CATEGORY 18: Stress Tests
        # =================================================================
        ("Stress: 100 columns", "row 1: " + "E " * 100, True, {'grid_size': (100, 1)}),
        ("Stress: 100 rows compact", " ".join([f"row {i}: EEE" for i in range(1, 101)]), True, {'grid_size': (3, 100)}),
        ("Stress: Large repeat value", "row 1: E*500", True, {'grid_size': (500, 1)}),
        ("Stress: Many repeats", "row 1: " + " ".join(["E*10"] * 10), True, {'grid_size': (100, 1)}),
        ("Stress: Deep nesting pattern", "row 1: E*10 W*10 S*10 D*10 C*10", True, {'grid_size': (50, 1)}),
        
        # =================================================================
        # CATEGORY 19: Recovery Tests
        # =================================================================
        ("Recovery: Error then valid row", "row 1: E @ E row 2: EEE", False, None),
        ("Recovery: Multiple errors", "row 1: @ $ % row 2: EEE", False, None),
        ("Recovery: Invalid then valid symbol", "row 1: X E E", False, None),
        
        # =================================================================
        # CATEGORY 20: Combination Tests
        # =================================================================
        ("Combo: Rule + compact + repeat", "rule x: 1 row 1: EE W*3 SS", True, None),
        ("Combo: Multi-rule + auto-fill", "rule a: 1 rule b: 2 row 1: EEEEE row 2: WW", True, None),
        ("Combo: Comments everywhere", "# c1\nrule x: 1 # c2\nrow 1: EEE # c3\n# c4", True, None),
        ("Combo: All features", "rule spacing: 2 row 1: E*2 C E*2 row 2: W W C W W row 3: E D D D E", True, None),
        
        # =================================================================
        # CATEGORY 21: Zero Count Repeat (E*0 means no cells)
        # =================================================================
        ("ZeroRepeat: E*0 at start", "row 1: E*0 W W W", True, {'grid_size': (3, 1)}),
        ("ZeroRepeat: C*0 at start", "row 1: C*0 E E E", True, {'grid_size': (3, 1)}),
        ("ZeroRepeat: Multiple zeros", "row 1: E*0 C*0 W*3", True, {'grid_size': (3, 1)}),
        ("ZeroRepeat: Zero in middle", "row 1: E E*0 E E", True, {'grid_size': (3, 1)}),
        ("ZeroRepeat: Zero at end", "row 1: W W W E*0", True, {'grid_size': (3, 1)}),
        ("ZeroRepeat: All zeros except one", "row 1: E*0 W*0 S*0 D*0 C*1", True, {'grid_size': (1, 1)}),
        ("ZeroRepeat: Mixed zero and non-zero", "row 1: E*0 C*1 E*4 C*1 E*4 C*1 E*1", True, None),  # 11 cells total
        ("ZeroRepeat: Auto-fill with zeros", "row 1: E*0 W*3 row 2: E*5", True, {'grid_size': (5, 2), 'auto_filled': [(1, 2)]}),
        ("ZeroRepeat: Two rows with zeros", 
         "row 1: E*0 E*5 W*0 E*5 E*1 row 2: E*0 W*1 S*1 W*3 E*1 W*3 S*1 W*1 E*1", 
         True, None),  # Both rows have 11 cells after expansion, no invalid chimneys
    ]
    
    print("\n" + "=" * 70)
    print("TEST SUITE")
    print("=" * 70)
    
    # Group tests by category
    categories = {}
    for test in tests:
        cat = test[0].split(":")[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(test)
    
    total_passed = total_failed = 0
    failed_tests = []
    
    for cat_name, cat_tests in categories.items():
        print(f"\n[{cat_name}]")
        cat_passed = cat_failed = 0
        
        for name, code, should_pass, expected in cat_tests:
            _, _, result, _ = FacadeCompiler(f"<{name}>").compile(code, verbose=False)
            
            # Check basic pass/fail
            ok = result['is_valid'] == should_pass
            
            # Check expected values if provided
            if ok and expected:
                for key, val in expected.items():
                    if key in result:
                        if result[key] != val:
                            ok = False
                            break
            
            if ok:
                print(f"  ✓ {name}")
                cat_passed += 1
            else:
                print(f"  ✗ {name}")
                print(f"      Expected: {'pass' if should_pass else 'fail'}, Got: {'pass' if result['is_valid'] else 'fail'}")
                if result['errors']:
                    print(f"      Errors: {result['errors'][:2]}")  # Show first 2 errors
                cat_failed += 1
                failed_tests.append(name)
        
        total_passed += cat_passed
        total_failed += cat_failed
        print(f"  [{cat_passed}/{cat_passed + cat_failed} passed]")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"TOTAL: {total_passed}/{total_passed + total_failed} tests passed")
    if failed_tests:
        print(f"\nFailed tests:")
        for t in failed_tests:
            print(f"  - {t}")
    print("=" * 70)
    
    return total_passed, total_failed


def main():
    import argparse
    p = argparse.ArgumentParser(description='Facade DSL Compiler')
    p.add_argument('--test', action='store_true')
    p.add_argument('--code', type=str)
    p.add_argument('--input', type=str)
    p.add_argument('--output-dir', type=str, default='.')
    args = p.parse_args()
    
    if args.test:
        run_tests()
        return
    
    # facade_1: standard, with duplicated row warnings (only the latest duplicated row would show)
    code = """ 
    row 1: E E E E E E E E C E E 
    row 2: S S S S S S S S C S S 
    row 3: S S W W S S S W W S S 
    row 4: S S W W S S S W W S S 
    row 5: S S S S S S S S S S S 
    row 6: E 
    row 6: S S D D S S W W W W S
    row 7: S S D D S S W W W W S 
    """

    # # facade_2: repeat expressions
    # code = """
    # row 1: E W S E*2 W W E S W E
    # row 2: W E W S E E S W E*3
    # row 3: E S E W*2 S E E W S E
    # row 4: S W S S E W E S E W E
    # row 5: E E W E S*2 W E E S W
    # row 6: W S E E W E S W S E E
    # row 7: D*2 E D E E D E D*2 E
    # """

    # # facade_3: auto missing cell filling
    # code = """
    # row 1: E C E S E S E C E
    # row 2: E C E W E W E C 
    # row 3: E C E W E W E C 
    # row 4: E C E W W W E C E
    # row 5: E C E W E W E C E
    # row 6: E C E W E W E C 
    # row 7: E C E S S S E C E
    # row 8: E E E S S S 
    # row 9: E D D E E E D D E
    # """

    # # facade_4: repeat expressions & auto missing cell filling
    # code = """
    # row 1: E E C E*2 
    # row 2: W W C C W W W W W W W
    # row 3: W S S C*2 S S S W S W
    # row 4: W S S S C C S W*2 S W
    # row 5: E S*4 C C S S S 
    # row 6: E D D S*3 C S D D 
    # """

    # # facade_5: element*0 & case-insensitive
    # code = """
    # row 1: E*1 C*1 E*4 C*1 E*4 c*1 e*1
    # row 2: E*1 C*1 S*1 W*3 C*1 W*3 S*1 c*1 E*1
    # row 3: E*1 C*1 S*1 W*1 S*1 W*1 C*1 W*1 S*1 W*1 S*1 C*1 E*1
    # row 4: E*1 C*1 S*1 W*1 S*1 W*1 C*1 W*1 S*1 W*1 S*1 C*1 e*1
    # row 5: E*1 C*1 S*1 W*3 C*1 W*3 S*1 C*1 E*1
    # row 6: e*0 E*1 S*11 E*1
    # row 7: E*0 E*1 W*2 S*3 W*1 S*3 W*2 E*1
    # row 8: E*0 D*2 S*1 E*1 S*1 D*3 S*1 e*1 s*1 d*2
    # """
    
    if args.input:
        with open(args.input) as f:
            code = f.read()
    
    json_out, svg_out, analysis, gen = FacadeCompiler().compile(code)
    
    if json_out:
        d = args.output_dir
        with open(f'{d}/facade_1.json', 'w') as f:
            f.write(json_out)
        with open(f'{d}/facade_1.svg', 'w') as f:
            f.write(svg_out)
        if HAS_CAIROSVG:
            gen.png(svg_out, f'{d}/facade_1.png')
        print(f"\n✓ Output saved to {d}/")


if __name__ == "__main__":
    main()
