'''
Parser & SDT 
Compilers 
Team: 02
Group: 05
Students:
320206102
316255819
423031180
320117174
320340312
'''

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    # Try to import the graphviz library
    from graphviz import Digraph
except ImportError:
    # If not found, print a warning and set Digraph to None
    print("Warning: 'graphviz' library not installed. 'pip install graphviz'")
    Digraph = None  

# ============================================================================
# CLASS FOR AST NODE 
# ============================================================================

@dataclass
class ASTNode:
    """A simple node for the Abstract Syntax Tree."""
    name: str
    id: int  # Unique ID for Graphviz

    def __init__(self, name: str):
        # Set the display name of the node (e.g., "Expr", "Term")
        self.name = name
        # Initialize an empty list to hold child nodes
        self.children: List['ASTNode'] = []
        # We use id(self) for a guaranteed unique ID for Graphviz
        # This gets the memory address, which is unique for each object
        self.id = id(self)  

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class LexicalError(Exception):
    """Custom exception for lexical analyzer (lexer) errors."""
    pass

class SyntacticError(Exception):
    """Custom exception for syntactic analyzer (parsing) errors."""
    pass

class SemanticError(Exception):
    """Custom exception for semantic analyzer (SDT) errors."""
    pass

# ============================================================================
# LEXICAL ANALYZER (LEXER)
# ============================================================================

@dataclass
class Token:
    """A dataclass to represent a single token found by the lexer."""
    type: str  # e.g., 'ID', 'NUM', 'KEYWORD'
    value: str # e.g., 'x', '10', 'int'
    
    def __repr__(self):
        # Helper for printing tokens clearly
        return f"<{self.type}, '{self.value}'>"


class Lexer:
    """The lexical analyzer, responsible for turning raw code into tokens."""
    def __init__(self, code: str):
        self.code = code  # The input code string
        self.pos = 0      # The current position in the code string
        # A set of reserved keywords
        self.keywords = {
            'int', 'float', 'char', 'double', 'long', 'const',
            'if', 'else', 'while', 'for'
        }
        
    def tokenize(self) -> List[Token]:
        """Main method to process the code and return a list of tokens."""
        tokens = []
        
        # Loop until we are at the end of the code
        while self.pos < len(self.code):
            # Skip whitespace
            if self.code[self.pos].isspace():
                self.pos += 1
                continue
            
            # Check for identifiers or keywords
            if self.code[self.pos].isalpha() or self.code[self.pos] == '_':
                tokens.append(self._read_identifier())
                continue
            
            # Check for numbers
            if self.code[self.pos].isdigit():
                tokens.append(self._read_number())
                continue
            
            # Check for character literals
            if self.code[self.pos] == "'":
                tokens.append(self._read_char())
                continue
            
            # Check for two-character operators (e.g., '==', '<=')
            if self.pos < len(self.code) - 1:
                two_char = self.code[self.pos:self.pos+2]
                if two_char in ['==', '<=', '>=', '!=']:
                    tokens.append(Token('OP', two_char))
                    self.pos += 2
                    continue
            
            # Check for single-character operators
            if self.code[self.pos] in '+-*/=<>':
                tokens.append(Token('OP', self.code[self.pos]))
                self.pos += 1
                continue
            
            # Check for special characters (delimiters)
            if self.code[self.pos] in ';(){}[]':
                tokens.append(Token('SPECIAL', self.code[self.pos]))
                self.pos += 1
                continue
            
            # If no rule matched, raise a lexical error
            raise LexicalError(f"Unknown character: '{self.code[self.pos]}'")
        
        # Add an End-of-File token to signal the end
        tokens.append(Token('EOF', '$'))
        return tokens
    
    def _read_identifier(self) -> Token:
        """Helper to read an identifier or keyword."""
        start = self.pos
        # Keep reading as long as it's alphanumeric or underscore
        while self.pos < len(self.code) and (self.code[self.pos].isalnum() or self.code[self.pos] == '_'):
            self.pos += 1
        value = self.code[start:self.pos]
        # Check if the value is a reserved keyword
        token_type = 'KEYWORD' if value in self.keywords else 'ID'
        return Token(token_type, value)
    
    def _read_number(self) -> Token:
        """Helper to read an integer or float number."""
        start = self.pos
        is_float = False
        
        while self.pos < len(self.code) and (self.code[self.pos].isdigit() or self.code[self.pos] == '.'):
            if self.code[self.pos] == '.':
                if is_float:
                    break # Found a second decimal point, stop
                is_float = True
            self.pos += 1
        
        value = self.code[start:self.pos]
        token_type = 'FLOAT' if is_float else 'NUM'
        return Token(token_type, value)
    
    def _read_char(self) -> Token:
        """Helper to read a character literal (e.g., 'a')."""
        self.pos += 1 # Skip the opening '
        if self.pos >= len(self.code):
            raise LexicalError("Incomplete character literal at end of file")
        value = self.code[self.pos] # Get the character
        self.pos += 1 # Skip the character
        if self.pos >= len(self.code) or self.code[self.pos] != "'":
            raise LexicalError("Character literal must end with '")
        self.pos += 1 # Skip the closing '
        return Token('CHAR', value)


# ============================================================================
# SYMBOL TABLE
# ============================================================================

@dataclass
class Symbol:
    """A dataclass to represent an entry in the symbol table."""
    name: str
    type: str
    value: Any
    line: int
    is_const: bool = False


class SymbolTable:
    """Manages variables (symbols) and their attributes."""
    def __init__(self):
        # A dictionary to store symbols, with the name as the key
        self.symbols: Dict[str, Symbol] = {}
    
    def add(self, name: str, sym_type: str, value: Any, line: int, is_const: bool = False):
        """Add a new symbol to the table."""
        if name in self.symbols:
            # Semantic error: variable re-declaration
            raise SemanticError(f"Variable '{name}' already declared")
        self.symbols[name] = Symbol(name, sym_type, value, line, is_const)
    
    def get(self, name: str) -> Optional[Symbol]:
        """Get a symbol by name."""
        return self.symbols.get(name)
    
    def update(self, name: str, value: Any):
        """Update the value of an existing symbol."""
        if name in self.symbols:
            self.symbols[name].value = value
    
    def exists(self, name: str) -> bool:
        """Check if a symbol exists."""
        return name in self.symbols
    
    def get_all(self) -> List[Symbol]:
        """Return a list of all symbols in the table."""
        return list(self.symbols.values())


# ============================================================================
# LL(1) PARSER WITH SDT
# ============================================================================

class Parser:
    """
    A recursive descent LL(1) parser.
    It performs syntactic analysis (parsing) and semantic analysis
    (Syntax-Directed Translation or SDT) simultaneously.
    It also builds the Abstract Syntax Tree (AST).
    """
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.symbol_table = SymbolTable()
        self.output = []      # Log for parser/SDT actions
        self.tree_lines = []  # For the textual tree view
        self.indent = 0       # Indentation level for the textual tree
        
        # For the graphical AST 
        self.ast_root: Optional[ASTNode] = None  # The root node of our tree
        self.node_stack: List[ASTNode] = []      # Stack to track the current parent node
    
    def log(self, message: str):
        """Adds a message to the main output log."""
        self.output.append(message)
    
    def add_to_tree(self, message: str):
        """
        Adds a node to both text tree and the graphical AST.
        """
        # Logic for the textual tree
        self.tree_lines.append("  " * self.indent + message)
        
        # Logic for the graphical AST
        new_node = ASTNode(message)
        
        if not self.ast_root:
            # If this is the first node, it becomes the root
            self.ast_root = new_node
        else:
            # If not, it's a child of the last node in the stack (if stack isn't empty)
            if self.node_stack:
                parent = self.node_stack[-1]
                parent.children.append(new_node)
                
        # We push the new node onto the stack. It is now the "current"
        # parent for any subsequent nodes created.
        self.node_stack.append(new_node)
    
    def lookahead(self) -> Token:
        """Return the current token without consuming it."""
        return self.tokens[self.pos]
    
    def match(self, expected: str) -> Token:
        """
        Consumes the current token if it matches the expected type or value.
        If it doesn't match, raises a SyntacticError.
        """
        token = self.lookahead()
        if token.type == expected or token.value == expected:
            self.log(f"Match: {token.value} ({token.type})")
            self.pos += 1
            return token
        # If no match, it's a syntax error
        raise SyntacticError(f"Error: expected '{expected}', found '{token.value}'")
    
    def parse(self) -> Tuple[bool, bool]:
        """
        Main entry point for the parser.
        Returns (parse_success, sdt_success)
        """
        self.add_to_tree("Program") # Create the root node
        self.indent += 1
        
        try:
            # Start parsing from the 'program' grammar rule
            self.program()
            # After parsing, we must be at the EOF token
            if self.lookahead().type != 'EOF':
                raise SyntacticError("Unexpected tokens at end of file")
            
            # If we get here, both were successful
            self.log("✓ Parsing Success!")
            self.log("✓ SDT Verified!")
            return (True, True)
        
        except SyntacticError as e:
            # A syntax error occurred
            self.log(f"✗ Parsing error...: {str(e)}")
            return (False, False)
        
        except SemanticError as e:
            # Syntax was correct, but a semantic error (SDT) occurred
            self.log("✓ Parsing Success!") # Parsing itself was fine
            self.log(f"✗ SDT error...: {str(e)}")
            return (True, False) # (Parse=True, SDT=False)
        
        except Exception as e:
            # Any other unexpected error
            self.log(f"✗ Unexpected Error: {str(e)}")
            return (False, False)

    
    def program(self):
        """Grammar rule: program -> statement*"""
        while self.lookahead().type != 'EOF':
            self.statement()
        self.indent -= 1
        self.node_stack.pop() # Pop the program node
    
    def statement(self):
        """Grammar rule: statement -> declaration | assignment"""
        self.add_to_tree("Statement")
        self.indent += 1
        
        token = self.lookahead()
        
        if token.type == 'KEYWORD':
            # Starts with a keyword, must be a declaration
            self.declaration()
        elif token.type == 'ID':
            # Starts with an ID, must be an assignment
            self.assignment()
        else:
            raise SyntacticError(f"Unexpected token: {token.value}, expected a declaration or assignment")
        
        self.indent -= 1
        self.node_stack.pop() # Pop the statement node
    
    def declaration(self):
        """Grammar rule: declaration -> [const] type ID = expr ;"""
        self.add_to_tree("Declaration")
        self.indent += 1

        # Check for optional 'const'
        is_const = False
        if self.lookahead().value == 'const':
            self.match('KEYWORD')
            is_const = True

        # Match the type (e.g., 'int', 'long', 'float')
        type_parts = []
        while self.lookahead().value in ['int', 'float', 'char', 'double', 'long']:
            type_parts.append(self.match('KEYWORD').value)
        full_type = ' '.join(type_parts)

        id_token = self.match('ID') # Match the variable name
        self.match('=')             # Match the assignment operator
        value = self.expr()         # Parse the expression on the right
        self.match(';')             # Match the semicolon

        # SDT action: Add the new variable to the symbol table
        self.symbol_table.add(id_token.value, full_type, value, self.pos, is_const)
        const_flag = " (const)" if is_const else ""
        self.log(f"SDT: Declared{const_flag} '{id_token.value}' ({full_type}) = {value}")

        self.indent -= 1
        self.node_stack.pop() # Pop the declaration node
    
    def assignment(self):
        """Grammar rule: assignment -> ID = expr ;"""
        self.add_to_tree("Assignment")
        self.indent += 1

        id_token = self.match('ID') # Match the variable name
        self.match('=')             # Match the '='
        value = self.expr()         # Parse the expression
        self.match(';')             # Match the ';'

        # SDT Action: Check if variable exists
        symbol = self.symbol_table.get(id_token.value)
        if not symbol:
            raise SemanticError(f"Variable '{id_token.value}' not declared")

        # SDT Action: Check if variable is constant
        if symbol.is_const:
            raise SemanticError(f"Cannot modify constant variable '{id_token.value}'")

        # SDT Action: Update the variable's value in the table
        self.symbol_table.update(id_token.value, value)
        self.log(f"SDT: Assigned '{id_token.value}' = {value}")

        self.indent -= 1
        self.node_stack.pop() # Pop the assignment node
    
    def expr(self) -> float:
        """Grammar rule: expr -> term expr'"""
        self.add_to_tree("Expr")
        self.indent += 1
        result = self.term()         # Parse the 'term'
        result = self.expr_prime(result) # Parse the rest ('expr_prime')
        self.indent -= 1
        self.node_stack.pop() # Pop the expr node
        return result
    
    def expr_prime(self, left: float) -> float:
        """Grammar rule: expr' -> + term expr' | - term expr' | ε"""
        token = self.lookahead()
        
        if token.value == '+':
            self.add_to_tree("Expr' -> +") # Add the '+' node
            self.indent += 1 
            self.match('+')
            right = self.term()        # Get the right-hand side
            result = left + right      # SDT Action: Calculate
            self.log(f"SDT: {left} + {right} = {result}")
            result = self.expr_prime(result) # Recursive call
            self.indent -= 1 
            self.node_stack.pop() # Pop the expr -> + node
            return result
        elif token.value == '-':
            self.add_to_tree("Expr' -> -") # Add the '-' node
            self.indent += 1 
            self.match('-')
            right = self.term()
            result = left - right      # SDT Action: Calculate
            self.log(f"SDT: {left} - {right} = {result}")
            result = self.expr_prime(result) # Recursive call
            self.indent -= 1 
            self.node_stack.pop() # Pop the expr -> - node
            return result
        
        # Epsilon (ε) case: No '+' or '-'
        self.add_to_tree("Expr' -> ε")
        self.node_stack.pop() # Pop the expr -> ε node
        return left # Return the result from the left
    
    def term(self) -> float:
        """Grammar rule: term -> factor term'"""
        self.add_to_tree("Term")
        self.indent += 1
        result = self.factor()       # Parse the 'factor'
        result = self.term_prime(result) # Parse the rest ('term_prime')
        self.indent -= 1
        self.node_stack.pop() # Pop the term node
        return result
    
    def term_prime(self, left: float) -> float:
        """Grammar rule: term' -> * factor term' | / factor term' | ε"""
        token = self.lookahead()
        
        if token.value == '*':
            self.add_to_tree("Term' -> *") # Add the * node
            self.indent += 1 
            self.match('*')
            right = self.factor()      # Get the right-hand side
            result = left * right      # SDT Action: Calculate
            self.log(f"SDT: {left} * {right} = {result}")
            result = self.term_prime(result) # Recursive call
            self.indent -= 1 
            self.node_stack.pop() # Pop the Term -> * node
            return result
        elif token.value == '/':
            self.add_to_tree("Term' -> /") # Add the / node
            self.indent += 1 
            self.match('/')
            right = self.factor()
            # SDT Action: Check for division by zero
            if right == 0:
                raise SemanticError("Division by zero")
            result = left / right      # SDT Action: Calculate
            self.log(f"SDT: {left} / {right} = {result}")
            result = self.term_prime(result) # Recursive call
            self.indent -= 1 
            self.node_stack.pop() # Pop the Term -> / node
            return result
        
        # Epsilon (ε) case: No * or /
        self.add_to_tree("Term' -> ε")
        self.node_stack.pop() # Pop the Term -> ε node
        return left # Return the result from the left
    
    def factor(self) -> float:
        """
        Grammar rule: factor -> +factor | -factor | NUM | FLOAT | CHAR | ID | ( expr )
        """
        self.add_to_tree("Factor")
        self.indent += 1
        
        token = self.lookahead()
        result = 0.0 # Default value
        
        # Case: Unary '+' or '-'
        if token.value == '+' or token.value == '-':
            sign_token = self.match(token.value)
            # The unary node is a child of Factor
            self.add_to_tree(f"-> Unary {sign_token.value}") 
            self.indent += 1
            value = self.factor() # Recursively call factor
            result = value if sign_token.value == '+' else -value
            self.log(f"SDT: Unary {sign_token.value}{value} = {result}")
            self.indent -= 1 
            self.node_stack.pop() # Pop the Unary node

        # Case: Number
        elif token.type == 'NUM':
            num_token = self.match('NUM')
            result = int(num_token.value)
            self.add_to_tree(f"-> {num_token.value}") # Add leaf node
            self.node_stack.pop() # Pop for leaf node

        # Case: Float
        elif token.type == 'FLOAT':
            float_token = self.match('FLOAT')
            result = float(float_token.value)
            self.add_to_tree(f"-> {float_token.value}") # Add leaf node
            self.node_stack.pop() # Pop for leaf node
        
        # Case: Char
        elif token.type == 'CHAR':
            char_token = self.match('CHAR')
            result = ord(char_token.value) # SDT Action: Convert char to ASCII value
            self.add_to_tree(f"-> '{char_token.value}'")
            self.node_stack.pop() # Pop for leaf node
        
        # Case: ID (variable)
        elif token.type == 'ID':
            id_token = self.match('ID')
            # SDT Action: Get variable's value from symbol table
            symbol = self.symbol_table.get(id_token.value)
            if not symbol:
                raise SemanticError(f"Variable '{id_token.value}' not declared")
            result = symbol.value
            self.add_to_tree(f"-> {id_token.value} = {result}")
            self.node_stack.pop() # Pop for leaf node
        
        # Case: Parenthesized expression
        elif token.value == '(':
            self.match('(')
            self.add_to_tree("-> ( Expr )") # Node for grouping
            self.indent += 1 
            result = self.expr() # Parse the inner expression
            self.match(')')
            self.indent -= 1 
            self.node_stack.pop() # Pop for ( Expr ) node
        
        # No match
        else:
            raise SyntacticError(f"Invalid factor: {token.value}")
        
        self.indent -= 1
        self.node_stack.pop() # Pop the Factor node
        return result


# ============================================================================
# GRAPHICAL INTERFACE
# ============================================================================

class ParserGUI:
    """The main application class for the Tkinter GUI."""
    def __init__(self, root):
        self.root = root
        self.root.title("Parser & SDT - Syntactic and Semantic Analyzer")
        self.root.geometry("1200x800")
        
        # Pre-defined example code snippets
        self.examples = {
            "Simple Declaration": "int x = 10;",
            "Arithmetic Expression": "int result = 5 + 3 * 2;",
            "Multiple Variables": "int a = 10;\nfloat b = 3.14;\nint c = a + 5;",
            "With Characters": "char c = 'A';\nint val = c + 10;",
            "Complex Expression": "int value = (10 + 5) * 2 - 8 / 4;",
            "Multiple Operations": "int x = 100;\nint y = 20;\nint z = x / y + 15 * 2;",
            "Unary Operator": "int x = -10;\nint y = -x + 5;",
            "Syntactic ERROR": "int x = 10", # Missing ;
            "Semantic ERROR (SDT)": "int x = 10;\nx = y + 5;", # 'y' not declared
            "Semantic ERROR (SDT) 2": "int x = 10 / 0;", # Division by zero
        }
        
        # Build the UI
        self.setup_ui()
    
    def setup_ui(self):
        """Creates all the Tkinter widgets."""
        # Header
        header = tk.Frame(self.root, bg='#2c3e50', pady=15)
        header.pack(fill='x')
        
        tk.Label(header, text="🔍 Parser & SDT", font=('Arial', 20, 'bold'),
                 bg='#2c3e50', fg='white').pack()
        tk.Label(header, text="Syntactic and Semantic Analyzer", font=('Arial', 10),
                 bg='#2c3e50', fg='#ecf0f1').pack()
        
        # Main container (divides left and right)
        main = tk.PanedWindow(self.root, orient='horizontal')
        main.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel (Input)
        left = tk.Frame(main, bg='white')
        main.add(left, width=400)
        
        tk.Label(left, text="📝 Input Code:", font=('Arial', 12, 'bold'),
                 bg='white').pack(anchor='w', padx=10, pady=5)
        
        # Input text box
        self.input_text = scrolledtext.ScrolledText(left, height=15, width=45,
                                                      font=('Consolas', 11))
        self.input_text.pack(fill='both', expand=True, padx=10, pady=5)
        self.input_text.insert('1.0', self.examples["Simple Declaration"])
        
        # Buttons frame
        btn_frame = tk.Frame(left, bg='white')
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(btn_frame, text="▶ Analyze", command=self.analyze,
                  bg='#3498db', fg='white', font=('Arial', 11, 'bold'),
                  padx=20, pady=8).pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="📁 Open", command=self.load_file,
                  bg='#2ecc71', fg='white', font=('Arial', 10, 'bold'),
                  padx=15, pady=8).pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="🗑 Clear", command=self.clear_all,
                  bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'),
                  padx=15, pady=8).pack(side='left', padx=5)
        
        # Examples dropdown
        tk.Label(left, text="📚 Examples:", font=('Arial', 10, 'bold'),
                 bg='white').pack(anchor='w', padx=10, pady=(10, 5))
        
        self.example_var = tk.StringVar()
        example_menu = ttk.Combobox(left, textvariable=self.example_var,
                                     values=list(self.examples.keys()),
                                     state='readonly', width=42)
        example_menu.pack(padx=10, fill='x')
        example_menu.bind('<<ComboboxSelected>>', self.load_example)
        
        # Right panel (Output)
        right = tk.Frame(main, bg='white')
        main.add(right, width=600)
        
        # Status label
        self.status_label = tk.Label(right, text="", font=('Arial', 12, 'bold'),
                                       bg='white')
        self.status_label.pack(pady=10, fill='x', padx=10)
        
        # Notebook (for tabs)
        notebook = ttk.Notebook(right)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Tab 1: Output Log
        output_frame = tk.Frame(notebook)
        notebook.add(output_frame, text='📋 Output')
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=25,
                                                       font=('Consolas', 10))
        self.output_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 2: Text Tree
        tree_frame = tk.Frame(notebook)
        notebook.add(tree_frame, text='🌳 Tree (Text)')
        
        self.tree_text = scrolledtext.ScrolledText(tree_frame, height=25,
                                                     font=('Consolas', 10))
        self.tree_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 3: Symbol Table
        table_frame = tk.Frame(notebook)
        notebook.add(table_frame, text='📊 Symbols')
        
        cols = ('Type', 'Value')
        self.table_tree = ttk.Treeview(table_frame, columns=cols,
                                       show='tree headings', height=20)
        self.table_tree.heading('#0', text='Name')
        self.table_tree.heading('Type', text='Type')
        self.table_tree.heading('Value', text='Value')
        
        for col in ['#0'] + list(cols):
            self.table_tree.column(col, width=150)
        
        # Scrollbar for symbol table
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical',
                                  command=self.table_tree.yview)
        self.table_tree.configure(yscrollcommand=scrollbar.set)
        
        self.table_tree.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        scrollbar.pack(side='right', fill='y')
    
    def load_example(self, event=None):
        """Loads a selected example into the input box."""
        name = self.example_var.get()
        if name in self.examples:
            self.input_text.delete('1.0', 'end')
            self.input_text.insert('1.0', self.examples[name])
    
    def load_file(self):
        """Opens a file dialog to load code from a .txt file."""
        filename = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filename:
            try:
                with open(filename, 'r') as f:
                    self.input_text.delete('1.0', 'end')
                    self.input_text.insert('1.0', f.read())
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def clear_all(self):
        """Clears all input and output fields."""
        self.input_text.delete('1.0', 'end')
        self.output_text.delete('1.0', 'end')
        self.tree_text.delete('1.0', 'end')
        # Clear the symbol table treeview
        for item in self.table_tree.get_children():
            self.table_tree.delete(item)
        self.status_label.config(text="", bg='white', fg='black')
    
    def generate_graph_image(self, ast_root: Optional[ASTNode]):
        """Uses Graphviz to build and render the AST."""
        
        # Check if the graphviz library was imported successfully
        if not Digraph:
            messagebox.showwarning("Graphviz not found", 
                                   "The Python 'graphviz' library is not installed (pip install graphviz).")
            return

        # Check if the parser actually produced a tree
        if not ast_root:
            messagebox.showwarning("No Tree", "No AST was generated (empty tree).")
            return

        try:
            # 1. Create the graph object
            dot = Digraph(comment='Parser AST')
            dot.attr('node', shape='box', fontname='Arial', fontsize='10', style='filled', fillcolor='lightblue')
            dot.attr('edge', fontname='Arial', fontsize='8')
            dot.attr(rankdir='TB') # Top-to-Bottom layout

            # 2. Recursive function to add nodes and edges
            def _build_graph(node: ASTNode):
                # Add the current node (using its unique ID and name)
                dot.node(str(node.id), node.name)
                # Add edges to its children and call recursively
                for child in node.children:
                    dot.edge(str(node.id), str(child.id))
                    _build_graph(child) # Recurse

            # 3. Build and render
            _build_graph(ast_root) # Start the recursive build from the root
            
            # Renders the graph to 'parser_ast.png' and (view=True)
            # attempts to open the file with the default image viewer.
            dot.render('parser_ast', format='png', cleanup=True, view=True)
            
            self.output_text.insert('end', "\n\n✓ Tree image generated and opened! (parser_ast.png)\n")

        except Exception as e:
            # This error will happen if 'pip install graphviz' is done, but the Graphviz the .exe is not in the system PATH.
            messagebox.showerror("Graphviz Error", 
                                 f"Could not render the graph.\n"
                                 f"Make sure the Graphviz engine is installed and in your system PATH.\n"
                                 f"Error: {e}")
            self.output_text.insert('end', f"\n\n✗ ERROR generating graph: {e}\n")

    def analyze(self):
        """Main function called by the 'Analyze' button."""
        code = self.input_text.get('1.0', 'end-1c').strip()
        
        if not code:
            messagebox.showwarning("Warning", "Please enter code")
            return
        
        # Clear previous results
        self.clear_all()
        
        parser = None # Define parser here so it's accessible after the try block
        
        try:
            # 1. Lexer phase
            lexer = Lexer(code)
            tokens = lexer.tokenize()
            
            self.output_text.insert('end', "="*50 + "\n")
            self.output_text.insert('end', "LEXICAL ANALYSIS\n")
            self.output_text.insert('end', "="*50 + "\n\n")
            
            for i, token in enumerate(tokens):
                self.output_text.insert('end', f"{i}: {token}\n")
            
            # 2. Parser phase
            parser = Parser(tokens)
            # This is where SyntacticError or SemanticError can be raised
            parse_success, sdt_success = parser.parse()
            
            # 3. Display results
            self.output_text.insert('end', "\n" + "="*50 + "\n")
            self.output_text.insert('end', "SYNTACTIC AND SEMANTIC (SDT) ANALYSIS\n")
            self.output_text.insert('end', "="*50 + "\n\n")
            
            # Display parser log
            for line in parser.output:
                self.output_text.insert('end', line + "\n")
            
            # Display textual tree
            self.tree_text.insert('end', "\n".join(parser.tree_lines))
            
            # Display symbol table
            for symbol in parser.symbol_table.get_all():
                self.table_tree.insert('', 'end', text=symbol.name,
                                       values=(symbol.type, symbol.value))
            
            # Update status bar
            if parse_success and sdt_success:
                self.status_label.config(text="✓ Parsing Success! | SDT Verified!",
                                         bg='#2ecc71', fg='white')
            elif parse_success and not sdt_success:
                # This is the "Parsing Success! | SDT error" case
                self.status_label.config(text="✓ Parsing Success! | ✗ SDT error...",
                                         bg='#e67e22', fg='white')
            else: # not parse_success
                self.status_label.config(text="✗ Parsing error...",
                                         bg='#e74c3c', fg='white')
            
            if parse_success and parser and parser.ast_root:
                # If syntactic analysis was successful, we can generate the graph even if there was a semantic (SDT) error.
                self.generate_graph_image(parser.ast_root)
                
        except LexicalError as e: 
            # Handle errors from the lexer
            self.output_text.insert('end', f"\n\n✗ Lexical Error: {e}")
            self.status_label.config(text="✗ Lexical Error!", bg='#e74c3c', fg='white')
            
        except Exception as e:
            # Handle any other unexpected errors
            messagebox.showerror("Unexpected Error", str(e))
            self.status_label.config(text="✗ Error", bg='#e74c3c', fg='white')


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to create and run the Tkinter application."""
    root = tk.Tk()
    app = ParserGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
