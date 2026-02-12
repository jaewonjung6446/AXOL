//! AXOL DSL Lexer — tokenizes .axol source files.
//!
//! Syntax example:
//! ```axol
//! declare "mood_classifier" {
//!     input text_vec(128)
//!     input context(64)
//!     relate sentiment <- text_vec, context via <~>
//!     output sentiment
//!     quality omega=0.9 phi=0.8
//! }
//!
//! weave mood_classifier quantum=true seed=42
//! observe mood_classifier { text_vec = [0.1, 0.2, ...] }
//! ```

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    // Keywords
    Declare,
    Input,
    Output,
    Relate,
    Via,
    Quality,
    Weave,
    Observe,
    Reobserve,
    Quantum,
    Seed,
    Count,

    // Compose keywords
    Compose,
    Gate,
    Confident,
    Iterate,
    Converge,
    Then,
    Else,
    Design,
    Learn,

    // Literals
    Ident(String),
    StringLit(String),
    Number(f64),
    Int(i64),

    // Symbols
    LBrace,     // {
    RBrace,     // }
    LParen,     // (
    RParen,     // )
    LBracket,   // [
    RBracket,   // ]
    Arrow,      // <-
    Comma,      // ,
    Equals,     // =
    Relation(String), // <~>, <+>, <*>, <!>, <?>

    // Control
    Newline,
    Eof,
}

pub struct Lexer {
    input: Vec<char>,
    pos: usize,
    pub tokens: Vec<Token>,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self {
            input: source.chars().collect(),
            pos: 0,
            tokens: Vec::new(),
        }
    }

    pub fn tokenize(&mut self) -> &[Token] {
        while self.pos < self.input.len() {
            self.skip_whitespace();
            if self.pos >= self.input.len() { break; }

            let ch = self.input[self.pos];

            match ch {
                '#' => { self.skip_line(); }
                '\n' | '\r' => {
                    self.tokens.push(Token::Newline);
                    self.pos += 1;
                    if ch == '\r' && self.peek() == Some('\n') { self.pos += 1; }
                }
                '{' => { self.tokens.push(Token::LBrace); self.pos += 1; }
                '}' => { self.tokens.push(Token::RBrace); self.pos += 1; }
                '(' => { self.tokens.push(Token::LParen); self.pos += 1; }
                ')' => { self.tokens.push(Token::RParen); self.pos += 1; }
                '[' => { self.tokens.push(Token::LBracket); self.pos += 1; }
                ']' => { self.tokens.push(Token::RBracket); self.pos += 1; }
                ',' => { self.tokens.push(Token::Comma); self.pos += 1; }
                '=' => { self.tokens.push(Token::Equals); self.pos += 1; }
                '<' => { self.read_angle(); }
                '"' => { self.read_string(); }
                _ if ch.is_ascii_digit() || (ch == '-' && self.peek_next_digit()) => {
                    self.read_number();
                }
                _ if ch.is_ascii_alphabetic() || ch == '_' => {
                    self.read_ident();
                }
                _ => { self.pos += 1; } // skip unknown
            }
        }
        self.tokens.push(Token::Eof);
        &self.tokens
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() && self.input[self.pos] == ' ' || self.pos < self.input.len() && self.input[self.pos] == '\t' {
            self.pos += 1;
        }
    }

    fn skip_line(&mut self) {
        while self.pos < self.input.len() && self.input[self.pos] != '\n' {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn peek_next_digit(&self) -> bool {
        self.input.get(self.pos + 1).map_or(false, |c| c.is_ascii_digit())
    }

    fn read_angle(&mut self) {
        // Could be <-, <~>, <+>, <*>, <!>, <?>
        let start = self.pos;
        self.pos += 1; // skip '<'

        if self.pos < self.input.len() && self.input[self.pos] == '-' {
            self.pos += 1;
            self.tokens.push(Token::Arrow);
            return;
        }

        // Read operator like ~>, +>, *>, !>, ?>
        let mut op = String::from("<");
        while self.pos < self.input.len() && self.input[self.pos] != '>' {
            op.push(self.input[self.pos]);
            self.pos += 1;
        }
        if self.pos < self.input.len() {
            op.push('>');
            self.pos += 1;
        }
        self.tokens.push(Token::Relation(op));
    }

    fn read_string(&mut self) {
        self.pos += 1; // skip opening "
        let mut s = String::new();
        while self.pos < self.input.len() && self.input[self.pos] != '"' {
            s.push(self.input[self.pos]);
            self.pos += 1;
        }
        if self.pos < self.input.len() { self.pos += 1; } // skip closing "
        self.tokens.push(Token::StringLit(s));
    }

    fn read_number(&mut self) {
        let start = self.pos;
        if self.input[self.pos] == '-' { self.pos += 1; }
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        let mut is_float = false;
        if self.pos < self.input.len() && self.input[self.pos] == '.' {
            is_float = true;
            self.pos += 1;
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let text: String = self.input[start..self.pos].iter().collect();
        if is_float {
            self.tokens.push(Token::Number(text.parse().unwrap_or(0.0)));
        } else {
            self.tokens.push(Token::Int(text.parse().unwrap_or(0)));
        }
    }

    fn read_ident(&mut self) {
        let start = self.pos;
        while self.pos < self.input.len() && (self.input[self.pos].is_ascii_alphanumeric() || self.input[self.pos] == '_') {
            self.pos += 1;
        }
        let word: String = self.input[start..self.pos].iter().collect();
        let token = match word.as_str() {
            "declare" => Token::Declare,
            "input" => Token::Input,
            "output" => Token::Output,
            "relate" => Token::Relate,
            "via" => Token::Via,
            "quality" => Token::Quality,
            "weave" => Token::Weave,
            "observe" => Token::Observe,
            "reobserve" => Token::Reobserve,
            "quantum" => Token::Quantum,
            "seed" => Token::Seed,
            "count" => Token::Count,
            "compose" => Token::Compose,
            "gate" => Token::Gate,
            "confident" => Token::Confident,
            "iterate" => Token::Iterate,
            "converge" => Token::Converge,
            "then" => Token::Then,
            "else" => Token::Else,
            "design" => Token::Design,
            "learn" => Token::Learn,
            "true" => Token::Int(1),
            "false" => Token::Int(0),
            _ => Token::Ident(word),
        };
        self.tokens.push(token);
    }
}
