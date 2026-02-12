//! AXOL DSL Parser — builds AST from tokens.

use crate::dsl::lexer::Token;
use crate::errors::{AxolError, Result};

// ---------------------------------------------------------------------------
// AST nodes
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Statement {
    Declare(DeclareBlock),
    Weave(WeaveCmd),
    Observe(ObserveCmd),
    Reobserve(ReobserveCmd),
    // Compose layer
    ComposeChain(ComposeCmd),
    GateOp(GateCmd),
    ConfidentObs(ConfidentCmd),
    IterateObs(IterateCmd),
    DesignBasins(DesignCmd),
    Learn(LearnCmd),
}

#[derive(Clone, Debug)]
pub struct DeclareBlock {
    pub name: String,
    pub inputs: Vec<InputDecl>,
    pub outputs: Vec<String>,
    pub relations: Vec<RelateDecl>,
    pub quality: Option<QualityDecl>,
}

#[derive(Clone, Debug)]
pub struct InputDecl {
    pub name: String,
    pub dim: usize,
}

#[derive(Clone, Debug)]
pub struct RelateDecl {
    pub target: String,
    pub sources: Vec<String>,
    pub relation: String,
}

#[derive(Clone, Debug)]
pub struct QualityDecl {
    pub omega: f64,
    pub phi: f64,
}

#[derive(Clone, Debug)]
pub struct WeaveCmd {
    pub name: String,
    pub quantum: bool,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct ObserveCmd {
    pub name: String,
    pub inputs: Vec<(String, Vec<f64>)>,
}

#[derive(Clone, Debug)]
pub struct ReobserveCmd {
    pub name: String,
    pub inputs: Vec<(String, Vec<f64>)>,
    pub count: usize,
}

// --- Compose layer AST nodes ---

/// compose "pipeline" stages=[encoder, decoder]
#[derive(Clone, Debug)]
pub struct ComposeCmd {
    pub name: String,
    pub stages: Vec<String>,
}

/// gate not { x = [0.1, 0.9] }
/// gate and { a = [0.1, 0.9], b = [0.9, 0.1] }
#[derive(Clone, Debug)]
pub struct GateCmd {
    pub gate_type: String,  // "not", "and", "or"
    pub inputs: Vec<(String, Vec<f64>)>,
}

/// confident my_tapestry max=100 threshold=0.95 { x = [0.8, 0.2] }
#[derive(Clone, Debug)]
pub struct ConfidentCmd {
    pub name: String,
    pub max_observations: usize,
    pub threshold: f64,
    pub inputs: Vec<(String, Vec<f64>)>,
}

/// iterate my_tapestry max=50 converge=prob_delta value=0.001 { x = [0.8, 0.2] }
#[derive(Clone, Debug)]
pub struct IterateCmd {
    pub name: String,
    pub max_iterations: usize,
    pub converge_type: String,
    pub converge_value: f64,
    pub inputs: Vec<(String, Vec<f64>)>,
}

/// design "binary" { dim 2, basins 2, sizes [0.5, 0.5] }
#[derive(Clone, Debug)]
pub struct DesignCmd {
    pub name: String,
    pub dim: usize,
    pub n_basins: usize,
    pub sizes: Vec<f64>,
}

/// learn "xor" dim=4 quantum=1 { [0.9, 0.1, 0.9, 0.1] = 0, ... }
#[derive(Clone, Debug)]
pub struct LearnCmd {
    pub name: String,
    pub dim: usize,
    pub quantum: bool,
    pub seed: u64,
    pub samples: Vec<(Vec<f64>, usize)>,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub statements: Vec<Statement>,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    pub fn parse(&mut self) -> Result<Program> {
        let mut statements = Vec::new();

        while !self.at_end() {
            self.skip_newlines();
            if self.at_end() { break; }

            match self.current() {
                Token::Declare => statements.push(self.parse_declare()?),
                Token::Weave => statements.push(self.parse_weave()?),
                Token::Observe => statements.push(self.parse_observe()?),
                Token::Reobserve => statements.push(self.parse_reobserve()?),
                Token::Compose => statements.push(self.parse_compose()?),
                Token::Gate => statements.push(self.parse_gate()?),
                Token::Confident => statements.push(self.parse_confident()?),
                Token::Iterate => statements.push(self.parse_iterate()?),
                Token::Design => statements.push(self.parse_design()?),
                Token::Learn => statements.push(self.parse_learn()?),
                _ => { self.advance(); } // skip unknown
            }
        }

        Ok(Program { statements })
    }

    // --- Declare ---
    fn parse_declare(&mut self) -> Result<Statement> {
        self.expect_token(&Token::Declare)?;
        let name = self.expect_string_or_ident()?;
        self.skip_newlines();
        self.expect_token(&Token::LBrace)?;

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut relations = Vec::new();
        let mut quality = None;

        loop {
            self.skip_newlines();
            if self.check(&Token::RBrace) { self.advance(); break; }
            if self.at_end() { return Err(AxolError::Parse("Unexpected end in declare block".into())); }

            match self.current() {
                Token::Input => {
                    self.advance();
                    let name = self.expect_ident()?;
                    self.expect_token(&Token::LParen)?;
                    let dim = self.expect_int()? as usize;
                    self.expect_token(&Token::RParen)?;
                    inputs.push(InputDecl { name, dim });
                }
                Token::Output => {
                    self.advance();
                    let name = self.expect_ident()?;
                    outputs.push(name);
                }
                Token::Relate => {
                    self.advance();
                    let target = self.expect_ident()?;
                    self.expect_token(&Token::Arrow)?;
                    let mut sources = vec![self.expect_ident()?];
                    while self.check(&Token::Comma) {
                        self.advance();
                        sources.push(self.expect_ident()?);
                    }
                    self.expect_token(&Token::Via)?;
                    let relation = self.expect_relation()?;
                    relations.push(RelateDecl { target, sources, relation });
                }
                Token::Quality => {
                    self.advance();
                    let mut omega = 0.9;
                    let mut phi = 0.7;
                    // Parse key=value pairs
                    loop {
                        if let Token::Ident(ref k) = self.current().clone() {
                            let key = k.clone();
                            self.advance();
                            self.expect_token(&Token::Equals)?;
                            let val = self.expect_number()?;
                            match key.as_str() {
                                "omega" => omega = val,
                                "phi" => phi = val,
                                _ => {}
                            }
                        } else {
                            break;
                        }
                    }
                    quality = Some(QualityDecl { omega, phi });
                }
                _ => { self.advance(); }
            }
        }

        Ok(Statement::Declare(DeclareBlock { name, inputs, outputs, relations, quality }))
    }

    // --- Weave ---
    fn parse_weave(&mut self) -> Result<Statement> {
        self.expect_token(&Token::Weave)?;
        let name = self.expect_ident()?;
        let mut quantum = false;
        let mut seed = 42u64;

        // Parse optional key=value pairs on same line
        loop {
            match self.current() {
                Token::Quantum => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    quantum = self.expect_int()? != 0;
                }
                Token::Seed => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    seed = self.expect_int()? as u64;
                }
                Token::Ident(ref k) if k == "quantum" => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    quantum = self.expect_int()? != 0;
                }
                Token::Ident(ref k) if k == "seed" => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    seed = self.expect_int()? as u64;
                }
                Token::Newline | Token::Eof => break,
                _ => { self.advance(); }
            }
        }

        Ok(Statement::Weave(WeaveCmd { name, quantum, seed }))
    }

    // --- Observe ---
    fn parse_observe(&mut self) -> Result<Statement> {
        self.expect_token(&Token::Observe)?;
        let name = self.expect_ident()?;
        let inputs = self.parse_input_block()?;
        Ok(Statement::Observe(ObserveCmd { name, inputs }))
    }

    // --- Reobserve ---
    fn parse_reobserve(&mut self) -> Result<Statement> {
        self.expect_token(&Token::Reobserve)?;
        let name = self.expect_ident()?;
        let mut count = 10usize;

        // Check for count= before input block
        if let Token::Ident(ref k) = self.current().clone() {
            if k == "count" {
                self.advance();
                self.expect_token(&Token::Equals)?;
                count = self.expect_int()? as usize;
            }
        }
        if self.check(&Token::Count) {
            self.advance();
            self.expect_token(&Token::Equals)?;
            count = self.expect_int()? as usize;
        }

        let inputs = self.parse_input_block()?;
        Ok(Statement::Reobserve(ReobserveCmd { name, inputs, count }))
    }

    // --- Compose ---
    // compose "pipeline" stages=[encoder, decoder]
    fn parse_compose(&mut self) -> Result<Statement> {
        self.expect_token(&Token::Compose)?;
        let name = self.expect_string_or_ident()?;

        let mut stages = Vec::new();
        // Parse stages=[name1, name2, ...]
        loop {
            match self.current() {
                Token::Ident(ref k) if k == "stages" => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    self.expect_token(&Token::LBracket)?;
                    loop {
                        if self.check(&Token::RBracket) { self.advance(); break; }
                        stages.push(self.expect_ident()?);
                        if self.check(&Token::Comma) { self.advance(); }
                    }
                }
                Token::Newline | Token::Eof => break,
                _ => { self.advance(); }
            }
        }

        Ok(Statement::ComposeChain(ComposeCmd { name, stages }))
    }

    // --- Gate ---
    // gate not { x = [0.1, 0.9] }
    fn parse_gate(&mut self) -> Result<Statement> {
        self.expect_token(&Token::Gate)?;
        let gate_type = self.expect_ident()?;
        let inputs = self.parse_input_block()?;
        Ok(Statement::GateOp(GateCmd { gate_type, inputs }))
    }

    // --- Confident ---
    // confident my_tapestry max=100 threshold=0.95 { x = [0.8, 0.2] }
    fn parse_confident(&mut self) -> Result<Statement> {
        self.expect_token(&Token::Confident)?;
        let name = self.expect_ident()?;
        let mut max_observations = 100usize;
        let mut threshold = 0.95;

        loop {
            match self.current() {
                Token::Ident(ref k) if k == "max" => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    max_observations = self.expect_int()? as usize;
                }
                Token::Ident(ref k) if k == "threshold" => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    threshold = self.expect_number()?;
                }
                Token::LBrace | Token::Newline | Token::Eof => break,
                _ => { self.advance(); }
            }
        }

        let inputs = self.parse_input_block()?;
        Ok(Statement::ConfidentObs(ConfidentCmd { name, max_observations, threshold, inputs }))
    }

    // --- Iterate ---
    // iterate my_tapestry max=50 converge=prob_delta value=0.001 { x = [0.8, 0.2] }
    fn parse_iterate(&mut self) -> Result<Statement> {
        self.expect_token(&Token::Iterate)?;
        let name = self.expect_ident()?;
        let mut max_iterations = 50usize;
        let mut converge_type = "prob_delta".to_string();
        let mut converge_value = 0.001;

        loop {
            match self.current() {
                Token::Ident(ref k) if k == "max" => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    max_iterations = self.expect_int()? as usize;
                }
                Token::Converge => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    converge_type = self.expect_ident()?;
                }
                Token::Ident(ref k) if k == "converge" => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    converge_type = self.expect_ident()?;
                }
                Token::Ident(ref k) if k == "value" => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    converge_value = self.expect_number()?;
                }
                Token::LBrace | Token::Newline | Token::Eof => break,
                _ => { self.advance(); }
            }
        }

        let inputs = self.parse_input_block()?;
        Ok(Statement::IterateObs(IterateCmd {
            name, max_iterations, converge_type, converge_value, inputs,
        }))
    }

    // --- Design ---
    // design "binary" { dim 2, basins 2, sizes [0.5, 0.5] }
    fn parse_design(&mut self) -> Result<Statement> {
        self.expect_token(&Token::Design)?;
        let name = self.expect_string_or_ident()?;
        self.skip_newlines();

        let mut dim = 2usize;
        let mut n_basins = 2usize;
        let mut sizes = Vec::new();

        if self.check(&Token::LBrace) {
            self.advance();
            loop {
                self.skip_newlines();
                if self.check(&Token::RBrace) { self.advance(); break; }
                if self.at_end() { break; }

                match self.current() {
                    Token::Ident(ref k) if k == "dim" => {
                        self.advance();
                        dim = self.expect_int()? as usize;
                    }
                    Token::Ident(ref k) if k == "basins" => {
                        self.advance();
                        n_basins = self.expect_int()? as usize;
                    }
                    Token::Ident(ref k) if k == "sizes" => {
                        self.advance();
                        self.expect_token(&Token::LBracket)?;
                        loop {
                            if self.check(&Token::RBracket) { self.advance(); break; }
                            sizes.push(self.expect_number()?);
                            if self.check(&Token::Comma) { self.advance(); }
                        }
                    }
                    _ => { self.advance(); }
                }
                if self.check(&Token::Comma) { self.advance(); }
            }
        }

        Ok(Statement::DesignBasins(DesignCmd { name, dim, n_basins, sizes }))
    }

    // --- Learn ---
    // learn "xor" dim=4 quantum=1 { [0.9, 0.1, ...] = 0, ... }
    fn parse_learn(&mut self) -> Result<Statement> {
        self.expect_token(&Token::Learn)?;
        let name = self.expect_string_or_ident()?;

        let mut dim = 4usize;
        let mut quantum = true;
        let mut seed = 42u64;

        loop {
            match self.current() {
                Token::Ident(ref k) if k == "dim" => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    dim = self.expect_int()? as usize;
                }
                Token::Quantum => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    quantum = self.expect_int()? != 0;
                }
                Token::Ident(ref k) if k == "quantum" => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    quantum = self.expect_int()? != 0;
                }
                Token::Seed => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    seed = self.expect_int()? as u64;
                }
                Token::Ident(ref k) if k == "seed" => {
                    self.advance();
                    self.expect_token(&Token::Equals)?;
                    seed = self.expect_int()? as u64;
                }
                Token::LBrace | Token::Newline | Token::Eof => break,
                _ => { self.advance(); }
            }
        }

        // Parse training data block { [v1, v2, ...] = index, ... }
        let mut samples = Vec::new();
        self.skip_newlines();
        if self.check(&Token::LBrace) {
            self.advance();
            loop {
                self.skip_newlines();
                if self.check(&Token::RBrace) { self.advance(); break; }
                if self.at_end() { break; }

                self.expect_token(&Token::LBracket)?;
                let mut values = Vec::new();
                loop {
                    if self.check(&Token::RBracket) { self.advance(); break; }
                    values.push(self.expect_number()?);
                    if self.check(&Token::Comma) { self.advance(); }
                }
                self.expect_token(&Token::Equals)?;
                let expected = self.expect_int()? as usize;
                samples.push((values, expected));
                if self.check(&Token::Comma) { self.advance(); }
            }
        }

        Ok(Statement::Learn(LearnCmd { name, dim, quantum, seed, samples }))
    }

    // --- Input block: { x = [1.0, 2.0], y = [3.0] } ---
    fn parse_input_block(&mut self) -> Result<Vec<(String, Vec<f64>)>> {
        self.skip_newlines();
        let mut inputs = Vec::new();
        if !self.check(&Token::LBrace) {
            return Ok(inputs);
        }
        self.advance(); // skip {

        loop {
            self.skip_newlines();
            if self.check(&Token::RBrace) { self.advance(); break; }
            if self.at_end() { break; }

            let name = self.expect_ident()?;
            self.expect_token(&Token::Equals)?;
            self.expect_token(&Token::LBracket)?;

            let mut values = Vec::new();
            loop {
                if self.check(&Token::RBracket) { self.advance(); break; }
                let val = self.expect_number()?;
                values.push(val);
                if self.check(&Token::Comma) { self.advance(); }
            }
            inputs.push((name, values));
            // skip optional comma
            if self.check(&Token::Comma) { self.advance(); }
        }

        Ok(inputs)
    }

    // --- Helpers ---

    fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> &Token {
        let tok = self.tokens.get(self.pos).unwrap_or(&Token::Eof);
        self.pos += 1;
        tok
    }

    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len() || matches!(self.current(), Token::Eof)
    }

    fn check(&self, expected: &Token) -> bool {
        std::mem::discriminant(self.current()) == std::mem::discriminant(expected)
    }

    fn skip_newlines(&mut self) {
        while self.pos < self.tokens.len() && matches!(self.current(), Token::Newline) {
            self.pos += 1;
        }
    }

    fn expect_token(&mut self, expected: &Token) -> Result<()> {
        if self.check(expected) {
            self.advance();
            Ok(())
        } else {
            Err(AxolError::Parse(format!("Expected {:?}, got {:?}", expected, self.current())))
        }
    }

    fn expect_ident(&mut self) -> Result<String> {
        if let Token::Ident(ref s) = self.current().clone() {
            let s = s.clone();
            self.advance();
            Ok(s)
        } else {
            Err(AxolError::Parse(format!("Expected identifier, got {:?}", self.current())))
        }
    }

    fn expect_string_or_ident(&mut self) -> Result<String> {
        match self.current().clone() {
            Token::StringLit(s) => { self.advance(); Ok(s) }
            Token::Ident(s) => { self.advance(); Ok(s) }
            other => Err(AxolError::Parse(format!("Expected string or ident, got {:?}", other))),
        }
    }

    fn expect_int(&mut self) -> Result<i64> {
        match self.current().clone() {
            Token::Int(n) => { self.advance(); Ok(n) }
            Token::Number(n) => { self.advance(); Ok(n as i64) }
            other => Err(AxolError::Parse(format!("Expected integer, got {:?}", other))),
        }
    }

    fn expect_number(&mut self) -> Result<f64> {
        match self.current().clone() {
            Token::Number(n) => { self.advance(); Ok(n) }
            Token::Int(n) => { self.advance(); Ok(n as f64) }
            other => Err(AxolError::Parse(format!("Expected number, got {:?}", other))),
        }
    }

    fn expect_relation(&mut self) -> Result<String> {
        if let Token::Relation(ref s) = self.current().clone() {
            let s = s.clone();
            self.advance();
            Ok(s)
        } else {
            Err(AxolError::Parse(format!("Expected relation (<~>, etc), got {:?}", self.current())))
        }
    }
}
