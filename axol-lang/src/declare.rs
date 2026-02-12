//! Declaration: RelationKind, QualityTarget, DeclarationBuilder.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// RelationKind
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum RelationKind {
    Proportional,   // <~>
    Additive,       // <+>
    Multiplicative, // <*>
    Inverse,        // <!>
    Conditional,    // <?>
}

impl RelationKind {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "<~>" | "proportional" => Some(Self::Proportional),
            "<+>" | "additive" => Some(Self::Additive),
            "<*>" | "multiplicative" => Some(Self::Multiplicative),
            "<!>" | "inverse" => Some(Self::Inverse),
            "<?>" | "conditional" => Some(Self::Conditional),
            _ => None,
        }
    }

    pub fn symbol(&self) -> &str {
        match self {
            Self::Proportional => "<~>",
            Self::Additive => "<+>",
            Self::Multiplicative => "<*>",
            Self::Inverse => "<!>",
            Self::Conditional => "<?>",
        }
    }
}

// ---------------------------------------------------------------------------
// QualityTarget
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct QualityTarget {
    pub omega: f64,
    pub phi: f64,
}

impl Default for QualityTarget {
    fn default() -> Self {
        Self { omega: 0.9, phi: 0.7 }
    }
}

// ---------------------------------------------------------------------------
// DeclaredInput
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct DeclaredInput {
    pub name: String,
    pub dim: usize,
    pub labels: HashMap<usize, String>,
}

// ---------------------------------------------------------------------------
// DeclaredRelation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct DeclaredRelation {
    pub target: String,
    pub sources: Vec<String>,
    pub kind: RelationKind,
    pub nonlinear: bool,
}

// ---------------------------------------------------------------------------
// EntangleDeclaration
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct EntangleDeclaration {
    pub name: String,
    pub inputs: Vec<DeclaredInput>,
    pub outputs: Vec<String>,
    pub relations: Vec<DeclaredRelation>,
    pub quality: QualityTarget,
}

// ---------------------------------------------------------------------------
// DeclarationBuilder
// ---------------------------------------------------------------------------

pub struct DeclarationBuilder {
    name: String,
    inputs: Vec<DeclaredInput>,
    outputs: Vec<String>,
    relations: Vec<DeclaredRelation>,
    quality: QualityTarget,
}

impl DeclarationBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            relations: Vec::new(),
            quality: QualityTarget::default(),
        }
    }

    pub fn input(&mut self, name: &str, dim: usize) -> &mut Self {
        self.inputs.push(DeclaredInput {
            name: name.to_string(),
            dim,
            labels: HashMap::new(),
        });
        self
    }

    pub fn input_with_labels(&mut self, name: &str, dim: usize, labels: HashMap<usize, String>) -> &mut Self {
        self.inputs.push(DeclaredInput {
            name: name.to_string(),
            dim,
            labels,
        });
        self
    }

    pub fn output(&mut self, name: &str) -> &mut Self {
        self.outputs.push(name.to_string());
        self
    }

    pub fn relate(&mut self, target: &str, sources: &[&str], kind: RelationKind) -> &mut Self {
        self.relations.push(DeclaredRelation {
            target: target.to_string(),
            sources: sources.iter().map(|s| s.to_string()).collect(),
            kind,
            nonlinear: false,
        });
        self
    }

    pub fn relate_nonlinear(&mut self, target: &str, sources: &[&str], kind: RelationKind) -> &mut Self {
        self.relations.push(DeclaredRelation {
            target: target.to_string(),
            sources: sources.iter().map(|s| s.to_string()).collect(),
            kind,
            nonlinear: true,
        });
        self
    }

    pub fn quality(&mut self, omega: f64, phi: f64) -> &mut Self {
        self.quality = QualityTarget { omega, phi };
        self
    }

    pub fn build(self) -> EntangleDeclaration {
        EntangleDeclaration {
            name: self.name,
            inputs: self.inputs,
            outputs: self.outputs,
            relations: self.relations,
            quality: self.quality,
        }
    }
}
