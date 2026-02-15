//! Simple word-level tokenizer with vocabulary management.
//!
//! For PoC: whitespace-based tokenization with lowercasing.
//! Special tokens: <PAD>, <UNK>, <BOS>, <EOS>

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Special token IDs
// ---------------------------------------------------------------------------

pub const PAD_ID: usize = 0;
pub const UNK_ID: usize = 1;
pub const BOS_ID: usize = 2;
pub const EOS_ID: usize = 3;
pub const SPECIAL_COUNT: usize = 4;

/// Word-start marker used by BPE tokenizer (SentencePiece-style).
const WORD_PREFIX: &str = "\u{2581}"; // ▁ (U+2581)

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

/// Word-level vocabulary mapping words ↔ integer IDs.
#[derive(Clone, Debug)]
pub struct Vocabulary {
    pub word_to_id: HashMap<String, usize>,
    pub id_to_word: Vec<String>,
    pub size: usize,
}

impl Vocabulary {
    /// Build vocabulary from a corpus (list of sentences).
    /// Keeps the top `max_size` most frequent words.
    pub fn from_corpus(sentences: &[&str], max_size: usize) -> Self {
        // Count word frequencies
        let mut freq: HashMap<String, usize> = HashMap::new();
        for sentence in sentences {
            for word in tokenize_raw(sentence) {
                *freq.entry(word).or_insert(0) += 1;
            }
        }

        // Sort by frequency (descending)
        let mut words: Vec<(String, usize)> = freq.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1));

        // Build vocabulary with special tokens first
        let mut word_to_id = HashMap::new();
        let mut id_to_word = vec![
            "<PAD>".to_string(),
            "<UNK>".to_string(),
            "<BOS>".to_string(),
            "<EOS>".to_string(),
        ];

        word_to_id.insert("<PAD>".to_string(), PAD_ID);
        word_to_id.insert("<UNK>".to_string(), UNK_ID);
        word_to_id.insert("<BOS>".to_string(), BOS_ID);
        word_to_id.insert("<EOS>".to_string(), EOS_ID);

        let capacity = max_size.saturating_sub(SPECIAL_COUNT);
        for (word, _count) in words.into_iter().take(capacity) {
            let id = id_to_word.len();
            word_to_id.insert(word.clone(), id);
            id_to_word.push(word);
        }

        let size = id_to_word.len();
        Self { word_to_id, id_to_word, size }
    }

    /// Encode a word to its ID. Returns UNK_ID for unknown words.
    pub fn encode_word(&self, word: &str) -> usize {
        let lower = word.to_lowercase();
        *self.word_to_id.get(&lower).unwrap_or(&UNK_ID)
    }

    /// Decode an ID to its word.
    pub fn decode_id(&self, id: usize) -> &str {
        if id < self.id_to_word.len() {
            &self.id_to_word[id]
        } else {
            "<UNK>"
        }
    }

    /// Encode a sentence to a sequence of IDs.
    pub fn encode(&self, sentence: &str) -> Vec<usize> {
        let mut ids = vec![BOS_ID];
        for word in tokenize_raw(sentence) {
            ids.push(self.encode_word(&word));
        }
        ids.push(EOS_ID);
        ids
    }

    /// Decode a sequence of IDs to a sentence.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter(|&&id| id != PAD_ID && id != BOS_ID && id != EOS_ID)
            .map(|&id| self.decode_id(id))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// ---------------------------------------------------------------------------
// Tokenization helpers
// ---------------------------------------------------------------------------

/// Simple whitespace tokenizer with lowercasing and punctuation splitting.
fn tokenize_raw(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for word in text.split_whitespace() {
        let lower = word.to_lowercase();
        // Split trailing punctuation
        let trimmed = lower.trim_end_matches(|c: char| c.is_ascii_punctuation());
        let punct = &lower[trimmed.len()..];

        if !trimmed.is_empty() {
            tokens.push(trimmed.to_string());
        }
        if !punct.is_empty() {
            tokens.push(punct.to_string());
        }
    }
    tokens
}

// ---------------------------------------------------------------------------
// BPE Tokenizer
// ---------------------------------------------------------------------------

/// A single BPE merge operation.
#[derive(Clone, Debug)]
pub struct BpeMerge {
    pub pair: (String, String),
    pub merged: String,
    pub rank: usize,
}

/// Byte-Pair Encoding tokenizer for subword tokenization.
///
/// Learns merge operations from corpus and applies them to split text into
/// subword tokens. Uses a word-start marker (▁) following the SentencePiece
/// convention so that decode can reconstruct word boundaries.
#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    pub merges: Vec<BpeMerge>,
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: Vec<String>,
    pub size: usize,
}

impl BpeTokenizer {
    /// Learn BPE merges from a corpus.
    ///
    /// `sentences`: Training text
    /// `num_merges`: Maximum number of merge operations to learn
    /// `max_vocab`: Maximum vocabulary size (including special tokens)
    pub fn from_corpus(sentences: &[&str], num_merges: usize, max_vocab: usize) -> Self {
        // 1. Count word frequencies
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for sentence in sentences {
            for word in tokenize_raw(sentence) {
                *word_freq.entry(word).or_insert(0) += 1;
            }
        }

        // 2. Split each word into characters with ▁ prefix
        //    "cat" → ["▁c", "a", "t"]
        let mut word_splits: Vec<(Vec<String>, usize)> = Vec::new();
        for (word, freq) in &word_freq {
            let mut chars: Vec<String> = Vec::new();
            for (i, c) in word.chars().enumerate() {
                if i == 0 {
                    chars.push(format!("{}{}", WORD_PREFIX, c));
                } else {
                    chars.push(c.to_string());
                }
            }
            word_splits.push((chars, *freq));
        }

        // 3. Build initial vocabulary from character tokens
        let mut token_to_id: HashMap<String, usize> = HashMap::new();
        let mut id_to_token = vec![
            "<PAD>".to_string(),
            "<UNK>".to_string(),
            "<BOS>".to_string(),
            "<EOS>".to_string(),
        ];
        token_to_id.insert("<PAD>".to_string(), PAD_ID);
        token_to_id.insert("<UNK>".to_string(), UNK_ID);
        token_to_id.insert("<BOS>".to_string(), BOS_ID);
        token_to_id.insert("<EOS>".to_string(), EOS_ID);

        for (splits, _) in &word_splits {
            for ch in splits {
                if !token_to_id.contains_key(ch) {
                    let id = id_to_token.len();
                    token_to_id.insert(ch.clone(), id);
                    id_to_token.push(ch.clone());
                }
            }
        }

        // 4. Iteratively merge the most frequent adjacent pair
        let mut merges = Vec::new();

        for merge_idx in 0..num_merges {
            if id_to_token.len() >= max_vocab {
                break;
            }

            // Count pair frequencies (weighted by word frequency)
            let mut pair_freq: HashMap<(String, String), usize> = HashMap::new();
            for (splits, freq) in &word_splits {
                for w in splits.windows(2) {
                    let pair = (w[0].clone(), w[1].clone());
                    *pair_freq.entry(pair).or_insert(0) += freq;
                }
            }

            if pair_freq.is_empty() {
                break;
            }

            // Find most frequent pair
            let (left, right) = match pair_freq
                .iter()
                .max_by_key(|&(_, &count)| count)
                .map(|(pair, _)| pair.clone())
            {
                Some(p) => p,
                None => break,
            };

            let merged = format!("{}{}", left, right);

            merges.push(BpeMerge {
                pair: (left.clone(), right.clone()),
                merged: merged.clone(),
                rank: merge_idx,
            });

            // Add merged token to vocabulary
            if !token_to_id.contains_key(&merged) {
                let id = id_to_token.len();
                token_to_id.insert(merged.clone(), id);
                id_to_token.push(merged.clone());
            }

            // Apply merge to all word splits
            for (splits, _) in word_splits.iter_mut() {
                let mut i = 0;
                while i + 1 < splits.len() {
                    if splits[i] == left && splits[i + 1] == right {
                        splits[i] = merged.clone();
                        splits.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        let size = id_to_token.len();
        Self { merges, token_to_id, id_to_token, size }
    }

    /// Encode a sentence into subword token IDs.
    ///
    /// Each word is split into characters with a ▁ prefix, then learned
    /// merges are applied in order to produce subword tokens.
    pub fn encode(&self, sentence: &str) -> Vec<usize> {
        let mut ids = vec![BOS_ID];

        for word in tokenize_raw(sentence) {
            // Split word into characters with ▁ prefix
            let mut pieces: Vec<String> = Vec::new();
            for (i, c) in word.chars().enumerate() {
                if i == 0 {
                    pieces.push(format!("{}{}", WORD_PREFIX, c));
                } else {
                    pieces.push(c.to_string());
                }
            }

            // Apply learned merges in order
            for merge in &self.merges {
                let mut i = 0;
                while i + 1 < pieces.len() {
                    if pieces[i] == merge.pair.0 && pieces[i + 1] == merge.pair.1 {
                        pieces[i] = merge.merged.clone();
                        pieces.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }

            // Convert pieces to IDs
            for piece in &pieces {
                ids.push(self.token_to_id.get(piece).copied().unwrap_or(UNK_ID));
            }
        }

        ids.push(EOS_ID);
        ids
    }

    /// Decode token IDs back to a string.
    ///
    /// Joins subword tokens and converts ▁ markers back to spaces.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut result = String::new();

        for &id in ids {
            if id == PAD_ID || id == BOS_ID || id == EOS_ID {
                continue;
            }
            if id >= self.id_to_token.len() || self.id_to_token[id] == "<UNK>" {
                result.push_str("<UNK>");
                continue;
            }
            result.push_str(&self.id_to_token[id]);
        }

        // ▁ → space, then trim leading space
        result = result.replace(WORD_PREFIX, " ");
        result.trim().to_string()
    }

    /// Convert to a Vocabulary for backward compatibility with SPS, CW, GO.
    ///
    /// Strips the ▁ prefix from word-start tokens so that the resulting
    /// Vocabulary's encode_word() can match whole words learned by BPE.
    pub fn to_vocabulary(&self) -> Vocabulary {
        let mut word_to_id: HashMap<String, usize> = HashMap::new();
        let mut id_to_word: Vec<String> = Vec::new();

        for (i, token) in self.id_to_token.iter().enumerate() {
            // Strip ▁ prefix for word-level compatibility
            let word = if let Some(stripped) = token.strip_prefix(WORD_PREFIX) {
                stripped.to_string()
            } else {
                token.clone()
            };

            // First mapping wins (preserves special tokens, avoids duplicates)
            if !word_to_id.contains_key(&word) {
                word_to_id.insert(word.clone(), i);
            }
            id_to_word.push(word);
        }

        Vocabulary {
            word_to_id,
            id_to_word,
            size: self.size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_raw() {
        let tokens = tokenize_raw("The cat sat on the mat.");
        assert_eq!(tokens, vec!["the", "cat", "sat", "on", "the", "mat", "."]);
    }

    #[test]
    fn test_vocabulary_from_corpus() {
        let corpus = vec![
            "the cat sat on the mat",
            "the dog sat on the log",
        ];
        let vocab = Vocabulary::from_corpus(&corpus, 100);
        assert!(vocab.size >= SPECIAL_COUNT + 6); // the, cat, sat, on, mat, dog, log
        assert_eq!(vocab.encode_word("the"), vocab.encode_word("The"));
        assert_eq!(vocab.encode_word("xyzzy"), UNK_ID);
    }

    #[test]
    fn test_encode_decode() {
        let corpus = vec!["hello world"];
        let vocab = Vocabulary::from_corpus(&corpus, 100);
        let ids = vocab.encode("hello world");
        assert_eq!(ids[0], BOS_ID);
        assert_eq!(*ids.last().unwrap(), EOS_ID);
        let decoded = vocab.decode(&ids);
        assert_eq!(decoded, "hello world");
    }
}
