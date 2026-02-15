//! Data module for AXOL Text Model.
//!
//! Provides built-in training corpora and file-based data loading.

use std::fs;
use std::path::Path;

/// Load sentences from a plain text file (one sentence per line).
///
/// Skips empty lines and lines starting with '#'.
pub fn load_corpus_from_file(path: &str) -> Vec<String> {
    let content = fs::read_to_string(Path::new(path))
        .unwrap_or_else(|e| panic!("Failed to read corpus file '{}': {}", path, e));
    content
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

/// Built-in medium corpus (~150 sentences) for scale testing.
///
/// Covers diverse patterns: subject-verb-object, adjectives, prepositions,
/// pronouns, temporal phrases, questions, and multi-clause sentences.
pub fn medium_corpus() -> Vec<&'static str> {
    vec![
        // --- Animals & actions ---
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat ate the fish",
        "the dog ate the bone",
        "the bird flew over the tree",
        "the fish swam in the water",
        "the cat chased the mouse",
        "the dog chased the cat",
        "the bird sang a song",
        "the mouse ran into the hole",
        "the cat slept on the bed",
        "the dog slept on the floor",
        "the bird built a nest",
        "the fish jumped out of the water",
        "the cat climbed the tree",
        "the dog dug a hole in the ground",
        "the rabbit hopped across the field",
        "the horse ran across the field",
        "the cow ate the grass",
        "the sheep followed the shepherd",

        // --- People & daily actions ---
        "the boy went to school",
        "the girl went to school",
        "the man walked to the store",
        "the woman walked to the park",
        "the boy read a book",
        "the girl read a book",
        "the man ate lunch at the table",
        "the woman ate dinner at the table",
        "the boy played in the park",
        "the girl played in the garden",
        "the man drove the car to work",
        "the woman drove the car to the store",
        "the boy kicked the ball",
        "the girl threw the ball",
        "the man wrote a letter",
        "the woman wrote a story",
        "the boy drew a picture",
        "the girl drew a picture of the cat",
        "the man built a house",
        "the woman built a garden",

        // --- Adjectives & descriptions ---
        "the big cat sat on the small mat",
        "the small dog sat on the big log",
        "a red bird flew over the tall tree",
        "the old man walked slowly to the store",
        "the young girl ran quickly to school",
        "a big fish swam in the deep water",
        "the small mouse hid in the dark hole",
        "the tall tree stood in the green field",
        "the bright sun shone in the blue sky",
        "the cold wind blew across the open field",
        "a beautiful flower grew in the garden",
        "the long road led to the old town",
        "a fast car drove down the wide road",
        "the round ball rolled across the flat floor",
        "a hot fire burned in the dark room",

        // --- Locations & prepositions ---
        "the cat is on the table",
        "the dog is under the table",
        "the book is on the shelf",
        "the ball is under the bed",
        "the bird is in the tree",
        "the fish is in the pond",
        "the keys are on the desk",
        "the cup is on the counter",
        "the hat is on the chair",
        "the shoes are by the door",
        "the car is in the garage",
        "the boat is on the lake",
        "the plane is in the sky",
        "the flowers are in the vase",
        "the food is on the plate",

        // --- Temporal patterns ---
        "the cat sat on the mat in the morning",
        "the dog ate the bone in the evening",
        "the boy went to school every day",
        "the girl read a book every night",
        "the man walked to work in the morning",
        "the woman cooked dinner in the evening",
        "the bird sang a song at dawn",
        "the sun set behind the mountains at dusk",
        "the rain fell all day long",
        "the snow covered the ground all winter",

        // --- Compound sentences ---
        "the cat sat on the mat and the dog sat on the log",
        "the boy went to school and the girl went to the park",
        "the man ate lunch and the woman ate dinner",
        "the bird flew over the tree and sang a song",
        "the fish swam in the water and jumped over the rock",
        "the cat chased the mouse but the mouse escaped",
        "the dog ran fast but the cat ran faster",
        "the boy read a book and then went to sleep",
        "the girl drew a picture and showed it to the teacher",
        "the man drove to work and then drove back home",

        // --- Knowledge & facts ---
        "the sun is a star",
        "the moon orbits the earth",
        "water flows down the river to the sea",
        "trees grow tall in the forest",
        "birds fly south in the winter",
        "fish live in the ocean",
        "the sky is blue during the day",
        "the stars shine at night",
        "rain comes from the clouds",
        "snow falls in the winter",
        "flowers bloom in the spring",
        "leaves fall in the autumn",
        "the wind blows from the west",
        "the river flows to the east",
        "mountains are tall and covered with snow",

        // --- Food & cooking ---
        "the cook made soup for dinner",
        "the baker baked bread in the oven",
        "the farmer grew corn in the field",
        "the boy ate an apple for lunch",
        "the girl drank milk with her meal",
        "the man made tea in the kitchen",
        "the woman made coffee in the morning",
        "the cook cut the vegetables on the board",
        "the family ate dinner at the table",
        "the children ate fruit after school",

        // --- Weather ---
        "the sun shone brightly in the sky",
        "the rain fell on the roof",
        "the wind blew through the trees",
        "the snow fell on the ground",
        "the clouds covered the sky",
        "the storm came from the north",
        "the fog covered the city in the morning",
        "the thunder roared across the sky",
        "the lightning flashed in the dark sky",
        "the rainbow appeared after the rain",

        // --- Repeated patterns for learning ---
        "the cat sat on the mat again",
        "the dog sat on the log again",
        "the boy went to school again",
        "the girl read another book",
        "the man walked to the store again",
        "the bird flew back to the nest",
        "the fish swam back to the pond",
        "the cat came back to the house",
        "the dog came back from the park",
        "the sun came up in the morning",

        // --- More variety ---
        "a small child played with a toy",
        "the teacher taught the students in the room",
        "the doctor helped the sick man",
        "the farmer fed the animals on the farm",
        "the driver drove the bus to the city",
        "the pilot flew the plane over the ocean",
        "the sailor sailed the boat across the sea",
        "the king ruled the land for many years",
        "the queen sat on the throne in the castle",
        "the knight rode the horse to the battle",
    ]
}

/// Tiny corpus for quick testing (~20 sentences).
pub fn tiny_corpus() -> Vec<&'static str> {
    vec![
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat ate the fish",
        "the dog ate the bone",
        "a big cat sat on a big mat",
        "a small dog sat on a small log",
        "the red cat sat on the red mat",
        "the blue dog sat on the blue log",
        "the cat is a good cat",
        "the dog is a good dog",
        "i saw the cat on the mat",
        "i saw the dog on the log",
        "the cat ran to the mat",
        "the dog ran to the log",
        "a cat and a dog sat on the mat",
        "the fish swam in the water",
        "the bone was on the ground",
        "a big fish swam in the water",
        "the cat sat on the mat again",
        "the dog sat on the log again",
    ]
}

/// Corpus statistics.
pub struct CorpusStats {
    pub num_sentences: usize,
    pub num_words: usize,
    pub unique_words: usize,
    pub avg_sentence_len: f64,
    pub max_sentence_len: usize,
}

impl CorpusStats {
    pub fn from_corpus(sentences: &[&str]) -> Self {
        let num_sentences = sentences.len();
        let mut total_words = 0;
        let mut max_len = 0;
        let mut unique = std::collections::HashSet::new();

        for s in sentences {
            let words: Vec<&str> = s.split_whitespace().collect();
            total_words += words.len();
            if words.len() > max_len {
                max_len = words.len();
            }
            for w in &words {
                unique.insert(w.to_lowercase());
            }
        }

        CorpusStats {
            num_sentences,
            num_words: total_words,
            unique_words: unique.len(),
            avg_sentence_len: total_words as f64 / num_sentences.max(1) as f64,
            max_sentence_len: max_len,
        }
    }

    pub fn print(&self) {
        println!("  Sentences:        {}", self.num_sentences);
        println!("  Total words:      {}", self.num_words);
        println!("  Unique words:     {}", self.unique_words);
        println!("  Avg sentence len: {:.1}", self.avg_sentence_len);
        println!("  Max sentence len: {}", self.max_sentence_len);
    }
}
