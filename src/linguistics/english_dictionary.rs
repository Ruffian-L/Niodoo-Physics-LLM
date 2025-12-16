//! GAUSSIAN PRIME (Gʘ) → English Dictionary Bridge
//! Building hierarchical language from geometric symbols

use std::collections::HashMap;
use crate::types::Vec3;

/// English word categories mapped from Gʘ symbols
#[derive(Debug, Clone, PartialEq)]
pub enum WordCategory {
    /// Objects with elongated structure (LINE, THIN_LINE)
    Vehicle,
    Tool,
    Weapon,
    Furniture,
    
    /// Objects with planar structure (PLANE, THIN_PLANE)
    Surface,
    Container,
    Building,
    Clothing,
    
    /// Objects with spherical structure (SPHERE, BALL)
    Organic,
    Food,
    Animal,
    Person,
    
    /// Complex structures (COMPLEX_1 through COMPLEX_7)
    Machine,
    Technology,
    Art,
    Nature,
    
    /// Chaotic structures (CHAOTIC_1 through CHAOTIC_3)
    Abstract,
    Emotion,
    Concept,
}

/// Hierarchical word builder from Gʘ symbols
pub struct EnglishDictionary {
    /// Symbol → word category mapping
    symbol_categories: HashMap<String, WordCategory>,
    
    /// Category → word lists (frequency ranked)
    category_words: HashMap<WordCategory, Vec<String>>,
    
    /// Contextual word relationships
    word_contexts: HashMap<String, Vec<String>>,
}

impl EnglishDictionary {
    pub fn new() -> Self {
        let mut dict = Self {
            symbol_categories: HashMap::new(),
            category_words: HashMap::new(),
            word_contexts: HashMap::new(),
        };
        
        dict.initialize_mappings();
        dict
    }
    
    fn initialize_mappings(&mut self) {
        // Map Gʘ symbols to word categories
        self.symbol_categories.insert("LINE".to_string(), WordCategory::Vehicle);
        self.symbol_categories.insert("THIN_LINE".to_string(), WordCategory::Tool);
        self.symbol_categories.insert("PLANE".to_string(), WordCategory::Surface);
        self.symbol_categories.insert("SPHERE".to_string(), WordCategory::Organic);
        self.symbol_categories.insert("COMPLEX_3".to_string(), WordCategory::Machine);
        self.symbol_categories.insert("CHAOTIC_2".to_string(), WordCategory::Emotion);
        
        // Initialize word vocabulary by category
        self.initialize_vocabulary();
    }
    
    fn initialize_vocabulary(&mut self) {
        // Vehicle vocabulary (most common first)
        self.category_words.insert(WordCategory::Vehicle, vec![
            "car".to_string(), "truck".to_string(), "bus".to_string(),
            "train".to_string(), "airplane".to_string(), "boat".to_string(),
            "bicycle".to_string(), "motorcycle".to_string(), "scooter".to_string(),
            "van".to_string(), "taxi".to_string(), "ambulance".to_string(),
            "firetruck".to_string(), "police_car".to_string(), "tractor".to_string(),
            "tank".to_string(), "helicopter".to_string(), "submarine".to_string(),
            "rocket".to_string(), "spaceship".to_string(), "cart".to_string(),
            "wagon".to_string(), "sled".to_string(), "trailer".to_string(),
            // ... could extend to thousands more
        ]);
        
        // Surface vocabulary
        self.category_words.insert(WordCategory::Surface, vec![
            "table".to_string(), "floor".to_string(), "wall".to_string(),
            "ceiling".to_string(), "road".to_string(), "ground".to_string(),
            "screen".to_string(), "paper".to_string(), "page".to_string(),
            "board".to_string(), "desk".to_string(), "counter".to_string(),
            "roof".to_string(), "window".to_string(), "door".to_string(),
            "mirror".to_string(), "glass".to_string(), "water".to_string(),
            "ice".to_string(), "sand".to_string(), "grass".to_string(),
            "field".to_string(), "meadow".to_string(), "plain".to_string(),
            // ... thousands more surface words
        ]);
        
        // Organic vocabulary
        self.category_words.insert(WordCategory::Organic, vec![
            "person".to_string(), "animal".to_string(), "plant".to_string(),
            "tree".to_string(), "flower".to_string(), "fruit".to_string(),
            "vegetable".to_string(), "body".to_string(), "head".to_string(),
            "hand".to_string(), "foot".to_string(), "eye".to_string(),
            "heart".to_string(), "brain".to_string(), "blood".to_string(),
            "skin".to_string(), "bone".to_string(), "muscle".to_string(),
            "leaf".to_string(), "root".to_string(), "seed".to_string(),
            "branch".to_string(), "trunk".to_string(), "bark".to_string(),
            // ... thousands more organic words
        ]);
        
        // Machine vocabulary
        self.category_words.insert(WordCategory::Machine, vec![
            "computer".to_string(), "phone".to_string(), "engine".to_string(),
            "motor".to_string(), "pump".to_string(), "fan".to_string(),
            "clock".to_string(), "watch".to_string(), "camera".to_string(),
            "printer".to_string(), "scanner".to_string(), "keyboard".to_string(),
            "mouse".to_string(), "monitor".to_string(), "speaker".to_string(),
            "microphone".to_string(), "router".to_string(), "server".to_string(),
            "robot".to_string(), "drone".to_string(), "appliance".to_string(),
            "tool".to_string(), "device".to_string(), "gadget".to_string(),
            // ... thousands more machine words
        ]);
        
        // Emotion vocabulary (abstract)
        self.category_words.insert(WordCategory::Emotion, vec![
            "love".to_string(), "hate".to_string(), "fear".to_string(),
            "anger".to_string(), "joy".to_string(), "sadness".to_string(),
            "happiness".to_string(), "excitement".to_string(), "calm".to_string(),
            "stress".to_string(), "anxiety".to_string(), "peace".to_string(),
            "hope".to_string(), "despair".to_string(), "trust".to_string(),
            "doubt".to_string(), "confidence".to_string(), "insecurity".to_string(),
            "pride".to_string(), "shame".to_string(), "guilt".to_string(),
            "gratitude".to_string(), "resentment".to_string(), "forgiveness".to_string(),
            // ... thousands more emotion words
        ]);
    }
    
    /// Translate Gʘ symbols to English words
    pub fn translate_to_english(&self, gzero_words: &[String]) -> Vec<String> {
        let mut english_words = Vec::new();
        
        for gzero_word in gzero_words {
            if let Some(category) = self.symbol_categories.get(gzero_word) {
                if let Some(words) = self.category_words.get(category) {
                    // Select word based on context and frequency
                    let word_index = self.select_word_index(gzero_word, words.len());
                    if let Some(word) = words.get(word_index) {
                        english_words.push(word.clone());
                    }
                }
            }
        }
        
        english_words
    }
    
    /// Select appropriate word index based on context
    fn select_word_index(&self, _gzero_word: &str, vocab_size: usize) -> usize {
        // For now, use frequency-based selection
        // In future, this could consider:
        // - Previous word context
        // - Semantic coherence
        // - User preferences
        // - Cultural context
        
        // Start with most common words, gradually expand
        std::cmp::min(vocab_size / 10, vocab_size - 1)
    }
    
    /// Get vocabulary statistics
    pub fn vocabulary_stats(&self) -> VocabularyStats {
        let total_words: usize = self.category_words.values()
            .map(|words| words.len())
            .sum();
            
        let total_categories = self.category_words.len();
        
        VocabularyStats {
            total_words,
            total_categories,
            avg_words_per_category: total_words / total_categories.max(1),
        }
    }
}

/// Vocabulary statistics
#[derive(Debug, Clone)]
pub struct VocabularyStats {
    pub total_words: usize,
    pub total_categories: usize,
    pub avg_words_per_category: usize,
}

/// Context-aware sentence builder
pub struct SentenceBuilder {
    dictionary: EnglishDictionary,
    grammar_rules: GrammarRules,
}

impl SentenceBuilder {
    pub fn new() -> Self {
        Self {
            dictionary: EnglishDictionary::new(),
            grammar_rules: GrammarRules::new(),
        }
    }
    
    /// Build coherent sentences from Gʘ symbols
    pub fn build_sentence(&self, gzero_words: &[String]) -> String {
        let english_words = self.dictionary.translate_to_english(gzero_words);
        
        // Apply grammar rules to form coherent sentences
        self.grammar_rules.apply_rules(english_words)
    }
}

/// Basic grammar rules for sentence construction
pub struct GrammarRules {
    /// Common sentence patterns
    patterns: Vec<Vec<String>>,
}

impl GrammarRules {
    pub fn new() -> Self {
        Self {
            patterns: vec![
                vec!["Subject".to_string(), "Verb".to_string(), "Object".to_string()],
                vec!["Article".to_string(), "Adjective".to_string(), "Noun".to_string()],
                vec!["Preposition".to_string(), "Article".to_string(), "Noun".to_string()],
            ],
        }
    }
    
    pub fn apply_rules(&self, words: Vec<String>) -> String {
        // For now, simple word joining
        // In future, this could apply proper grammar:
        // - Subject-verb agreement
        // - Tense consistency
        // - Pluralization
        // - Articles and prepositions
        
        words.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dictionary_creation() {
        let dict = EnglishDictionary::new();
        let stats = dict.vocabulary_stats();
        
        assert!(stats.total_words > 100);
        assert!(stats.total_categories > 5);
    }
    
    #[test]
    fn test_basic_translation() {
        let dict = EnglishDictionary::new();
        let gzero_words = vec!["LINE".to_string(), "PLANE".to_string(), "SPHERE".to_string()];
        let english = dict.translate_to_english(&gzero_words);
        
        assert_eq!(english.len(), 3);
        assert!(english.contains(&"car".to_string()));
        assert!(english.contains(&"table".to_string()));
        assert!(english.contains(&"person".to_string()));
    }
}
