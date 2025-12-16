use anyhow::Result;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, IndexWriter, ReloadPolicy};

use tantivy::tokenizer::{
    Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, StopWordFilter, TextAnalyzer,
};

use tantivy::TantivyDocument; // Concrete doc type

pub struct TantivyIndex {
    index: Index,
    writer: Arc<Mutex<IndexWriter>>,
    reader: tantivy::IndexReader,
    // Schema fields
    field_id: Field,
    field_text: Field,
    field_tags: Field,
}

impl TantivyIndex {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let index_path = path.as_ref();
        std::fs::create_dir_all(index_path)?;

        let mut schema_builder = Schema::builder();

        // ID: Stored, not indexed (we lookup by it rarely, mostly return it)
        // Actually we need FAST lookups so we use u64 fast field? No, we return it.
        // We store it so we can return the ID of the match.
        let field_id = schema_builder.add_u64_field("id", STORED | FAST);

        // Text: Indexed, Tokenized (Standard + Ngram)
        // We want to support BOTH exact token matching and fuzzy ngrams.
        // But tantivy fields have one tokenizer.
        // Strategy: Use standard tokenizer for main field, add ngram tokenizer for robustness?
        // Or stick to Ngram for robustness as requested.
        // If Ngram returns 0, maybe the query is too short?
        // "continue with next steps" is long enough for 3-grams.
        // The issue might be the query parser behavior with ngrams.

        // Let's revert to Standard tokenizer but ensure it handles special chars by not stripping them?
        // Actually, "Raw" tokenizer keeps everything.
        // "Standard" strips punctuation.

        // For code logs like `[Project: ...`, standard tokenizer splits to "Project".
        // If I search `[Project:`, standard query parser might get confused.

        // Let's try a simple tokenizer that preserves more, or rely on the sanitization I added in retrieve.rs.
        // I sanitized `[ ] :` to spaces. So `[Project:` becomes ` Project `.
        // Standard tokenizer is fine for that.

        // The Ngram tokenizer might be failing because the query parser expects tokens.

        // Let's switch back to Standard tokenizer to verify baseline functionality first.

        // Use Custom Tokenizer for Text
        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("english_stem")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();

        let field_text = schema_builder.add_text_field("text", text_options);

        // Tags: Standard whitespace
        let field_tags = schema_builder.add_text_field("tags", TEXT | STORED);

        let schema = schema_builder.build();

        let index = Index::create_in_dir(index_path, schema.clone())
            .or_else(|_| Index::open_in_dir(index_path))?;

        // Register Custom Tokenizer (English Stemmer + Stop Words)
        let tokenizer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(40))
            .filter(LowerCaser)
            .filter(StopWordFilter::new(Language::English).unwrap())
            .filter(Stemmer::new(Language::English))
            .build();

        index.tokenizers().register("english_stem", tokenizer);

        // Register Ngram Tokenizer
        let tokenizer_ngram = tantivy::tokenizer::NgramTokenizer::new(3, 3, false).unwrap();
        index.tokenizers().register("ngram3", tokenizer_ngram);

        // 50MB buffer for indexing
        let writer = index.writer(50_000_000)?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual) // Explicit control
            .try_into()?;

        Ok(Self {
            index,
            writer: Arc::new(Mutex::new(writer)),
            reader,
            field_id,
            field_text,
            field_tags,
        })
    }

    pub fn add_document(&self, id: u64, text: &str, tags: &[String]) -> Result<()> {
        let mut doc = TantivyDocument::default();
        doc.add_u64(self.field_id, id);
        doc.add_text(self.field_text, text);
        doc.add_text(self.field_tags, tags.join(" "));

        let writer = self.writer.lock().unwrap();
        writer.add_document(doc)?;
        // Removed auto-commit for performance. User must call commit().
        Ok(())
    }

    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.lock().unwrap();
        writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// Search for documents matching the query using BM25.
    /// Returns a list of (id, score).
    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<(u64, f32)>> {
        let searcher = self.reader.searcher();

        let query_parser =
            QueryParser::for_index(&self.index, vec![self.field_text, self.field_tags]);
        let query = query_parser.parse_query(query_str)?;

        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(id_val) = retrieved_doc.get_first(self.field_id) {
                if let Some(id) = id_val.as_u64() {
                    results.push((id, score));
                }
            }
        }

        Ok(results)
    }

    pub fn num_docs(&self) -> u64 {
        let searcher = self.reader.searcher();
        searcher.num_docs()
    }
}
