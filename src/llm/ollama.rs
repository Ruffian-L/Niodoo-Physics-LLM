use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

const OLLAMA_API_URL: &str = "http://localhost:11434/api/chat";
const DEFAULT_MODEL: &str = "gemma3:4b-it-qat";

#[derive(Clone)]
pub struct OllamaClient {
    client: Client,
    model: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Serialize, Debug)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    format: Option<String>, // "json"
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct OllamaResponse {
    message: OllamaMessage,
    done: bool,
}

#[derive(Deserialize, Debug)]
pub struct SentimentResponse {
    pub response: String,
    pub valence: f32, // -1.0 to 1.0
}

impl OllamaClient {
    pub fn new(model: Option<String>) -> Self {
        Self {
            client: Client::new(),
            model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
        }
    }

    pub async fn chat(
        &self,
        system_prompt: &str,
        user_query: &str,
        context: &str,
    ) -> Result<String> {
        let full_system_prompt = format!("{}\n\nCONTEXT FROM MEMORY:\n{}", system_prompt, context);

        let request = OllamaRequest {
            model: self.model.clone(),
            messages: vec![
                OllamaMessage {
                    role: "system".to_string(),
                    content: full_system_prompt,
                },
                OllamaMessage {
                    role: "user".to_string(),
                    content: user_query.to_string(),
                },
            ],
            stream: false,
            format: None,
        };

        let res = self
            .client
            .post(OLLAMA_API_URL)
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to contact Ollama: {}", e))?;

        if !res.status().is_success() {
            return Err(anyhow!("Ollama API error: {}", res.status()));
        }

        let body: OllamaResponse = res
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse Ollama response: {}", e))?;

        Ok(body.message.content)
    }

    pub async fn chat_with_sentiment(
        &self,
        system_prompt: &str,
        user_query: &str,
        context: &str,
    ) -> Result<SentimentResponse> {
        let full_system_prompt = format!(
            "{}\n\nCONTEXT FROM MEMORY:\n{}\n\nIMPORTANT: Output ONLY valid JSON with fields 'response' (string) and 'valence' (float -1.0 to 1.0).",
            system_prompt, context
        );

        let request = OllamaRequest {
            model: self.model.clone(),
            messages: vec![
                OllamaMessage {
                    role: "system".to_string(),
                    content: full_system_prompt,
                },
                OllamaMessage {
                    role: "user".to_string(),
                    content: user_query.to_string(),
                },
            ],
            stream: false,
            format: Some("json".to_string()), // Force JSON output
        };

        let res = self
            .client
            .post(OLLAMA_API_URL)
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to contact Ollama: {}", e))?;

        if !res.status().is_success() {
            return Err(anyhow!("Ollama API error: {}", res.status()));
        }

        let body: OllamaResponse = res
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse Ollama response: {}", e))?;

        let content = body.message.content;
        let sentiment: SentimentResponse = serde_json::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse JSON from LLM: {}. Content: {}", e, content))?;

        Ok(sentiment)
    }
}
