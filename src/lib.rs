use anyhow::{anyhow, Error, Result};
use ollama_rs::{generation::embeddings::request::GenerateEmbeddingsRequest, Ollama};

#[derive(Debug)]
pub struct VectorRecord {
    pub prompt: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug)]
pub struct Similarity {
    pub prompt: String,
    pub cosine_similarity: f32,
}

pub fn chunk_text(text: &String, chunk_size: usize, overlap: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let end = if start + chunk_size > text.len() {
            text.len()
        } else {
            start + chunk_size
        };

        // Ensure we are slicing at valid UTF-8 character boundaries
        let end = text.char_indices().nth(end).map_or(text.len(), |(i, _)| i);
        chunks.push(text[start..end].to_string());

        start += chunk_size - overlap;
    }

    chunks
}

/// Get the embedding of a text chunk
pub async fn generate_chunk_embedding(
    ollama_client: &Ollama,
    chunk: &str,
) -> Result<Vec<f32>, Error> {
    let request =
        GenerateEmbeddingsRequest::new("snowflake-arctic-embed:33m".to_string(), chunk.into());
    let res = ollama_client.generate_embeddings(request).await?;
    Ok(res.embeddings.get(0).expect("Missing embedding").clone())
}

pub fn cosine_similarity(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Result<f32, Error> {
    if vec1.len() != vec2.len() {
        return Err(anyhow!("Vector lengths do not match"));
    }

    // See formula : https://www.geeksforgeeks.org/cosine-similarity/
    // SC​(x, y) = x . y / ||x|| × ||y||
    // where,
    //  - x . y = product (dot) of the vectors ‘x’ and ‘y’.
    //  - ||x|| and ||y|| = length (magnitude) of the two vectors ‘x’ and ‘y’.
    //  - ||x|| ×× ||y|| = regular product of the two vectors ‘x’ and ‘y’.
    let mut dot_product = 0.0;
    let mut magnitude1 = 0.0;
    let mut magnitude2 = 0.0;

    for i in 0..vec1.len() {
        dot_product += vec1[i] * vec2[i];
        magnitude1 += vec1[i] * vec1[i];
        magnitude2 += vec2[i] * vec2[i];
    }

    magnitude1 = magnitude1.sqrt();
    magnitude2 = magnitude2.sqrt();

    if magnitude1 == 0.0 || magnitude2 == 0.0 {
        return Err(anyhow!("One of the vectors has zero magnitude"));
    }

    Ok(dot_product / (magnitude1 * magnitude2))
}
