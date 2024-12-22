use ollama_rs::{
    generation::{
        chat::{request::ChatMessageRequest, ChatMessage},
        options::GenerationOptions,
    },
    Ollama,
};
use rust_tiny_rag::{
    chunk_text, cosine_similarity, generate_chunk_embedding, Similarity, VectorRecord,
};

// set main as async
#[tokio::main]
async fn main() {
    println!("Welcome to Rust LLM RAG !");

    let system_instructions = "You are a V lang programmer expert.
	Your role is to answer to noob users questions about V language.
	If you need information about the V programmation language, refer only to the provided content.
	";
    let question = "How can I install V ?";

    let ollama = Ollama::new_default_with_history(30);

    let text = std::fs::read_to_string("resources/v-doc.md").unwrap();
    let chunks = chunk_text(&text, 1024, 256);
    let mut vector_store = Vec::new();
    for chunk in chunks {
        let embedding = generate_chunk_embedding(&ollama, &chunk)
            .await
            .expect("Failed to generate embedding");
        let record = VectorRecord {
            prompt: chunk,
            embedding: embedding,
        };
        vector_store.push(record);
    }

    let embedding_question = generate_chunk_embedding(&ollama, &question)
        .await
        .expect("Failed to generate embedding");

    let mut similarities = Vec::new();
    for chunk in vector_store.iter() {
        let cos_similarity = cosine_similarity(&embedding_question, &chunk.embedding)
            .expect("Failed to calculate cosine similarity");
        similarities.push(Similarity {
            prompt: chunk.prompt.clone(),
            cosine_similarity: cos_similarity,
        })
    }

    similarities.sort_by(|a, b| {
        a.cosine_similarity
            .partial_cmp(&b.cosine_similarity)
            .unwrap()
    });
    similarities.reverse();

    let top_5_similarities = &similarities[..5];
    println!("Top 5 similarities :");
    for item in top_5_similarities {
        println!("🔍 Prompt: {}", item.prompt);
        println!("🔍 Cosine similarity: {}", item.cosine_similarity);
        println!("--------------------------------------------------");
    }

    let mut rag_context = String::new();
    for item in top_5_similarities {
        rag_context += item.prompt.as_str();
    }

    let options = GenerationOptions::default()
        .temperature(0.0)
        .repeat_penalty(1.8)
        .repeat_last_n(2)
        .top_k(10)
        .top_p(0.5);

    let result = ollama
        .send_chat_messages(
            ChatMessageRequest::new(
                "qwen2.5:0.5b".to_string(),
                vec![
                    ChatMessage::system(system_instructions.to_string()),
                    ChatMessage::system(format!("CONTENT:\n{}", rag_context)),
                    ChatMessage::user(question.to_string()),
                ],
            )
            .options(options),
        )
        .await
        .unwrap();

    let assistant_message = result.message.unwrap().content;
    println!("🤖 {}", assistant_message);
}
