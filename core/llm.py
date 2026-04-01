from google import genai
from core.config import LLM_MODEL, MAX_CHAT_HISTORY


def _build_prompt(question, context_chunks, chat_history=None):
    """Build the prompt with context and optional chat history."""
    context = "\n\n".join([f"Source: {src}\n{text}" for text, src in context_chunks])

    history_str = ""
    if chat_history:
        for msg in chat_history[-MAX_CHAT_HISTORY:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful assistant that answers questions based only on the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

{"Conversation History:" + chr(10) + history_str if history_str else ""}
Question: {question}
Answer:"""
    return prompt

def generate_answer(question, context_chunks, chat_history=None, model_name=None):
    """Generate a complete answer (non-streaming)."""
    client = genai.Client()
    prompt = _build_prompt(question, context_chunks, chat_history)

    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def generate_answer_stream(question, context_chunks, chat_history=None, model_name=None):
    """Generate answer as a stream of text chunks (for real-time display)."""
    client = genai.Client()
    prompt = _build_prompt(question, context_chunks, chat_history)

    try:
        response = client.models.generate_content_stream(
            model=LLM_MODEL,
            contents=prompt,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error generating answer: {str(e)}"
