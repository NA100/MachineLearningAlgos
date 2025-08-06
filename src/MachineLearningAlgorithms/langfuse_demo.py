"""
Langfuse is an open-source LLM observability and evaluation tool. It helps you track, debug, and improve your AI application's performance in production or during development.
Since your project (Therapy Notes Summarizer) uses LLMs like GPT-4, Langfuse can be especially useful for:

🧠 What Does Langfuse Do?
Capability	Explanation
🔍 Prompt + Response Logging	Automatically tracks prompts, responses, model versions, timing, and user feedback.
🧪 Evaluation & Scoring	Supports custom evaluation functions and human feedback (e.g., accuracy, tone).
🧭 Traceability & Debugging	Helps trace what prompt generated a bad result, so you can iterate easily.
🔁 Prompt Versioning	Track changes to your prompts over time and see which version performs best.
📊 Dashboard & Analytics	Visual dashboards to inspect inputs, outputs, latency, errors, etc.
🧩 Integrations	Works with OpenAI, LangChain, custom Python apps, FastAPI, Streamlit, etc.
"""

from langfuse import Langfuse

langfuse = Langfuse(public_key="your-public-key", secret_key="your-secret-key")

def query_openai_logged(prompt, system_msg="You are a helpful assistant.", temperature=0.4, max_tokens=200):
    trace = langfuse.trace(name="therapy_summarizer")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    output = response["choices"][0]["message"]["content"].strip()

    trace.log_input_output(input=prompt, output=output)

    return output
