# Project: AI-Powered Therapy Notes Summarizer

"""
This project builds a prototype to summarize therapy session notes using a Large Language Model (LLM).
You’ll learn prompt engineering, evaluation of outputs, logging, and how to build a basic API wrapper.
New features added:
- Sentiment analysis for the note
- JSON structured output
- Confidence scoring (mocked)
"""

# --------------------------
# Step 1: Sample Data Setup
# --------------------------

therapy_notes = [
    """Client reports increased anxiety over the past week, especially in social settings.
    Difficulty sleeping and persistent worry about work performance.
    Reports mild improvement with deep breathing exercises. Plans to attend a social event this weekend.""",

    """Client shared that they had a conflict with a colleague at work.
    Felt overwhelmed but was able to express their needs clearly.
    Reports feeling proud about handling the situation maturely. Continues to work on assertiveness.""",

    """Client feels low energy and lack of motivation. Sleeping more than usual.
    Reports withdrawing from friends and skipping meals. Expressed feelings of hopelessness.
    Therapist introduced basic CBT techniques."""
]

# --------------------------
# Step 2: Prompt Templates
# --------------------------

def create_summary_prompt(note):
    return f"""
    Summarize the following therapy session note using a professional and compassionate tone.
    Focus on clinical observations and next steps:

    Therapy Note:
    {note}

    Summary:
    """

def create_sentiment_prompt(note):
    return f"""
    Analyze the overall emotional tone of the following therapy note.
    Respond with one of: Positive, Neutral, Negative.

    Therapy Note:
    {note}

    Sentiment:
    """

# --------------------------
# Step 3: LLM Integration (OpenAI)
# --------------------------

import openai
import os
import json

# Load your API key from environment variable or a secure source
openai.api_key = os.getenv("OPENAI_API_KEY")

def query_openai(prompt, system_msg="You are a helpful assistant.", temperature=0.4, max_tokens=200):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response["choices"][0]["message"]["content"].strip()

# --------------------------
# Step 4: Run Summarization + Sentiment Analysis
# --------------------------

def analyze_note(note):
    summary_prompt = create_summary_prompt(note)
    sentiment_prompt = create_sentiment_prompt(note)

    summary = query_openai(summary_prompt, system_msg="You summarize therapy notes.")
    sentiment = query_openai(sentiment_prompt, system_msg="You analyze emotional tone.")

    result = {
        "summary": summary,
        "sentiment": sentiment,
        "confidence_score": round(0.85 + 0.1 * (hash(note) % 10) / 10, 2)  # Mocked score
    }
    return result

# --------------------------
# Step 5: Display Output as JSON
# --------------------------

if __name__ == "__main__":
    print("\nTherapy Note Analyses:\n------------------------")
    for i, note in enumerate(therapy_notes):
        result = analyze_note(note)
        print(f"\nNote {i+1} Analysis:\n{json.dumps(result, indent=2)}\n")

# --------------------------
# Step 6 (Optional): Evaluation
# --------------------------
"""
Manual Evaluation Suggestions:
- Does the summary retain the essential clinical details?
- Is it professional, compassionate, and concise?
- Would it be usable in a clinical EHR or report?

Automated Eval Tools:
- Trulens for hallucination/faithfulness checks
- Langfuse for prompt versioning and evals
- Use ROUGE/BLEU scores for comparison (if gold summaries are available)

Future Extensions:
- Store output to a database
- Add UI using Streamlit or Flask
- Use Langfuse/Trulens for logging and dashboard
- Incorporate real feedback from clinicians for model fine-tuning

Here are several technical aspects of your Therapy Notes Summarizer project that are important both from an engineering and applied AI perspective:
1. 🧠 Prompt Engineering
You’ve defined two prompts:
One for summary generation in a clinical tone
One for sentiment classification
Key Details:
Prompts are structured and contextualized to nudge the LLM toward task-specific outputs.
The system message and temperature are adjusted to balance creativity and factual accuracy (temperature=0.4 is conservative for summarization).
🛠️ You could further improve by experimenting with few-shot examples or role-based prompts ("You're a licensed clinical therapist...").
2. 🔁 Function Abstraction with query_openai
This reusable function abstracts:
API calls to openai.ChatCompletion
Prompt formatting
Token limit and temperature control
Importance:
Clean separation of LLM interface logic
Easy to extend to other providers (Claude, Mistral, etc.)
Central place to handle retries, timeouts, logging
3. 📊 Multi-Task Output: Summary + Sentiment
Instead of a single task, you run two different LLM workflows:
This simulates a multi-head AI system: One for summarization, another for classification
You return structured JSON, making it easily consumable by a front-end or API
🧩 This structure aligns with real-world AI microservices where each task is modular.
4. 🧪 Confidence Score (Mocked)
"confidence_score": round(0.85 + 0.1 * (hash(note) % 10) / 10, 2)
Though mocked, this introduces the concept of uncertainty or reliability. In production:
You’d replace this with metrics from model confidence calibration (e.g., log-likelihood)
Or human-verified confidence scores via HITL evaluation
5. 📦 Deployment-Ready Structure
You’ve designed the code to:
Accept batch input (multiple notes)
Output JSON summaries
Run from CLI via if __name__ == "__main__" block
💡 Makes it easy to wrap this into a REST API, CLI tool, or cloud job.
6. 🧹 Evaluation Readiness
Your comments in Step 6 suggest:
Manual review criteria (fidelity, tone, usability)
ROUGE/BLEU if you build a labeled dataset
Integration potential with tools like:
Trulens for hallucination tracing
Langfuse for prompt versioning, input/output monitoring
🧠 This shows awareness of ML evaluation challenges—key for production AI systems.
7. 🔐 Security Best Practices
openai.api_key = os.getenv("OPENAI_API_KEY")
You're loading secrets from environment variables, which is:
Best practice for security and portability
Easy to integrate with platforms like GitHub Actions, AWS Lambda, or Docker containers
Would you like a diagram showing the full workflow architecture next? Or a suggestion on how to extend this to a Streamlit app or AWS Lambda deployment?


┌──────────────────────────┐
│   Therapy Notes (Text)   │
└────────────┬─────────────┘
             │
             ▼
 ┌─────────────────────────┐
 │   Prompt Engineering    │◄────────────┐
 │ (Summary + Sentiment)   │             │
 └────────────┬────────────┘             │
              ▼                          │
 ┌─────────────────────────────┐         │
 │      query_openai()        │         │
 │  - Uses OpenAI Chat API     │         │
 │  - System + User prompts    │         │
 └────────────┬────────────────┘         │
              ▼                          │
 ┌─────────────────────────────┐         │
 │      Parsed LLM Output      │         │
 │  - Summary                  │         │
 │  - Sentiment (Pos/Neu/Neg) │         │
 │  - Confidence Score (mock) │         │
 └────────────┬────────────────┘         │
              ▼                          │
   ┌──────────────────────────────┐      │
   │     JSON Output Format       │      │
   └────────────┬─────────────────┘      │
                ▼                        │
     ┌────────────────────────┐          │
     │   CLI / Console Print  │          │
     └────────────────────────┘          │
                │                        │
                ▼                        ▼
     ┌─────────────────────────────┐     ┌────────────────────────────┐
     │      Manual Evaluation      │     │  Future: Langfuse / Trulens│
     └─────────────────────────────┘     └────────────────────────────┘

"""

