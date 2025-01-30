from flask import Flask, render_template, request, jsonify, session
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import re
from collections import Counter

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session handling

# Load the fine-tuned T5 model and tokenizer
MODEL_PATH = r"E:\advanced-multi-task-learning-using-T5-model\local_t5_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# Function to analyze input and generate keywords
def analyze_summary(input_text):
    """Generate a brief, meaningful summary phrase of the input text."""
    words = re.findall(r'\b\w+\b', input_text.lower())

    stopwords = {
        "a", "an", "the", "is", "of", "and", "to", "in", "on", "with",
        "for", "about", "as", "has", "this", "that", "it", "at", "by", "from"
    }

    # Count word occurrences excluding stopwords
    word_counts = Counter(word for word in words if word not in stopwords)

    # Extract the 2-3 most frequent words to form a topic phrase
    important_words = [word.capitalize() for word, _ in word_counts.most_common(2)]

    return " ".join(important_words) if important_words else "General Topic"

app.jinja_env.filters["analyze_summary"] = analyze_summary

@app.route("/")
def index():
    user_history = session.get("history", [])
    return render_template("index.html", user_history=user_history)

@app.route("/process_input", methods=["POST"])
def process_input():
    task = request.form.get("task")
    output = ""

    try:
        if task == "summarization":
            text = request.form.get("text")
            if not text:
                return jsonify({"error": "Please provide text for summarization."}), 400
            output = summarize_text(text)

        elif task == "translation":
            text = request.form.get("text")
            if not text:
                return jsonify({"error": "Please provide text for translation."}), 400
            output = translate_text(text)

        elif task == "qa":
            question = request.form.get("question")
            context = request.form.get("context")
            if not question or not context:
                return jsonify({"error": "Please provide both a question and context for QA."}), 400
            output = answer_question(question, context)

        else:
            return jsonify({"error": "Invalid task type."}), 400

        # Store history in session
        history = session.get("history", [])
        history.insert(0, {
            "task": task.capitalize(),
            "input": analyze_summary(text if task != "qa" else f"Q: {question} | C: {context}"),
            "output": output
        })
        session["history"] = history
        session.modified = True

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"output": output})

# Model Inference Functions
def summarize_text(text):
    inputs = tokenizer(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, min_length=50, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_text(text):
    inputs = tokenizer(f"translate English to French: {text}", return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=200, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def answer_question(question, context):
    inputs = tokenizer(f"question: {question} context: {context}", return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    app.run(debug=True)
