# Advanced NLP Framework (T5-based Multi-Task Learning)

## 🚀 Overview
This is an Advanced NLP Framework built on Multi-Task Learning (MTL) using the T5 (Text-to-Text Transfer Transformer) architecture. It supports multiple NLP tasks such as Text Summarization, Translation, and Question Answering in a unified model.

This framework is designed for scalability, efficiency, and real-world adaptability, making it suitable for automation, real-time insights, and large-scale NLP solutions.

## ✨ Features
- 🔥 Unified Multi-Task Learning (MTL): Handles multiple NLP tasks under a single model.
- 🏆 Supports Key NLP Tasks: Text Summarization, Machine Translation, and Question Answering.
- ⚡ Optimized Performance: Fine-tuned on T5-small, T5-base, and T5-large variants.
- 📊 Evaluation Metrics: BLEU, ROUGE, F1 Score** for performance tracking.
- 🛠 Built with: PyTorch, Hugging Face Transformers, and TensorBoard.

## 📌 Installation
To get started, clone the repository and install the required dependencies.

bash
# Clone the repository
git clone https://github.com/rakshithacodes112/advanced-multi-task-learning-using-T5-model.git

cd your-repo

also download the file "https://drive.google.com/file/d/1hIbz1bEJ8zjPCUQcYGtsdiJfHDio-OlL/view?usp=drive_link" and copy the same into the local_t5_model folder

# Create a virtual environment (optional)
python -m venv venv

source venv/bin/activate ( On Windows, use 'venv\Scripts\activate') 

# Install dependencies
pip install -r requirements.txt

## 🚀 Usage
### ⿡ Run the Web App
Start the Flask-based web interface for testing NLP tasks interactively.

bash -   python app.py
Open http://127.0.0.1:5000/ in your browser.

## 📊 Model Training & Fine-Tuning
If you want to fine-tune the model on new datasets:

bash -   python train.py --model_name_or_path t5-base --dataset your_dataset

## 📝 Datasets Used
The model is fine-tuned using:
- CNN/DailyMail (Summarization)
- Opus-100 (Translation)
- SQuAD (Question Answering)

## 📖 Evaluation Metrics
We use the following metrics to evaluate model performance:
- ROUGE Score (Summarization)
- BLEU Score (Translation)
- F1 Score (Question Answering)

## 🛠 Tech Stack
- Backend: Flask, PyTorch, Hugging Face Transformers
- Frontend: HTML, CSS, JavaScript
- Deployment: Docker (optional)

🚀 *Happy Coding!* 🎯
