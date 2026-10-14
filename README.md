# AI Interview Coach

A simple local AI Interview Coach built with Python and Streamlit.

## Features

- Job roles: Python Developer, Data Analyst, AI/ML Engineer
- Interview types: Technical, HR, Mixed
- Difficulty: Beginner, Intermediate, Advanced
- Five-question interview session
- Local question bank stored in JSON
- NLP-based answer evaluation
- Keyword matching
- TF-IDF cosine similarity
- Score out of 10
- Matched and missing concepts
- Final performance summary
- No OpenAI/Gemini/external API required

## Technologies

- Python
- Streamlit
- scikit-learn
- NumPy
- JSON

## Project structure

```text
AI Interview Coach/
├── app.py
├── questions.json
├── requirements.txt
└── README.md
```

## Run the project

Open PowerShell in this folder and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If `python` is not recognized, try:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## How evaluation works

The app uses two local NLP signals:

1. Keyword matching checks whether important concepts appear in the user's answer.
2. TF-IDF cosine similarity compares the user's answer with a reference answer.

The final score combines both signals.

## Note

This is an educational PBL project. The evaluation is intentionally simple and is not intended to replace a human interviewer.
