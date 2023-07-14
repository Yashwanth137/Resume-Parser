import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize each sentence into words and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokenized_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word.lower() for word in words if word.lower() not in stop_words]
        tokenized_sentences.append(words)
    
    return tokenized_sentences

def extract_skills(tokenized_sentences, skills):
    matched_skills = []
    
    for sentence in tokenized_sentences:
        for skill in skills:
            if skill.lower() in sentence:
                matched_skills.append(skill)
    
    return matched_skills

# Sample resumes and their corresponding labels
resumes = [
    ("John Doe is an experienced software engineer proficient in Python and machine learning.", "IT"),
    ("Sarah Smith has a background in marketing and data analysis.", "Marketing"),
    ("Tom Johnson is a skilled project manager with experience in agile methodologies.", "Project Management"),
    ("Lisa Brown is a qualified nurse with expertise in patient care and medical documentation.", "Healthcare"),
]

# Required skills for each job category
required_skills = {
    "IT": ['Python', 'Machine Learning'],
    "Marketing": ['Data Analysis'],
    "Project Management": ['Agile Methodologies'],
    "Healthcare": ['Patient Care', 'Medical Documentation'],
}

# Preprocess the resumes and extract matched skills
tokenized_resumes = [preprocess_text(resume) for resume, _ in resumes]
matched_skills = [extract_skills(tokenized_resumes, required_skills[label]) for _, label in resumes]

# Prepare training data
X = [' '.join(sentence) for tokenized_resume in tokenized_resumes for sentence in tokenized_resume]
y = [label for _, label in resumes]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline (TF-IDF vectorizer and Support Vector Machine classifier)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear')),
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
accuracy = pipeline.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Predict job categories for new resumes
new_resumes = [
    "Emily Wilson is a software developer with expertise in Java and web development.",
    "Michael Thompson has a background in finance and financial analysis.",
]
predicted_labels = pipeline.predict(new_resumes)

print("Predicted Labels:", predicted_labels)
