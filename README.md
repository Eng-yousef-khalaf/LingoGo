# 📊 LingoGo - Smart Vocabulary Performance Analyzer

LingoGo is a data-driven language learning assistant that leverages **machine learning** and **user performance data** to analyze and improve vocabulary retention. The project focuses on transforming raw learning interaction data into insightful analytics using modern data science techniques.

## 🚀 Project Overview

This project simulates a smart evaluation system that tracks how users interact with language learning material—such as presentations, quizzes, and live sessions—and uses a machine learning model to predict their overall performance and suggest improvements.

The core idea was to apply **real-world data science skills** to a practical problem in **edtech** (educational technology), combining Python, scikit-learn, and regression modeling to build a personalized analysis system.

## 🧠 Problem Statement

Language learners interact with multiple forms of content—slides, quizzes, interactive games—and receive little to no feedback on how well they are doing. This project answers:

- How can we use performance data to understand a learner’s strengths and weaknesses?
- Can we predict their learning outcomes?
- How can we provide smart, adaptive feedback to help learners succeed?

## 🔍 Data & Features

The model was trained on synthetic data representing:
- 📚 `num_presentations`: Number of learning sessions presented.
- 🧩 `num_levels`: Levels completed.
- 📝 `total_quiz_score` and `avg_quiz_score`: Total and average performance on quizzes.
- 🖼️ `num_slides`: Number of visual slides reviewed.
- 🌐 `unique_topics`: Number of unique topics presented.
- 🔁 `language_pair`: Target language group (e.g., English↔Arabic).

The target variable was the learner's **overall performance score**.

## 🧪 Model

The final model was trained using:
- ✅ `RandomForestRegressor` from scikit-learn.
- 📈 The model demonstrated reliable performance in estimating a user’s learning effectiveness based on input features.
- 🧠 Pickle was used to serialize and save the model for later deployment and integration.

## 💡 Key Data Science Concepts Applied

- Feature Engineering  
- Regression Modeling  
- Model Evaluation  
- Serialization with Pickle  
- DataFrame manipulation with Pandas

## 📦 Tech Stack

- **Python** 🐍  
- **Pandas** for data handling  
- **Scikit-learn** for machine learning  
- **Jupyter Notebook** for experimentation and visualization

## 🎯 Future Work

- Integrate real-time performance tracking from the frontend.  
- Build a REST API using Flask or FastAPI to serve predictions.  
- Improve model accuracy using advanced techniques like Gradient Boosting or Neural Networks.  
- Deploy the system within a live educational web application (React + Node.js).

## 🤝 Author

**Yousef Khalaf**  
Software Engineering Student | Data Science Enthusiast  
🔍 Specialized in building intelligent data-driven systems for real-world applications.
