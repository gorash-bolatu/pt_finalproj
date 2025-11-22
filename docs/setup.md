# How to set up the development environment for the **AI-Powered Sentiment Analysis & Recommendation System** project

## **1. System Requirements**

* Python **3.9+**
* pip (latest)
* Git
* MySQL Server 8.x
* (Optional) MySQL Workbench or any SQL client
* ~5–10 GB free storage

## **2. Project Directory Structure**

Create the following structure:

```
project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── app/               # Django project
├── src/               # ML + preprocessing code
├── tests/
├── docs/
└── requirements.txt
```

## **3. Python Environment**

### Install dependencies

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn tensorflow langdetect spacy nltk joblib django mysqlclient sqlalchemy jupyterlab pytest
```

### Save dependency list

```bash
pip freeze > requirements.txt
```

## **4. NLP Resources**

### Install spaCy models

```bash
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

### Install NLTK resources

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## **5. MySQL Setup**

1. Install and start MySQL Server.
2. Create a database (example name):

   ```sql
   CREATE DATABASE sentiment_db;
   ```

3. Create a user and grant privileges:

   ```sql
   CREATE USER 'sentiment_user'@'%' IDENTIFIED BY 'your_password';
   GRANT ALL PRIVILEGES ON sentiment_db.* TO 'sentiment_user'@'%';
   FLUSH PRIVILEGES;
   ```

4. Test connection from terminal or SQL client.
5. Add credentials to Django's `settings.py`:

   ```python
   DATABASES = {
       'default': {
           'ENGINE': 'django.db.backends.mysql',
           'NAME': 'sentiment_db',
           'USER': 'sentiment_user',
           'PASSWORD': 'your_password',
           'HOST': 'localhost',
           'PORT': '3306',
       }
   }
   ```

## **6. Django Project Setup**

### Create Django project

```bash
django-admin startproject sentiment_app
cd sentiment_app
python manage.py startapp reviews
```

### Run initial migrations

```bash
python manage.py migrate
```

## **7. Jupyter Notebook Setup**

Launch notebook environment:

```bash
jupyter lab
```

Use Jupyter for:

* data exploration
* preprocessing
* model training/testing

## **8. Version Control Setup**

### Initialize Git repository

```bash
git init
```

### Add `.gitignore` (recommended includes)

```
venv/
__pycache__/
*.pyc
*.pkl
data/raw/
data/processed/
models/
.ipynb_checkpoints/
```

### Initial commit

```bash
git add .
git commit -m "Initial project setup"
```
