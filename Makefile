arxiv-data:
	poetry run python arxiv_article_classifier/data/scrape_arxiv.py

interim-data:
	poetry run python arxiv_article_classifier/data/make_interim_data.py

processed-data-tfidf: interim-data
	poetry run python arxiv_article_classifier/data/make_processed_data_tfidf.py

processed-data-distilbert: interim-data
	poetry run python arxiv_article_classifier/data/make_processed_data_distilbert.py

code-lint:
	poetry run isort */*.py */*/*.py tests/*.py
	poetry run black */*.py */*/*.py tests/*.py
	poetry run flake8 */*.py */*/*.py tests/*.py

install:
	poetry install

language-corpora:
	poetry run python -m spacy download en_core_web_sm
	poetry run python -m nltk.downloader stopwords

test:
	poetry run pytest tests/

requirements-file:
	poetry export -f requirements.txt --without-hashes > requirements.txt

