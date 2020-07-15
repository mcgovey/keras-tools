# setup:
# 	python3 -m venv ~/.myrepo

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
		
test:
	# python3 -m pytest -vv --cov=myrepolib tests/*.py
	python3 -m pytest -W ignore::ResourceWarning tests/*.py
	
lint:
	pylint --disable=R,C keras-rnn-tools/__init__.py
	
all: install lint test