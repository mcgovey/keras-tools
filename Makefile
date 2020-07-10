# setup:
# 	python3 -m venv ~/.myrepo

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
		
test:
	# python3 -m pytest -vv --cov=myrepolib tests/*.py
	
lint:
	pylint --disable=R,C keras-rnn-tools/base-test.py
	
all: install lint test