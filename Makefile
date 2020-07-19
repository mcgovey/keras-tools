# setup:
# 	python3 -m venv ~/.myrepo

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
		
test:
	# python3 -m pytest -vv --cov=myrepolib tests/*.py -W ignore::ResourceWarning 
	pytest --cov='.' tests/*.py 
	
lint:
	pylint --disable=R,C KerasTools/__init__.py
	
all: install lint test