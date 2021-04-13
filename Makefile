# setup:
# 	python3 -m venv ~/.myrepo

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt --upgrade
		
test:
	# python3 -m pytest -vv --cov=myrepolib tests/*.py -W ignore::ResourceWarning 
	pytest --cov=KerasTools tests/test_*.py 
	
lint:
	pylint --disable=R,C KerasTools/__init__.py
	
all: install lint test