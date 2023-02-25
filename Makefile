local:
	- rm ./dist/*
	poetry build
	pip install dist/*.whl
global:
	poetry publish --build