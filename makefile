
# web server
web:
	python src/luna_html_wrapper.py

# normal build
build:
	python src/main.py

# build w clar
cbuild: 
	cls
	python src/main.py

# playground
playground:
	python src/playground.py

# mostly used for debug
eval:
	python src/luna/eval.py

# Specific lint
s:
	pylint src/luna/luna_RL.py