
# web server
web:
	python src/luna_html_wrapper.py

# normal build
build:
	python src/main.py

# infinite train
train:
	python src/train.py

# mostly used for debug
eval:
	python src/eval.py