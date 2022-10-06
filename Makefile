PACKAGE_DIR := tts_world

.PHONY: test
test:
	isort $(PACKAGE_DIR)
	black $(PACKAGE_DIR)
	pylint $(PACKAGE_DIR) || exit 0
	pytest $(PACKAGE_DIR) --capture=no
	mypy --disallow-untyped-defs --no-implicit-optional $(PACKAGE_DIR)

${DATASET_DIR}/jsut_ver1.1.zip:
	echo "Please download jsut_ver1.1.zip from here. https://sites.google.com/site/shinnosuketakamichi/publication/jsut"

${DATASET_DIR}/jsut_ver1.1/: jsut_ver1.1.zip
	unzip $<

../jsut-label/:
	echo "Please clone jsut-label from here. git@github.com:sarulab-speech/jsut-label.git"
	echo "1978271ca6212e1ea742da8f149160f5679e8971"

.PHONY: clean
clean:
