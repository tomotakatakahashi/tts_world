PACKAGE_DIR := tts_world

JSUT_LABEL_DIR := ../jsut-label/labels/basic5000/
JSUT_WAV_DIR := ../../datasets/jsut_ver1.1/basic5000/wav/


.PHONY: test
test:
	isort $(PACKAGE_DIR)
	black $(PACKAGE_DIR)
	pylint $(PACKAGE_DIR) || exit 0
	pytest $(PACKAGE_DIR) --capture=no
	mypy --disallow-untyped-defs --no-implicit-optional $(PACKAGE_DIR)

generated/duration: $(PACKAGE_DIR) $(JSUT_LABEL_DIR)
	python -m tts_world.preprocess $(JSUT_LABEL_DIR) $@ duration

generated/linguistic: $(PACKAGE_DIR) $(JSUT_LABEL_DIR)
	python -m tts_world.preprocess $(JSUT_LABEL_DIR) $@ linguistic

generated/duration_model.h5: $(PACKAGE_DIR) generated/duration generated/linguistic
	python -m tts_world.train_duration

generated/linguistic_frame: $(PACKAGE_DIR) $(JSUT_LABEL_DIR)
	python -m tts_world.preprocess $(JSUT_LABEL_DIR) $@ linguistic_frame

generated/acoustic: $(PACKAGE_DIR) $(JSUT_WAV_DIR) generated/linguistic_frame
	python -m tts_world.preprocess $(JSUT_WAV_DIR) $@ --extra-dir generated/linguistic_frame acoustic

generated/acoustic_model.h5: $(PACKAGE_DIR) generated/linguistic_frame generated/acoustic
	python -m tts_world.train_acoustic

${DATASET_DIR}/jsut_ver1.1.zip:
	echo "Please download jsut_ver1.1.zip from here. https://sites.google.com/site/shinnosuketakamichi/publication/jsut"

${DATASET_DIR}/jsut_ver1.1/: jsut_ver1.1.zip
	unzip $<

$(JSUT_LABEL_DIR):
	echo "Please clone jsut-label from here. git@github.com:sarulab-speech/jsut-label.git"
	echo "1978271ca6212e1ea742da8f149160f5679e8971"

.PHONY: clean
clean:
	rm -r generated/
