WORLDTREE := tg2021task-wt-expert-ratings.initial.zip

evaluate: predict-tfidf-dev.txt
	./evaluate.py --gold=questions.dev.tsv $<

submission: predict-tfidf-test.zip

predict-tfidf-%.zip: predict-tfidf-%.txt
	rm -f $@
	$(eval TMP := $(shell mktemp -d))
	ln -sf $(CURDIR)/$< $(TMP)/predict.txt
	zip -j $@ $(TMP)/predict.txt
	rm -rf $(TMP)

predict-tfidf-%.txt: wt-expert-ratings.train.json
	./baseline_tfidf.py tables $< > $@

dataset: $(WORLDTREE)
	unzip -o $<

SHA256SUM := $(shell type -p sha256sum || echo shasum -a 256)

$(WORLDTREE): tg2021task-practice.sha256
	@echo 'Please note that this distribution is subject to the terms set in the license:'
	@echo 'http://cognitiveai.org/explanationbank/'
	curl -sL -o "$@" 'http://cognitiveai.org/dist/$(WORLDTREE)'
	$(SHA256SUM) -c "$<"

checksum:
	$(SHA256SUM) "$(WORLDTREE)" > tg2021task-practice.sha256

clean:
	rm -fv predict*.txt predict*.zip
