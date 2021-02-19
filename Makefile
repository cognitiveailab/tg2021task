WORLDTREE := tg2021-alldata-practice.zip

evaluate: predict-tfidf-dev.txt
	./evaluate.py --gold=questions.dev.tsv '$<'

submission: predict-tfidf-test.zip

predict-tfidf-%.zip: predict-tfidf-%.txt
	rm -f '$@'
	$(eval TMP := $(shell mktemp -d))
	ln -sf '$(CURDIR)/$<' '$(TMP)/predict.txt'
	zip -j '$@' '$(TMP)/predict.txt'
	rm -rf '$(TMP)'

predict-tfidf-%.txt: data/wt-expert-ratings.%.json
	./baseline_tfidf.py data/tables '$<' > '$@'

dataset: $(WORLDTREE)
	unzip -o '$<'

SHA256 := $(if $(shell which sha256sum),sha256sum,shasum -a 256)

$(WORLDTREE): tg2021task-practice.sha256
	@echo 'Please note that this distribution is subject to the terms set in the license:'
	@echo 'http://cognitiveai.org/explanationbank/'
	curl -sL -o '$@' 'http://cognitiveai.org/dist/$(WORLDTREE)'
	$(SHA256) -c "$<"

checksum:
	$(SHA256) '$(WORLDTREE)' > tg2021task-practice.sha256

clean:
	rm -rfv data/ predict*.txt predict*.zip
