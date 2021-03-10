WORLDTREE := tg2021-alldata-evalperiod.zip

evaluate: predict-tfidf-dev.txt
	./evaluate.py --gold=data-evalperiod/wt-expert-ratings.dev.json '$<'

submission: predict-tfidf-test.zip

predict-tfidf-%.zip: predict-tfidf-%.txt
	rm -f '$@'
	$(eval TMP := $(shell mktemp -d))
	ln -sf '$(CURDIR)/$<' '$(TMP)/predict.txt'
	zip -j '$@' '$(TMP)/predict.txt'
	rm -rf '$(TMP)'

predict-tfidf-%.txt: data-evalperiod/wt-expert-ratings.%.json
	./baseline_tfidf.py data-evalperiod/tables '$<' > '$@'

dataset: $(WORLDTREE)
	unzip -o '$<'

SHA256 := $(if $(shell which sha256sum),sha256sum,shasum -a 256)

$(WORLDTREE): tg2021task-evalperiod.sha256
	@echo 'Please note that this distribution is subject to the terms set in the license:'
	@echo 'http://cognitiveai.org/explanationbank/'
	curl -sL -o '$@' 'http://cognitiveai.org/dist/$(WORLDTREE)'
	$(SHA256) -c "$<"

checksum:
	$(SHA256) '$(WORLDTREE)' > tg2021task-evalperiod.sha256

clean:
	rm -rfv data/ data-evalperiod/ predict*.txt predict*.zip
