TextGraphs-15 Shared Task on Multi-Hop Inference Explanation Regeneration
=========================================================================

We invite participation in the 3rd Shared Task on Explanation Regeneration associated with the 15th Workshop on Graph-Based Natural Language Processing (TextGraphs 2021).

All systems participating in the shared task will be invited to submit system description papers. Each system description paper will be peer-reviewed by (two) other participating teams and will be presented as a poster at the main workshop: https://sites.google.com/view/textgraphs2021.

Overview
========

This shared task focuses on explanation reconstruction, a stepping-stone towards general multi-hop inference over language. In particular, the inputs to this task consist of questions and their correct answers. Participating systems must extract and rank explanation sentences from a provided structured knowledge base such that the top-ranked sentences provide a complete explanation for the given answer.  This requires combining scientific and common-sense/world knowledge with compositional inference.

While large language models (BERT, XLNet) achieved the highest performance in the 2019 and 2020 shared tasks, substantially advancing the state-of-the-art over previous methods, absolute performance remains modest, highlighting the difficulty of generating detailed explanations through multi-hop reasoning.

*NEW FOR 2021:*
Many-hop multi-hop inference is challenging because there are often multiple ways of assembling a good explanation for a given question.  This 2021 instantiation of the shared task focuses on the theme of determining relevance versus completeness in large multi-hop explanations.  To this end, it includes a very large dataset of approximately 250,000 expert-annotated relevancy ratings of top language model (e.g. BERT, RoBERTa) results that augment the 2020 shared task data. Submissions that evaluate how well existing models designed on 2-hop multihop question answering datasets (e.g. HotPotQA, QASC, etc) perform at many-fact multi-hop explanation regeneration are encouraged.

![Tests](https://github.com/cognitiveailab/tg2021task/workflows/Tests/badge.svg?branch=main)

![Example explanation graph](images/example-girl-eating-apple.jpg)

## Important Dates

* 2021-02-15: Training data release
* 2021-03-10: Test data release; Evaluation start
* 2021-03-24: Evaluation end
* 2021-04-01: System description paper deadline
* 2021-04-13: Deadline for reviews of system description papers
* 2021-04-15: Author notifications
* 2021-04-26: Camera-ready description paper deadline
* 2021-06-11: [TextGraphs-15 workshop](https://sites.google.com/view/textgraphs2021)

Dates are specified in the ISO&nbsp;8601 format.

## Baselines

The shared task data distribution includes a baseline that uses a term frequency model (tf.idf) to rank how likely table row sentences are to be a part of a given explanation. The performance of this baseline on the development partition is 0.513 NDCG.

### Python

```shell
$ make dataset
```

```shell
$ ./baseline_tfidf.py tables wt-expert-ratings.dev.json > predict.txt
```

The format of the `predict.txt` file is `questionID<TAB>explanationID` without header; the order is important. When [tqdm](https://github.com/tqdm/tqdm) is installed, `baseline_tfidf.py` will show a nicely-looking progress bar.

To compute the NDCG for the model, run the following command:

```shell
$ ./evaluate.py --gold wt-expert-ratings.dev.json predict.txt
```
If you want to run the evaluation script without tqdm, adopt the following command:

```shell
$ ./evaluate.py --no-tqdm --gold wt-expert-ratings.dev.json predict.txt
```

In order to prepare a submission file for CodaLab, create a ZIP file containing your `predict.txt` for the *test* dataset, cf. `make predict-tfidf-test.zip`.

## Submission

Please submit your solutions via CodaLab: <https://competitions.codalab.org/competitions/29228>.

## Contacts

This shared task is organized within the 15th workshop on graph-based natural language processing, TextGraphs-15: <https://sites.google.com/view/textgraphs2021>.

We welcome questions and answers on the shared task on CodaLab Forums: <https://competitions.codalab.org/forums/25924/>.

To contact the task organizers directly, please send an email to [textgraphsoc@gmail.com](mailto:textgraphsoc@gmail.com).

## Terms and Conditions

By submitting results to this competition, you consent to the public release of your scores at the TextGraph-15 workshop and in the associated proceedings, at the task organizers' discretion. Scores may include, but are not limited to, automatic and manual quantitative judgements, qualitative judgements, and such other metrics as the task organizers see fit. You accept that the ultimate decision of metric choice and score value is that of the task organizers.

You further agree that the task organizers are under no obligation to release scores and that scores may be withheld if it is the task organizers' judgement that the submission was incomplete, erroneous, deceptive, or violated the letter or spirit of the competition's rules. Inclusion of a submission's scores is not an endorsement of a team or individual's submission, system, or science.

You further agree that your system may be named according to the team name provided at the time of submission, or to a suitable shorthand as determined by the task organizers.

You agree not to use or redistribute the shared task data except in the manner prescribed by its licence.

**To encourage transparency and replicability, all teams must publish their code, tuning procedures, and instructions for running their models with their submission of shared task papers.**

## References

* Jansen, P., Ustalov, D.: [TextGraphs 2020 Shared Task on Multi-Hop Inference for Explanation Regeneration](https://www.aclweb.org/anthology/2020.textgraphs-1.10). In: Proceedings of the Graph-based Methods for Natural Language Processing (TextGraphs). pp. 85&ndash;97. Association for Computational Linguistics, Barcelona, Spain (Online) (2020).

```
@inproceedings{Jansen:20,
  author    = {Jansen, Peter and Ustalov, Dmitry},
  title     = {{TextGraphs~2020 Shared Task on Multi-Hop Inference for Explanation Regeneration}},
  year      = {2020},
  booktitle = {Proceedings of the Graph-based Methods for Natural Language Processing (TextGraphs)},
  pages     = {85--97},
  address   = {Barcelona, Spain (Online)},
  publisher = {Association for Computational Linguistics},
  isbn      = {978-1-952148-42-2},
  url       = {https://www.aclweb.org/anthology/2020.textgraphs-1.10},
  language  = {english},
}
```
