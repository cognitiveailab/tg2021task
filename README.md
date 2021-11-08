TextGraphs-15 Shared Task on Multi-Hop Inference Explanation Regeneration
=========================================================================

We invite participation in the 3rd Shared Task on Explanation Regeneration associated with the 15th Workshop on Graph-Based Natural Language Processing (TextGraphs 2021).

All systems participating in the shared task will be invited to submit system description papers. Each system description paper will be peer-reviewed by (two) other participating teams and will be presented as a poster at the main workshop: https://sites.google.com/view/textgraphs2021.

![Tests](https://github.com/cognitiveailab/tg2021task/workflows/Tests/badge.svg?branch=main)

## Overview

Multi-hop inference is the task of combining more than one piece of information to solve an inference task, such as question answering.  This can take many forms, from combining free-text sentences read from books or the web, to combining linked facts from a structured knowledge base.  The Shared Task on Explanation Regeneration asks participants to develop methods that reconstruct large explanations for science questions, using a corpus of gold explanations that provides supervision and instrumentation for this multi-hop inference task.  Each explanation is represented as an "explanation graph", a set of atomic facts (between 1 and 16 per explanation, drawn from a knowledge base of 9,000 facts) that, together, form a detailed explanation for the reasoning required to answer and explain the resoning behind a question.

Explanation Regeneration is a stepping-stone towards general multi-hop inference over language.  In this shared task we frame explanation regeneration as a ranking task, where the inputs to a given system consist of questions and their correct answers. Participating systems must then rank atomic facts from a provided semi-structured knowledge base such that the combination of top-ranked facts provide a detailed explanation for the answer.  This requires combining scientific and common-sense/world knowledge with compositional inference.

While large language models (BERT, ERNIE) achieved the highest performance in the [2019](https://www.aclweb.org/anthology/D19-5309/) and [2020](https://www.aclweb.org/anthology/2020.textgraphs-1.10/) shared tasks, substantially advancing the state-of-the-art over previous methods, absolute performance remains modest, highlighting the difficulty of generating detailed explanations through multi-hop reasoning.

### New for 2021

Many-hop multi-hop inference is challenging because there are often multiple ways of assembling a good explanation for a given question.  This 2021 instantiation of the shared task focuses on the theme of determining relevance versus completeness in large multi-hop explanations.  To this end, this year we include a very large dataset of approximately 250,000 expert-annotated relevancy ratings for facts ranked highly by baseline language models from previous years (e.g. BERT, RoBERTa).

Submissions using a variety of methods (graph-based or otherwise) are encouraged.  Submissions that evaluate how well existing models designed on 2-hop multihop question answering datasets (e.g. HotPotQA, QASC, etc) perform at many-fact multi-hop explanation regeneration are welcome.

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

## Data
The data used in this shared task contains approximately 5,100 science exam questions drawn from the AI2 Reasoning Challenge (ARC) dataset ([Clark et al., 2018](https://allenai.org/data/arc)), together with multi-fact explanations  for their correct answers drawn from the WorldTree V2.1 explanation corpus ([Xie et al., 2020](https://www.aclweb.org/anthology/2020.lrec-1.671/), [Jansen et al., 2018](https://www.aclweb.org/anthology/L18-1433/)).  For 2021, this has been augmented with a new set of approximately 250k pre-release expert-generated relevancy ratings.

The knowledge base supporting these questions and their explanations contains approximately 9,000 facts. To encourage a variety of solving methods, the knowledge base is available both as plain-text sentences (unstructured) as well as semi-structured tables. Facts are a combination of scientific knowledge as well as common-sense/world knowledge.

The full dataset (WorldTree V2.1  Relevancy Ratings) can be downloaded at the following links:
* Practice data (train + dev): <http://www.cognitiveai.org/dist/tg2021-alldata-practice.zip>
* Evaluation period data (train + dev + test): <http://www.cognitiveai.org/dist/tg2021-alldata-evalperiod.zip>  NOW AVAILABLE!


More information about the WorldTree V2.1 corpus, including a book of explanation graphs, can be found [here](http://cognitiveai.org/explanationbank/).

## Baselines

The shared task data distribution includes a baseline that uses a term frequency model (tf.idf) to rank how likely table row sentences are to be a part of a given explanation. The performance of this baseline on the development partition is 0.513 NDCG.

### Python

First, get the data using `make dataset` or the following sequence of commands:

```shell
$ wget cognitiveai.org/dist/tg2021-alldata-practice.zip
$ unzip tg2021-alldata-practice.zip
```

Then, run the baseline

```shell
$ ./baseline_tfidf.py data/tables data/wt-expert-ratings.dev.json > predict.txt
```

The format of the `predict.txt` file is `questionID<TAB>explanationID` without header; the order is important. When [tqdm](https://github.com/tqdm/tqdm) is installed, `baseline_tfidf.py` will show a nicely-looking progress bar.

To compute the NDCG for the model, run the following command:

```shell
$ ./evaluate.py --gold data/wt-expert-ratings.dev.json predict.txt
```
If you want to run the evaluation script without tqdm, adopt the following command:

```shell
$ ./evaluate.py --no-tqdm --gold data/wt-expert-ratings.dev.json predict.txt
```

In order to prepare a submission file for CodaLab, create a ZIP file containing your `predict.txt`.

Dataset sample for the Practice phase can be created with `make predict-tfidf-dev.zip` using the *dev* dataset, while the one for the Evaluation phase can be created with `make predict-tfidf-test.zip` using the *test* dataset.

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

* Thayaparan, M. et al.: [TextGraphs 2021 Shared Task on Multi-Hop Inference for Explanation Regeneration](https://doi.org/10.18653/v1/2021.textgraphs-1.17). In: Proceedings of the Fifteenth Workshop on Graph-Based Methods for Natural Language Processing (TextGraphs-15). pp. 156&ndash;165. Association for Computational Linguistics, Mexico City, Mexico (2021).

```
@inproceedings{Thayaparan:21,
  author    = {Thayaparan, Mokanarangan and Valentino, Marco and Jansen, Peter and Ustalov, Dmitry},
  title     = {{TextGraphs~2021 Shared Task on Multi-Hop Inference for Explanation Regeneration}},
  year      = {2021},
  booktitle = {Proceedings of the Fifteenth Workshop on Graph-Based Methods for Natural Language Processing (TextGraphs-15)},
  pages     = {156--165},
  address   = {Mexico City, Mexico},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2021.textgraphs-1.17},
  isbn      = {978-1-954085-38-1},
  language  = {english},
}
```
