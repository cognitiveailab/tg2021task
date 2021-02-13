TextGraphs-14 Shared Task on Multi-Hop Inference Explanation Regeneration
=========================================================================

Multi-hop inference is the task of combining more than one piece of information to solve an inference task, such as question answering.

![Example explanation graph](images/example-girl-eating-apple.jpg)

## Important Dates

* 2020-03-06: Training data release
* 2020-04-06: Test data release; Evaluation start
* 2020-09-21: Evaluation end
* 2020-10-02: System description paper deadline
* 2020-10-18: Deadline for reviews of system description papers
* 2020-10-25: Author notifications
* 2020-11-01: Camera-ready description paper deadline
* 2020-12-13: [TextGraphs-14 workshop](https://sites.google.com/view/textgraphs2020)

Dates are specified in the ISO&nbsp;8601 format.

## Baselines

The shared task data distribution includes a baseline that uses a term frequency model (tf.idf) to rank how likely table row sentences are to be a part of a given explanation. The performance of this baseline on the development partition is 0.255 MAP.

### Python

```shell
$ make dataset
```

```shell
$ ./baseline_tfidf.py tables questions.dev.tsv > predict.txt
```

The format of the `predict.txt` file is `questionID<TAB>explanationID` without header; the order is important. When [tqdm](https://github.com/tqdm/tqdm) is installed, `baseline_tfidf.py` will show a nicely-looking progress bar.

```shell
$ ./evaluate.py questions.dev.tsv predict.txt
```

In order to prepare a submission file for CodaLab, create a ZIP file containing your `predict.txt` for the *test* dataset, cf. `make predict-tfidf-test.zip`.

## Submission

Please submit your solutions via CodaLab: <https://competitions.codalab.org/competitions/23615>.

## Contacts

This shared task is organized within the 14th workshop on graph-based natural language processing, TextGraphs-14: <https://sites.google.com/view/textgraphs2020>.

We welcome questions and answers on the shared task on CodaLab Forums: <https://competitions.codalab.org/forums/20311/>.

To contact the task organizers directly, please send an email to [textgraphsoc@gmail.com](mailto:textgraphsoc@gmail.com).

## Terms and Conditions

By submitting results to this competition, you consent to the public release of your scores at the TextGraph-14 workshop and in the associated proceedings, at the task organizers' discretion. Scores may include, but are not limited to, automatic and manual quantitative judgements, qualitative judgements, and such other metrics as the task organizers see fit. You accept that the ultimate decision of metric choice and score value is that of the task organizers.

You further agree that the task organizers are under no obligation to release scores and that scores may be withheld if it is the task organizers' judgement that the submission was incomplete, erroneous, deceptive, or violated the letter or spirit of the competition's rules. Inclusion of a submission's scores is not an endorsement of a team or individual's submission, system, or science.

You further agree that your system may be named according to the team name provided at the time of submission, or to a suitable shorthand as determined by the task organizers.

You agree not to use or redistribute the shared task data except in the manner prescribed by its licence.

**To encourage transparency and replicability, all teams must publish their code, tuning procedures, and instructions for running their models with their submission of shared task papers.**

## References

* Jansen P. and Ustalov D. [TextGraphs 2019 Shared Task on Multi-Hop Inference for Explanation Regeneration](https://doi.org/10.18653/v1/D19-5309). *Proceedings of the Thirteenth Workshop on Graph-Based Methods for Natural Language Processing (TextGraphs-13).* Hong Kong: Association for Computational Linguistics, 2019, pp. 63&ndash;77.

```
@inproceedings{Jansen:19,
  author    = {Jansen, Peter and Ustalov, Dmitry},
  title     = {{TextGraphs~2019 Shared Task on Multi-Hop Inference for Explanation Regeneration}},
  booktitle = {Proceedings of the Thirteenth Workshop on Graph-Based Methods for Natural Language Processing (TextGraphs-13)},
  year      = {2019},
  pages     = {63--77},
  doi       = {10.18653/v1/D19-5309},
  isbn      = {978-1-950737-86-4},
  address   = {Hong Kong},
  publisher = {Association for Computational Linguistics},
  language  = {english},
}
```
