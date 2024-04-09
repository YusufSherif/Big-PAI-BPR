*(See [GitHub - WorksApplications/chiVe](https://github.com/WorksApplications/chiVe) for the latest README)*


# chiVe: Japanese Word Embedding with Sudachi & NWJC

[日本語 README](#chive-sudachiとnwjcによる日本語単語ベクトル)

"chiVe" (Suda**chi Ve**ctor) is a Japanese pre-trained word embedding resource using large-scale corpus and multi-granular tokenization.

## Abstract

Based on the [skip-gram algorithm](https://arxiv.org/abs/1301.3781), we used word2vec ([gensim](https://radimrehurek.com/gensim/)) to train the vectors.

We used [NINJAL Web Japanese Corpus (NWJC)](https://pj.ninjal.ac.jp/corpus_center/nwjc/) from National Institute for Japanese Language and Linguistics which contains around 100 million web page text as a training corpus, and used [Sudachi](https://github.com/WorksApplications/Sudachi) by Works Applications for tokenization.

We used Sudachi's multi-granular tokenziation results (short, mid, and named entity) of NWJC text to train word vectors.

## Licence

Copyright (c) 2020 National Institute for Japanese Language and Linguistics and Works Applications Co., Ltd. All rights reserved.

"chiVe" is distributed by [National Institute for Japanese Langauge and Linguistics](https://www.ninjal.ac.jp/) and [Works Applications Co.,Ltd.](https://www.worksap.co.jp/) under [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Slack

We have a Slack workspace for developers and users to ask questions and discuss a variety of topics.

- https://sudachi-dev.slack.com/
- (Please get an invite from [here](https://join.slack.com/t/sudachi-dev/shared_invite/enQtMzg2NTI2NjYxNTUyLTMyYmNkZWQ0Y2E5NmQxMTI3ZGM3NDU0NzU4NGE1Y2UwYTVmNTViYjJmNDI0MWZiYTg4ODNmMzgxYTQ3ZmI2OWU))


## Citing chiVe

We have published a following paper about chiVe;

- 真鍋陽俊, 岡照晃, 海川祥毅, 髙岡一馬, 内田佳孝, 浅原正幸. [複数粒度の分割結果に基づく日本語単語分散表現](https://www.anlp.jp/proceedings/annual_meeting/2019/pdf_dir/P8-5.pdf) *(Japanese Word Embedding based on Multi-granular Tokenization Results, in Japanese)*. 言語処理学会第25回年次大会, 2019.
- 河村宗一郎, 久本空海, 真鍋陽俊, 高岡一馬, 内田佳孝, 岡照晃, 浅原正幸. [chiVe 2.0: SudachiとNWJCを用いた実用的な日本語単語ベクトルの実現へ向けて](https://www.anlp.jp/nlp2020/) *(chiVe 2.0: Towards Prctical Japanese Embedding wiht Sudachi and NWJC, in Japanese)*. 言語処理学会第26回年次大会, 2020.

When citing chiVe in papers, books, or services, please use the follow BibTex entries;

```
@INPROCEEDINGS{manabe2019chive,
    author    = {真鍋陽俊, 岡照晃, 海川祥毅, 髙岡一馬, 内田佳孝, 浅原正幸},
    title     = {複数粒度の分割結果に基づく日本語単語分散表現},
    booktitle = "言語処理学会第25回年次大会(NLP2019)",
    year      = "2019",
    pages     = "NLP2019-P8-5",
    publisher = "言語処理学会",
}
```

```
@INPROCEEDINGS{kawamura2020chive,
    author    = {河村宗一郎, 久本空海, 真鍋陽俊, 高岡一馬, 内田佳孝, 岡照晃, 浅原正幸},
    title     = {chiVe 2.0: SudachiとNWJCを用いた実用的な日本語単語ベクトルの実現へ向けて},
    booktitle = "言語処理学会第26回年次大会(NLP2020)",
    year      = "2020",
    pages     = "NLP2020-P6-16",
    publisher = "言語処理学会",
}
```


***


*（最新のREADMEは [GitHub - WorksApplications/chiVe](https://github.com/WorksApplications/chiVe) を参照してください）*


# chiVe: SudachiとNWJCによる日本語単語ベクトル

[English README](#chive-japanese-word-embedding-with-sudachi--nwjc)

"chiVe" (チャイブ, Suda**chi Vec**tor) は、大規模コーパスと複数粒度分割に基づく日本語単語ベクトルです。

## 概要

[Skip-gramアルゴリズム](https://arxiv.org/abs/1301.3781)を元に、word2vec （[gensim](https://radimrehurek.com/gensim/)） を使用して単語分散表現を構築しています。

学習には約1億のウェブページ文章を含む国立国語研究所の[日本語ウェブコーパス（NWJC）](https://pj.ninjal.ac.jp/corpus_center/nwjc/)を採用し、分かち書きにはワークスアプリケーションズの形態素解析器[Sudachi](https://github.com/WorksApplications/Sudachi)を使用しています。

Sudachiで定義されている短・中・長単位の3つの分割単位でNWJCを解析した結果を元に分散表現の学習を行なっています。

## ライセンス

Copyright (c) 2020 National Institute for Japanese Language and Linguistics and Works Applications Co., Ltd. All rights reserved.

[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)の下で[国立国語研究所](https://www.ninjal.ac.jp/)と[株式会社ワークスアプリケーションズ](https://www.worksap.co.jp/)によって提供されています。

## Slack

開発者やユーザーの方々が質問したり議論するためのSlackワークスペースを用意しています。

- https://sudachi-dev.slack.com/
- ([こちら](https://join.slack.com/t/sudachi-dev/shared_invite/enQtMzg2NTI2NjYxNTUyLTMyYmNkZWQ0Y2E5NmQxMTI3ZGM3NDU0NzU4NGE1Y2UwYTVmNTViYjJmNDI0MWZiYTg4ODNmMzgxYTQ3ZmI2OWU)から招待を受けてください)

## chiVeの引用

chiVeについて、論文を発表しています;

- 真鍋陽俊, 岡照晃, 海川祥毅, 髙岡一馬, 内田佳孝, 浅原正幸. [複数粒度の分割結果に基づく日本語単語分散表現](https://www.anlp.jp/proceedings/annual_meeting/2019/pdf_dir/P8-5.pdf). 言語処理学会第25回年次大会, 2019.
- 河村宗一郎, 久本空海, 真鍋陽俊, 高岡一馬, 内田佳孝, 岡照晃, 浅原正幸. [chiVe 2.0: SudachiとNWJCを用いた実用的な日本語単語ベクトルの実現へ向けて](https://www.anlp.jp/nlp2020/). 言語処理学会第26回年次大会, 2020.

Sudachiを論文や書籍、サービスなどで引用される際には、以下のBibTexをご利用ください。

```
@INPROCEEDINGS{manabe2019chive,
    author    = {真鍋陽俊, 岡照晃, 海川祥毅, 髙岡一馬, 内田佳孝, 浅原正幸},
    title     = {複数粒度の分割結果に基づく日本語単語分散表現},
    booktitle = "言語処理学会第25回年次大会(NLP2019)",
    year      = "2019",
    pages     = "NLP2019-P8-5",
    publisher = "言語処理学会",
}
```

```
@INPROCEEDINGS{kawamura2020chive,
    author    = {河村宗一郎, 久本空海, 真鍋陽俊, 高岡一馬, 内田佳孝, 岡照晃, 浅原正幸},
    title     = {chiVe 2.0: SudachiとNWJCを用いた実用的な日本語単語ベクトルの実現へ向けて},
    booktitle = "言語処理学会第26回年次大会(NLP2020)",
    year      = "2020",
    pages     = "NLP2020-P6-16",
    publisher = "言語処理学会",
}
```