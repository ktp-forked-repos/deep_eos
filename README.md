# *DEEP-EOS*: Deep end-of-sentence boundary detection

In this repository we present a network models for sentence
boundary detection. This repository superseeds the [old implementation](https://github.com/stefan-it/deep-eos).

## Introduction

The task of sentence boundary detection is to identify sentences within
a text. Many natural language processing tasks take a sentence as an
input unit, such as part-of-speech tagging ([Manning,
2011](http://dl.acm.org/citation.cfm?id=1964799.1964816)), dependency
parsing ([Yu and Vu, 2017](http://aclweb.org/anthology/P17-2106)),
named entity recognition or machine translation.

Sentence boundary detection is a nontrivial task, because of the
ambiguity of the period sign `.`, which has several functions
([Grefenstette and Tapanainen,
1994](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.28.5162)),
e.g.:

* End of sentence
* Abbreviation
* Acronyms and initialism
* Mathematical numbers

A sentence boundary detection system has to resolve the use of
ambiguous punctuation characters to determine if the punctuation
character is a true end-of-sentence marker. In this implementation we
define `?!:;.` as potential end-of sentence markers.

Various approaches have been employed to achieve sentence boundary
detection in different languages. Recent research in sentence boundary
detection focus on machine learning techniques, such as hidden Markov
models ([Mikheev, 2002](http://dx.doi.org/10.1162/089120102760275992)),
maximum entropy ([Reynar and Ratnaparkhi,
1997](https://doi.org/10.3115/974557.974561)), conditional random
fields ([Tomanek et al.,
2007](http://www.bootstrep.org/files/publications/FSU_2007_Tomanek_Pacling.pdf)),
decision tree ([Wong et al.,
2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4030568/)) and
neural networks ([Palmer and Hearst,
1997](http://dl.acm.org/citation.cfm?id=972695.972697)). [Kiss and
Strunk (2006)](http://dx.doi.org/10.1162/coli.2006.32.4.485) use an
unsupervised sentence detection system called Punkt, which does not
depend on any additional resources. The system use collocation
information as evidence from unannotated corpora to detect e.g.
abbreviations or ordinal numbers.

The sentence boundary detection task can be treated as a classification
problem. Our work is similar to the *SATZ* system, proposed by [Palmer
and Hearst (1997)](http://dl.acm.org/citation.cfm?id=972695.972697),
which uses a fully-connected feed-forward neural network. The *SATZ*
system disambiguates a punctuation mark given a context of *k*
surrounding words. This is different to our approach, as we use a
char-based context window instead of a word-based context window.

In the present work, we train a long short-term memory (LSTM) and compare the results
with *OpenNLP*. *OpenNLP* is a state-of-the-art tool and uses a maximum
entropy model for sentence boundary detection. To test the robustness of our
model, we use the *Universal Dependencies* for several languages as
datasets.

# Datasets

At the moment, training, prediction and evaluation is supported on the
*Universal Dependencies* (<http://universaldependencies.org/>) datasets, as well as for plain-text
datasets (one sentence per line).

## *Universal Dependencies*

*Universal Dependencies* in version 2.2 can be downloaded with the following command (you should not
place these files in the `deep-eos` repository folder):

```bash
cd ..
curl --remote-name-all "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2837/ud-treebanks-v2.2.tgz"
tar -xzf ud-treebanks-v2.2.tgz
```

# Configuration

*deep-eos* uses a configuration-based commandline interface for training, prediction and evaluation. 
That means, all parameters are specified in a *toml* configuration file. This heavily reduces the 
amount of commandline arguments. The only commandline option is the path to the *toml* configuration
file.

## *toml* configuration

### *Universal Dependencies* datasets

A typical configurations for a *Universal Dependencies* dataset looks like:

```toml
[ud]
train_path = "../ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-train.conllu"
dev_path = "../ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-dev.conllu"
test_path = "../ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-test.conllu"

[lstm]
embedding_size = 256
hidden_size = 256
output_size = 2

[training]
batch_size = 32
epochs = 10
model_file = "./best_ud_german_model.pt"
learning_rate = 0.001

[deep-eos]
left_ws = 4
right_ws = 4

[prediction]
dev_tagged_file = "ud_german_dev_tagged.txt"
test_tagged_file = "ud_german_test_tagged.txt"
eos_marker = "</eos>"
```

### Plain-text datasets

It is also possible to use plain-text datasets for training, development and testing. Then you
have to specify *one* sentence per line. In the *toml* configuration you have to specifiy a
`[text]` section instead of `[ud]` as used for *Universal Dependencies* datasets.

# Model

We use a long short-term memory for sentence boundary detection. This
model captures information at the character level. Our models
disambiguate potential end-of-sentence markers followed by a whitespace
or line break given a context of *k* surrounding characters. The
potential end-of-sentence marker is also included in the context
window. The following table shows an example of a sentence and its
extracted contexts: left context, middle context and right context. We
also include the whitespace or line break after a potential
end-of-sentence marker.


| Input sentence        | Left  | Middle | Right
| --------------------- | ----- | ------ | -----
| I go to Mr. Pete Tong | to Mr | .      | _Pete

## LSTM

We use a standard LSTM ([Hochreiter and Schmidhuber,
1997](http://dx.doi.org/10.1162/neco.1997.9.8.1735); [Gers et al.,
2000](http://dx.doi.org/10.1162/089976600300015015)) network with an
embedding size of 256. The number of hidden states is 256.

## Other Hyperparameters

Our proposed character-based model disambiguates a punctuation mark
given a context of *k* surrounding characters. In our experiments we
found that a context size of 4 surrounding characters gives the best
results. We found that it is very important to include the
end-of-sentence marker in the context, as this increases the F1-score
of 2%.  All models are trained with averaged stochastic gradient
descent with a learning rate of 0.001 and mini-batch size of 32. We use
Adam for first-order gradient-based optimization. We use binary
cross-entropy as loss function.

# Implementation

We use *PyTorch* for the implementation of the neural network
architecture.

# Contact (Bugs, Feedback, Contribution and more)

For questions about *deep-eos*, just open an issue [here](https://github.com/stefan-it/deep_eos/issues).
A contribution is coming soon!

# License

To respect the Free Software Movement and the enormous work of Dr.
Richard Stallman this implementation is released under the *GNU Affero
General Public License* in version 3. More information can be found
[here](https://www.gnu.org/licenses/licenses.html) and in `COPYING`.
