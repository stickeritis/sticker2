# Pretrained models

This page contains an overview of pretrained models that are available
for sticker2. The models are available as
[Nix](https://nixos.org/nix/) packages and
[Docker](https://www.docker.com/) images. We first provide an overview
of the models that are available and then describes how to use them.

## Models

### Dutch

| Model       | UD POS | Lemma | Morphology | Dependency (LAS) | Size (MiB) |
|:------------|-------:|------:|-----------:|-----------------:|-----------:|
| nl-ud-large |  98.81 | 99.05 |      98.82 |            93.35 |        192 |

| Model       | Docker image                             | Nix attribute                 |
|:------------|:-----------------------------------------|:------------------------------|
| nl-ud-large | `danieldk/sticker2:nl-ud-large-20200420` | `sticker2_models.nl-ud-large` |


## Usage

### Docker

The Docker images have a tag of the form `LANG-TAGSET-SIZE-DATE`. For
example, `nl-ud-large-20200420` is a large Dutch UD model from April
20, 2020. Each model image contains two useful commands:

* `/bin/sticker2-annotate-LANG-TAGSET-SIZE`: annotate data using the
  model. If this command is run without any arguments, it reads from
  the standard input and writes to the standard output.
* `/bin/sticker2-server-LANG-TAGSET-SIZE`: starts a sticker2 server
  for the model. This server will listen on a socket for CoNLL-U data.

For example, you can annotate tokenized text using the
`nl-ud-large-20200420` model using the following command:

```bash
$ docker run -i --rm danieldk/sticker2:nl-ud-large-20200420 \
  /bin/sticker2-annotate-nl-ud-large \
  < test.conllu > annotated.conllu
```

### Nix

A Nix model attribute such as `sticker2_models.nl-ud-large` evaluates
to an attribute set with three attributes:

* `model`: the model data
* `wrapper`: a convenient wrapper of sticker2 and a the model
* `dockerImage`: a Docker image containint the model wrapper

If you are not very familiar with Nix, the easiest way use a model is
to install its wrapper it into your local user environment. For
example:

~~~bash
$ nix-env \
  -f https://github.com/stickeritis/nix-packages/archive/master.tar.gz \
  -iA sticker2_models.nl-ud-large.wrapper
~~~

This installs wrappers of the form
`sticker-{annotate,server}-model`. These wrappers call sticker2 with
the applicable model configuration file. For example if you have
installed `sticker2_model.nl-ud-large.wrapper`, you can annotate a
CoNLL-U file using:

~~~bash
$ sticker2-annotate-nl-ud-large corpus.conllu annotated.conllu
~~~

You can remove a model again using the `-e` flag of `nix-env`:

~~~
$ nix-env -e sticker2-nl-ud-large-wrapper
~~~

If you are an advanced Nix user, we recommend you to add the [package
set](https://github.com/stickeritis/nix-packages) to your Nix
configuration.

## Disclaimer

These models come as is, without any warranty or condition, and no
contributor will be liable to anyone for any damages related to these
models, under any kind of legal claim.
