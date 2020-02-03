let
  sources = import ./sources.nix;
in {
  # Vocabularies for testing.
  BERT_BASE_GERMAN_CASED_VOCAB = sources.bert-base-german-cased-vocab;
  XLM_ROBERTA_BASE_SENTENCEPIECE = sources.xlm-roberta-base-sentencepiece;
}
