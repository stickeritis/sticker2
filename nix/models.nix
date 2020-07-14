let
  sources = import ./sources.nix;
in {
  # Vocabularies for testing.
  ALBERT_BASE_V2_SENTENCEPIECE = sources.albert-base-v2-sentencepiece;
  BERT_BASE_GERMAN_CASED_VOCAB = sources.bert-base-german-cased-vocab;
  XLM_ROBERTA_BASE_SENTENCEPIECE = sources.xlm-roberta-base-sentencepiece;
}
