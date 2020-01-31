use conllx::graph::{Node, Sentence};
use ndarray::Array1;
use wordpieces::WordPieces;

/// Trait for wordpiece tokenizers.
pub trait Tokenize {
    /// Tokenize the tokens in a sentence into word pieces.
    fn tokenize(self, tokenizer: &WordPieceTokenizer) -> SentenceWithPieces;
}

impl Tokenize for Sentence {
    fn tokenize(self, tokenizer: &WordPieceTokenizer) -> SentenceWithPieces {
        tokenizer.tokenize(self)
    }
}

/// A sentence and its word pieces.
pub struct SentenceWithPieces {
    /// Word pieces in a sentence.
    pub pieces: Array1<i64>,

    /// Sentence graph.
    pub sentence: Sentence,

    /// The the offsets of tokens in `pieces`.
    pub token_offsets: Vec<usize>,
}

/// A word piece tokenizer.
///
/// This token splits CoNLL-X tokens into word pieces. For
/// example, a sentence such as:
///
/// > Veruntreute die AWO Spendengeld ?
///
/// Could be split (depending on the vocabulary) into the following
/// word pieces:
///
/// > Ver ##unt ##reute die A ##W ##O Spenden ##geld [UNK]
///
/// The unknown token (here `[UNK]`) can be specified while
/// constructing a tokenizer.
pub struct WordPieceTokenizer {
    word_pieces: WordPieces,
    unknown_piece: String,
}

impl WordPieceTokenizer {
    /// Construct a tokenizer from wordpieces and the unknown piece.
    pub fn new(word_pieces: WordPieces, unknown_piece: impl Into<String>) -> Self {
        WordPieceTokenizer {
            word_pieces,
            unknown_piece: unknown_piece.into(),
        }
    }

    /// Tokenize a CoNLL-X sentence.
    ///
    /// Returns a pair of:
    ///
    /// * The word pieces of all tokens;
    /// * the offset of each token into the word pieces.
    ///
    /// The offset at position 0 represents the first word (and not the
    /// special *ROOT* token).
    pub fn tokenize(&self, sentence: Sentence) -> SentenceWithPieces {
        // An average of three pieces per token ought to enough for
        // everyone ;).
        let mut pieces = Vec::with_capacity((sentence.len() - 1) * 3);
        let mut token_offsets = Vec::with_capacity(sentence.len());

        for token in sentence.iter().filter_map(Node::token) {
            token_offsets.push(pieces.len());

            match self
                .word_pieces
                .split(token.form())
                .map(|piece| piece.idx().map(|piece| piece as i64))
                .collect::<Option<Vec<_>>>()
            {
                Some(word_pieces) => pieces.extend(word_pieces),
                None => pieces.push(
                    self.word_pieces
                        .get_initial(&self.unknown_piece)
                        .expect("Cannot get unknown piece") as i64,
                ),
            }
        }

        SentenceWithPieces {
            pieces: pieces.into(),
            sentence,
            token_offsets,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::iter::FromIterator;

    use conllx::graph::Sentence;
    use conllx::token::Token;
    use ndarray::array;
    use wordpieces::WordPieces;

    use super::WordPieceTokenizer;

    fn read_pieces() -> WordPieces {
        let f = File::open("testdata/bert-base-german-cased-vocab.txt").unwrap();
        WordPieces::try_from(BufReader::new(f).lines()).unwrap()
    }

    fn sentence_from_forms(forms: &[&str]) -> Sentence {
        Sentence::from_iter(forms.iter().map(|&f| Token::new(f)))
    }

    #[test]
    fn test_pieces() {
        let tokenizer = WordPieceTokenizer::new(read_pieces(), "[UNK]");

        let sentence = sentence_from_forms(&["Veruntreute", "die", "AWO", "Spendengeld", "?"]);

        let sentence_pieces = tokenizer.tokenize(sentence);
        assert_eq!(
            sentence_pieces.pieces,
            array![133i64, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2]
        );
        assert_eq!(sentence_pieces.token_offsets, &[0, 3, 4, 7, 9]);
    }
}
