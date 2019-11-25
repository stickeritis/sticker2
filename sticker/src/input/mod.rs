use conllx::graph::{Node, Sentence};
use wordpiece::{WordPiece, WordPieces};

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
    /// The offset at position 0 contains a dummy value, since this
    /// token represents the special *ROOT* token.
    pub fn tokenize(&self, sentence: &Sentence) -> (Vec<String>, Vec<usize>) {
        // An average of three pieces per token ought to enough for
        // everyone ;).
        let mut pieces = Vec::with_capacity((sentence.len() - 1) * 3);
        let mut token_offsets = Vec::with_capacity(sentence.len());

        // Stub for artificial root.
        token_offsets.push(0);

        for token in sentence.iter().filter_map(Node::token) {
            token_offsets.push(pieces.len());

            match self
                .word_pieces
                .split(token.form())
                .add_continuation_markers("##")
                .collect::<Option<Vec<_>>>()
            {
                Some(word_pieces) => pieces.extend(word_pieces),
                None => pieces.push(self.unknown_piece.clone()),
            }
        }

        (pieces, token_offsets)
    }
}

/// Add continuation markers for every piece except the first.
trait AddContinuationMarkers {
    type Iter;

    /// Return an iterator that adds continuation markers.
    fn add_continuation_markers(self, marker: impl Into<String>) -> Self::Iter;
}

impl<'a, I> AddContinuationMarkers for I
where
    I: Iterator<Item = WordPiece<'a>>,
{
    type Iter = ContinuationMarkerIter<I>;

    fn add_continuation_markers(self, marker: impl Into<String>) -> Self::Iter {
        ContinuationMarkerIter {
            initial: true,
            inner: self,
            marker: marker.into(),
        }
    }
}

struct ContinuationMarkerIter<I> {
    initial: bool,
    inner: I,
    marker: String,
}

impl<'a, I> Iterator for ContinuationMarkerIter<I>
where
    I: Iterator<Item = WordPiece<'a>>,
{
    type Item = Option<String>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next()?;

        let item = if self.initial {
            self.initial = false;
            item.piece().map(ToOwned::to_owned)
        } else {
            item.piece()
                .map(|piece| format!("{}{}", self.marker, piece))
        };

        Some(item)
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
    use wordpiece::WordPieces;

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

        let (pieces, offsets) = tokenizer.tokenize(&sentence);
        assert_eq!(
            pieces,
            &["Ver", "##unt", "##reute", "die", "A", "##W", "##O", "Spenden", "##geld", "[UNK]"]
        );
        assert_eq!(offsets, &[0, 0, 3, 4, 7, 9]);
    }
}
