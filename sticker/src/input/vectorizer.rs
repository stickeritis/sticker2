use std::io::{self, BufRead};

use numberer::Numberer;
use tch::Tensor;

/// Trait for reading word pieces.
pub trait ReadWordPieces
where
    Self: Sized,
{
    /// Read word pieces from a buffered reader.
    fn read_word_pieces(buf_read: impl BufRead) -> io::Result<Self>;
}

/// Word word pieces vectorizer.
pub struct WordPieceVectorizer {
    numberer: Numberer<String>,
}

impl WordPieceVectorizer {
    /// Vectorize a slice of word pieces.
    pub fn vectorize(&self, word_pieces: &[impl AsRef<str>]) -> Tensor {
        let indices = word_pieces
            .iter()
            .map(|piece| self.numberer.number(piece.as_ref()).expect("Missing piece") as i64)
            .collect::<Vec<_>>();

        Tensor::of_slice(&indices)
    }
}

impl ReadWordPieces for WordPieceVectorizer {
    fn read_word_pieces(buf_read: impl BufRead) -> io::Result<Self> {
        let mut numberer = Numberer::new(0);

        for line in buf_read.lines() {
            let piece = line?;
            numberer.add(piece);
        }

        Ok(WordPieceVectorizer { numberer })
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;

    use tch::Tensor;

    use super::{ReadWordPieces, WordPieceVectorizer};

    fn read_vectorizer() -> WordPieceVectorizer {
        let f = File::open("testdata/bert-base-german-cased-vocab.txt").unwrap();
        WordPieceVectorizer::read_word_pieces(BufReader::new(f)).unwrap()
    }

    #[test]
    fn test_vectorizer() {
        let vectorizer = read_vectorizer();

        let pieces = &[
            "Ver", "##unt", "##reute", "die", "A", "##W", "##O", "Spenden", "##geld", "[UNK]",
        ];

        let tensor = vectorizer.vectorize(pieces);

        assert_eq!(
            tensor,
            Tensor::of_slice(&[133i64, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2])
        );
    }
}
