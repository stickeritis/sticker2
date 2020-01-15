use std::collections::HashMap;
use std::io::{BufReader, Read, Seek, SeekFrom};

use conllx::graph::Sentence;
use conllx::io::{ReadSentence, Reader};
use failure::Fallible;
use ndarray::Array1;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use sticker_encoders::SentenceEncoder;

use crate::encoders::NamedEncoder;
use crate::input::vectorizer::WordPieceVectorizer;
use crate::input::{SentenceWithPieces, Tokenize, WordPieceTokenizer};
use crate::tensor::{NoLabels, TensorBuilder, Tensors};
use crate::util::RandomRemoveVec;

/// A set of training/validation data.
///
/// A data set provides an iterator over the batches in that
/// dataset.
pub trait DataSet<'a> {
    type Iter: Iterator<Item = Fallible<Tensors>>;

    /// Get an iterator over the dataset batches.
    ///
    /// The sequence inputs are encoded with the given `vectorizer`,
    /// the sequence labels using the `encoder`.
    ///
    /// Sentences longer than `max_len` are skipped. If you want to
    /// include all sentences, you can use `usize::MAX` as the maximum
    /// length.
    #[allow(clippy::too_many_arguments)]
    fn batches(
        self,
        encoders: &'a [NamedEncoder],
        tokenizer: &'a WordPieceTokenizer,
        vectorizer: &'a WordPieceVectorizer,
        batch_size: usize,
        max_len: Option<usize>,
        shuffle_buffer_size: Option<usize>,
        labels: bool,
    ) -> Fallible<Self::Iter>;
}

/// A CoNLL-X data set.
pub struct ConllxDataSet<R>(R);

impl<R> ConllxDataSet<R> {
    /// Construct a CoNLL-X dataset.
    pub fn new(read: R) -> Self {
        ConllxDataSet(read)
    }

    /// Returns an `Iterator` over `Result<Sentence, Error>`.
    ///
    /// Depending on the parameters the returned iterator filters
    /// sentences by their lengths or returns the sentences in
    /// sequence without filtering them.
    ///
    /// If `max_len` == `None`, no filtering is performed.
    fn get_sentence_iter<'a>(
        reader: R,
        max_len: Option<usize>,
        shuffle_buffer_size: Option<usize>,
    ) -> Box<dyn Iterator<Item = Fallible<Sentence>> + 'a>
    where
        R: ReadSentence + 'a,
    {
        match (max_len, shuffle_buffer_size) {
            (Some(max_len), Some(buffer_size)) => Box::new(
                reader
                    .sentences()
                    .filter_by_len(max_len)
                    .shuffle(buffer_size),
            ),
            (Some(max_len), None) => Box::new(reader.sentences().filter_by_len(max_len)),
            (None, Some(buffer_size)) => Box::new(reader.sentences().shuffle(buffer_size)),
            (None, None) => Box::new(reader.sentences()),
        }
    }
}

impl<'a, 'ds, R> DataSet<'a> for &'ds mut ConllxDataSet<R>
where
    R: Read + Seek,
{
    type Iter = ConllxIter<'a, Box<dyn Iterator<Item = Fallible<Sentence>> + 'ds>>;

    fn batches(
        self,
        encoders: &'a [NamedEncoder],
        tokenizer: &'a WordPieceTokenizer,
        vectorizer: &'a WordPieceVectorizer,
        batch_size: usize,
        max_len: Option<usize>,
        shuffle_buffer_size: Option<usize>,
        labels: bool,
    ) -> Fallible<Self::Iter> {
        // Rewind to the beginning of the data (if necessary).
        self.0.seek(SeekFrom::Start(0))?;

        let reader = Reader::new(BufReader::new(&mut self.0));

        Ok(ConllxIter {
            batch_size,
            encoders,
            labels,
            sentences: ConllxDataSet::get_sentence_iter(reader, max_len, shuffle_buffer_size),
            tokenizer,
            vectorizer,
        })
    }
}

pub struct ConllxIter<'a, I>
where
    I: Iterator<Item = Fallible<Sentence>>,
{
    batch_size: usize,
    labels: bool,
    encoders: &'a [NamedEncoder],
    tokenizer: &'a WordPieceTokenizer,
    vectorizer: &'a WordPieceVectorizer,
    sentences: I,
}

impl<'a, I> ConllxIter<'a, I>
where
    I: Iterator<Item = Fallible<Sentence>>,
{
    fn next_with_labels(
        &mut self,
        tokenized_sentences: Vec<SentenceWithPieces<String>>,
        max_seq_len: usize,
    ) -> Option<Fallible<Tensors>> {
        let mut builder = TensorBuilder::new(
            tokenized_sentences.len(),
            max_seq_len,
            self.encoders.iter().map(NamedEncoder::name),
        );

        for sentence in tokenized_sentences {
            let input = self.vectorizer.vectorize(&sentence.pieces);
            let mut token_mask = Array1::zeros((input.len(),));
            for token_idx in &sentence.token_offsets {
                token_mask[*token_idx] = 1;
            }

            let mut encoder_labels = HashMap::with_capacity(self.encoders.len());
            for encoder in self.encoders {
                let encoding = match encoder.encoder().encode(&sentence.sentence) {
                    Ok(encoding) => encoding,
                    Err(err) => return Some(Err(err)),
                };

                let mut labels = Array1::from_elem((input.len(),), 1i64);
                for (encoding, offset) in encoding.into_iter().zip(&sentence.token_offsets) {
                    labels[*offset] = encoding as i64;
                }

                encoder_labels.insert(encoder.name(), labels);
            }

            builder.add_with_labels(input.view(), encoder_labels, token_mask.view());
        }

        Some(Ok(builder.into()))
    }

    fn next_without_labels(
        &mut self,
        tokenized_sentences: Vec<SentenceWithPieces<String>>,
        max_seq_len: usize,
    ) -> Option<Fallible<Tensors>> {
        let mut builder: TensorBuilder<NoLabels> = TensorBuilder::new(
            tokenized_sentences.len(),
            max_seq_len,
            self.encoders.iter().map(NamedEncoder::name),
        );

        for sentence in tokenized_sentences {
            let input = self.vectorizer.vectorize(&sentence.pieces);
            let mut token_mask = Array1::zeros((input.len(),));
            for token_idx in &sentence.token_offsets {
                token_mask[*token_idx] = 1;
            }

            builder.add_without_labels(input.view(), token_mask.view());
        }

        Some(Ok(builder.into()))
    }
}

impl<'a, I> Iterator for ConllxIter<'a, I>
where
    I: Iterator<Item = Fallible<Sentence>>,
{
    type Item = Fallible<Tensors>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch_sentences = Vec::with_capacity(self.batch_size);
        while let Some(sentence) = self.sentences.next() {
            let sentence = match sentence {
                Ok(sentence) => sentence,
                Err(err) => return Some(Err(err)),
            };
            batch_sentences.push(sentence);
            if batch_sentences.len() == self.batch_size {
                break;
            }
        }

        // Check whether the reader is exhausted.
        if batch_sentences.is_empty() {
            return None;
        }

        let tokenized_sentences = batch_sentences
            .into_iter()
            .map(|s| s.tokenize(&self.tokenizer))
            .collect::<Vec<_>>();

        let max_seq_len = tokenized_sentences
            .iter()
            .map(|s| s.pieces.len())
            .max()
            .unwrap_or(0);

        if self.labels {
            self.next_with_labels(tokenized_sentences, max_seq_len)
        } else {
            self.next_without_labels(tokenized_sentences, max_seq_len)
        }
    }
}

/// Trait providing adapters for `conllx::io::Sentences`.
pub trait SentenceIter: Sized {
    fn filter_by_len(self, max_len: usize) -> LengthFilter<Self>;
    fn shuffle(self, buffer_size: usize) -> Shuffled<Self>;
}

impl<I> SentenceIter for I
where
    I: Iterator<Item = Fallible<Sentence>>,
{
    fn filter_by_len(self, max_len: usize) -> LengthFilter<Self> {
        LengthFilter {
            inner: self,
            max_len,
        }
    }

    fn shuffle(self, buffer_size: usize) -> Shuffled<Self> {
        Shuffled {
            inner: self,
            buffer: RandomRemoveVec::with_capacity(buffer_size, XorShiftRng::from_entropy()),
            buffer_size,
        }
    }
}

/// An Iterator adapter filtering sentences by maximum length.
pub struct LengthFilter<I> {
    inner: I,
    max_len: usize,
}

impl<I> Iterator for LengthFilter<I>
where
    I: Iterator<Item = Fallible<Sentence>>,
{
    type Item = Fallible<Sentence>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(sent) = self.inner.next() {
            // Treat Err as length 0 to keep our type as Result<Sentence, Error>. The iterator
            // will properly return the Error at a later point.
            let len = sent.as_ref().map(|s| s.len()).unwrap_or(0);
            if len > self.max_len {
                continue;
            }
            return Some(sent);
        }
        None
    }
}

/// An Iterator adapter performing local shuffling.
///
/// Fills a buffer with size `buffer_size` on the first
/// call. Subsequent calls add the next incoming item to the buffer
/// and pick a random element from the buffer.
pub struct Shuffled<I> {
    inner: I,
    buffer: RandomRemoveVec<Sentence, XorShiftRng>,
    buffer_size: usize,
}

impl<I> Iterator for Shuffled<I>
where
    I: Iterator<Item = Fallible<Sentence>>,
{
    type Item = Fallible<Sentence>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.is_empty() {
            while let Some(sent) = self.inner.next() {
                match sent {
                    Ok(sent) => self.buffer.push(sent),
                    Err(err) => return Some(Err(err)),
                }

                if self.buffer.len() == self.buffer_size {
                    break;
                }
            }
        }

        match self.inner.next() {
            Some(sent) => match sent {
                Ok(sent) => Some(Ok(self.buffer.push_and_remove_random(sent))),
                Err(err) => Some(Err(err)),
            },
            None => self.buffer.remove_random().map(Result::Ok),
        }
    }
}
