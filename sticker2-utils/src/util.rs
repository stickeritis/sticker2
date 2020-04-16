use std::io::BufRead;

use anyhow::Result;

pub fn count_conllu_sentences(buf_read: impl BufRead) -> Result<usize> {
    let mut n_sents = 0;

    for line in buf_read.lines() {
        let line = line?;
        if line.starts_with("1\t") {
            n_sents += 1;
        }
    }

    Ok(n_sents)
}
