use std::error::Error;
use std::fs::File;
use std::io::{prelude::*, ErrorKind};

fn main() -> std::io::Result<()> {
    let mut file = File::open("results")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let mut results: Vec<String> = contents
        .trim()
        .split("\n\n")
        .map(|result| {
            let (time, rest) = result.split_once('\n').unwrap();
            String::from(rest)
        })
        .collect();

    for result in &results {
        if results.iter().any(|r| !result.eq(r)) {
            return Err(std::io::Error::new(
                ErrorKind::Other,
                "check_results failed",
            ));
        }
    }

    Ok(())
}
