#  Applying  NMF to discover hidden topics in email datasets and visualize dominant topics and their keyword distributions.

This project applies **Non-negative Matrix Factorization (NMF)** to a cleaned email dataset to extract hidden topics, analyze keyword distributions, and visualize topic frequencies using bar plots.

## Objective

- Automatically identify hidden topics in emails.
- Extract the top keywords for each topic using NMF.
- Count and visualize the frequency of these keywords across the dataset.
- Display keyword distributions for each topic using bar graphs.

---

## Dataset

- The dataset used is a sample or fake email dataset (e.g. from GitHub or cleaned Enron subset).
- Required column: `Message` – the main email text.
- The dataset is preprocessed to clean and tokenize email contents into lowercase, punctuation-free text.
https://github.com/MWiechmann/enron_spam_data/blob/master/enron_spam_data.zip
---

##  Methodology

1. **Preprocessing:**
   - Convert text to lowercase.
   - Remove punctuation and non-alphabetic characters.
   - Remove stopwords and apply tokenization.

2. **Feature Extraction:**
   - Apply **TF-IDF Vectorization** to convert text to numerical form.

3. **Topic Modeling:**
   - Fit an **NMF model** to the TF-IDF matrix.
   - Extract top N keywords for each topic.

4. **Visualization:**
   - Count frequency of each topic's top keywords in the entire dataset.
   - Plot:
     - A bar chart of **total keyword frequency per topic**.
     - Multiple bar charts (one per topic) showing individual keyword counts.
     - All topic plots are displayed in one window using subplots.

---

##  Visual Output

- One summary plot showing **total keyword frequencies per topic**.
- One subplot grid showing **each topic's top keyword frequency**.

Example:
Topic #1 - Keyword Frequencies
Keyword1 : http
Keyword2 | html
...
## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- nltk

Install them with:

```bash
pip install pandas numpy scikit-learn matplotlib nltk

## How to Run
Clone the repository or download the script.

Place your dataset CSV in the same directory (with a Message column).

Run the main Python file:

```bash
Copy
Edit
python nmf.py

## References
Scikit-learn NMF Docs

NLTK Stopwords

Sample datasets (e.g., Enron, GitHub mock emails)

