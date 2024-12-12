from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Fetch and print target names
emails = fetch_20newsgroups()
print(emails.target_names)

# Fetch specific categories
emails = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'])

# Print email at index 5
print(emails.data[5])

# Print label of email at index 5
label_index = emails.target[5]
print(label_index)
print(emails.target_names[label_index])

# Split data into training and test sets
train_emails = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'], subset='train', shuffle=True, random_state=108)
test_emails = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'], subset='test', shuffle=True, random_state=108)

# Create CountVectorizer and fit it
counter = CountVectorizer()
counter.fit(train_emails.data)

# Transform data into word counts
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

# Create and fit the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)

# Print the accuracy of the classifier
print(classifier.score(test_counts, test_emails.target))

# Test with different categories
train_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], subset='train', shuffle=True, random_state=108)
test_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], subset='test', shuffle=True, random_state=108)

# Optionally test with more categories
train_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey', 'sci.space'], subset='train', shuffle=True, random_state=108)
test_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey', 'sci.space'], subset='test', shuffle=True, random_state=108)

