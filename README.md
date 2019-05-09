# Text-Classifiers with Attention
Repo that contains experiments run on text categorizes using CNNs, LSTMs, GRUs with Attention.
That's right, CNNs. 

One might wonder why CNNs are used in text categorization. The idea here is that the spatial features of a sentence is preserved just like in an image, typically by using n-grams. These n-gram representations of the text is converted to an embeddings matrix and a CNN is used to classify the embeddings matrix.
