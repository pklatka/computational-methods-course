import dask.array as da
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import pickle
import numpy as np
import re
from multiprocessing.pool import ThreadPool
from collections import Counter
import os
import scipy as sp
import requests
nlp = spacy.load("en_core_web_sm")
global_folder_name = ""  # Used to store data in different folders for different runs


def get_articles_from_category(start_category, max_article_number=500):
    current_articles = 0
    articles = []
    stack = [start_category]
    visited = set([start_category])

    while current_articles < max_article_number and stack:
        category = stack.pop()
        # Get articles
        response = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle={category}&cmlimit=500&format=json")

        response = response.json()

        # Get query
        query = response["query"]['categorymembers']

        for article in query:
            if article["ns"] == 0:
                articles.append(article)
                current_articles += 1
                if current_articles == max_article_number:
                    stack = []
                    break
            elif article["ns"] == 14 and article["title"] not in visited:
                stack.append(article["title"])
                visited.add(article["title"])

    # Filter duplicates
    pageid_set = set()
    filtered_articles = []
    for article in articles:
        if article["pageid"] not in pageid_set:
            pageid_set.add(article["pageid"])
            filtered_articles.append(article)

    start_category = start_category.replace(":", "-")
    with open(f'./pages-from-categories/{start_category}.pickle', 'wb') as f:
        pickle.dump(filtered_articles, f)

    return filtered_articles


def create_dict(pageids):
    errors = 0
    for pageid in pageids:
        try:
            response = requests.get(
                f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext&exsectionformat=wiki&redirects=1&pageids={pageid}")

            page = response.json()
            article_id = list(page['query']['pages'].keys())[0]
            content = re.sub(
                r"={2} .+", "", page['query']['pages'][article_id]['extract'])

            # Tokenize text
            tokens = re.findall(
                "[^\W\d_]+", content)

            # Lemmatize tokens and remove stop words
            tokens = [token.lemma_.lower() for token in nlp(
                " ".join(tokens)) if token.lemma_ not in STOP_WORDS]

            # Count tokens
            counted_tokens = Counter(tokens)
            counted_tokens = {k: v for k,
                              v in counted_tokens.items() if v > 1}

            # Get summary
            summary_request = requests.get(
                f"https://en.wikipedia.org/w/api.php?format=json&exintro&action=query&prop=extracts&explaintext&exsectionformat=wiki&redirects=1&pageids={article_id}&exsentences=2")

            summary = summary_request.json()

            result = {
                "title": page['query']['pages'][article_id]['title'],
                "pageid": article_id,
                "link": f"https://en.wikipedia.org/wiki?curid={article_id}",
                "tokens": counted_tokens,
                "tokensNumber": sum(counted_tokens.values()),
                "summary": summary["query"]["pages"][article_id]["extract"],
            }

            with open(f"./parsed-articles{global_folder_name}/article-{article_id}.pickle", "wb") as f:
                pickle.dump(result, f)

        except Exception as e:
            print(f"ERROR: {pageid} - {e}")
            errors += 1
    return errors


def get_pages(filename, number_of_articles=100, thread_pool_size=10):
    if number_of_articles < thread_pool_size:
        thread_pool_size = number_of_articles

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    data = list(map(lambda x: x["pageid"], data))

    # Get random titles
    random_titles = np.random.choice(data, number_of_articles, replace=False)

    # Split titles into chunks
    random_titles = np.array_split(random_titles, thread_pool_size)

    errors = 0
    with ThreadPool(thread_pool_size) as pool:
        # Call a function on each item in a list and handle results
        for result in pool.map(create_dict, random_titles):
            # Count errors
            errors += result

    return errors


def build_sparse_matrix(n=-1, use_idf=False, k=-1):
    # Get n articles
    i = 0
    articles = []
    words_in_articles = {}
    for file in os.listdir('./parsed-articles'):
        if os.path.getsize(f"./parsed-articles/{file}") == 0:
            continue

        with open(f"./parsed-articles/{file}", "rb") as f:
            article = pickle.load(f)
            for word in article["tokens"].keys():
                if word not in words_in_articles:
                    words_in_articles[word] = 0
                words_in_articles[word] += 1

            articles.append(
                {"title": article["title"], "pageid": article["pageid"], "link": article["link"], "summary": article["summary"]})
            i += 1

        if n != -1 and i == n:
            break

    word_set = list(words_in_articles.keys())

    # Map words to indexes
    word_set_index = {k: v for v, k in enumerate(word_set)}

    # Create sparse matrix
    matrix = sp.sparse.lil_matrix((len(articles), len(word_set)))

    # Fill sparse matrix
    for i, article in enumerate(articles):
        if os.path.getsize(f"./parsed-articles/article-{article['pageid']}.pickle") == 0:
            continue

        with open(f"./parsed-articles/article-{article['pageid']}.pickle", "rb") as f:
            article = pickle.load(f)
            for word in article["tokens"]:
                matrix[i, word_set_index[word]] = article["tokens"][word]

    idf_text = ""
    svd_text = ""

    # Normalize matrix using Inverse Document Frequency
    if use_idf:
        for i in range(matrix.shape[0]):
            matrix[i] = matrix[i] * \
                np.log(matrix.shape[0] / words_in_articles[word_set[i]])
        idf_text = "-idf"

    print(type(matrix))
    # Remove noises using SVD
    if k != -1:
        print("Using SVD ...", min(matrix.shape)-k)
        matrix = da.asarray(matrix.transpose())
        print(matrix.shape)
        u, s, vt = da.linalg.svd_compressed(matrix, k=min(matrix.shape)-k)
        # u, s, vt = sp.sparse.linalg.svds(matrix, k=k)
        matrix = None  # Free memory
        print("SVD done")
        print("Multiplying ...")
        matrix = sp.sparse.csr_matrix(u * s @ vt).transpose()
        print("Multiplication done")
        svd_text = "-svd"

    # Transpose matrix
    matrix = matrix.transpose()

    # Get better memory representation
    matrix = matrix.tocsr()

    # Calculate matrix norm
    matrix_norm = sp.sparse.linalg.norm(matrix, axis=0)

    # Save objects to pickle files
    with open(f"./calculated-components/matrix{svd_text}{idf_text}.pickle", "wb") as f:
        pickle.dump((matrix, matrix_norm), f)

    with open("./calculated-components/word_set.pickle", "wb") as f:
        pickle.dump(word_set, f)

    with open("./calculated-components/articles.pickle", "wb") as f:
        pickle.dump(articles, f)

    # print("Matrix shape:", matrix)
    print("Number of words:", len(word_set))
    print("Number of articles:", len(articles))

    return matrix, word_set, articles


def get_results_from_query_vector(query, k=10, matrix_filename="matrix"):
    # Get matrix
    with open(f"./calculated-components/{matrix_filename}.pickle", "rb") as f:
        matrix, matrix_norm = pickle.load(f)

    # Get word set
    with open("./calculated-components/word_set.pickle", "rb") as f:
        word_set = pickle.load(f)

    # Get articles
    with open("./calculated-components/articles.pickle", "rb") as f:
        articles = pickle.load(f)

    # Create query vector
    query_vector = sp.sparse.lil_matrix((len(word_set), 1))

    # Fill query vector
    for word in re.findall("[^\W\d_]+", query.lower()):
        if word in word_set:
            query_vector[word_set.index(word)] = 1

    # Get vector norm
    query_vector_norm = sp.sparse.linalg.norm(query_vector)
    norm = query_vector_norm * matrix_norm

    # Calculate probabilities
    probabilities = (query_vector.T @ matrix) / norm

    probabilities = [(probabilities[0, i], i) for i in range(matrix.shape[1])]

    # Sort probabilities
    probabilities = list(map(lambda x: (x[0], articles[x[1]]), filter(lambda x: not np.isnan(x[0]) and np.isfinite(
        x[0]) and x[0] > 0, sorted(probabilities, key=lambda x: x[0], reverse=True))))

    # Get top k results
    return probabilities[: k]


if __name__ == "__main__":
    # Category:Computing
    # category_name = "Category:Fields_of_mathematics"
    # x = get_articles_from_category(category_name, 10000)
    # category_name = category_name.replace(':', '-')
    # with open(f"./pages-from-categories/{category_name}.pickle", 'rb') as f:
    #     data = pickle.load(f)
    # print(len(data))
    # global_folder_name = f"-{category_name}"
    # os.mkdir(f"./parsed-articles{global_folder_name}")
    # res = get_pages(
    #     f'./pages-from-categories/{category_name}.pickle', len(data), 1000)
    # print(res)

    # category_name = "Category:Physics"
    # x = get_articles_from_category(category_name, 10000)
    # category_name = category_name.replace(':', '-')
    # with open(f"./pages-from-categories/{category_name}.pickle", 'rb') as f:
    #     data = pickle.load(f)
    # print(len(data))
    # global_folder_name = f"-{category_name}"
    # os.mkdir(f"./parsed-articles{global_folder_name}")
    # res = get_pages(
    #     f'./pages-from-categories/{category_name}.pickle', len(data), 1000)
    # print(res)

    # category_name = "Category:Religion"
    # x = get_articles_from_category(category_name, 10000)
    # category_name = category_name.replace(':', '-')
    # with open(f"./pages-from-categories/{category_name}.pickle", 'rb') as f:
    #     data = pickle.load(f)
    # print(len(data))
    # global_folder_name = f"-{category_name}"
    # os.mkdir(f"./parsed-articles{global_folder_name}")
    # res = get_pages(
    #     f'./pages-from-categories/{category_name}.pickle', len(data), 1000)
    # print(res)
    # Category:Fields_of_mathematics

    # Category:Physics

    # Category:Religion

    # articles = build_sparse_matrix(n=1000)
    # articles = build_sparse_matrix(use_idf=True)
    # articles = build_sparse_matrix(k=10, n=1000, use_idf=True)
    # articles = build_sparse_matrix(k=500, use_idf=True)

    print(get_results_from_query_vector(
        "algorithm", 10, matrix_filename="matrix"))

    print(get_results_from_query_vector(
        "algorithm", 10, matrix_filename="matrix-svd"))
