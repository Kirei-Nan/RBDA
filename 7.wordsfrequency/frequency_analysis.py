import string
import pandas as pd
import re
import nltk
from datetime import datetime
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def parse_year(date_str):
    date_str = date_str[0: 10]
    date = datetime.strptime(date_str, "%Y-%m-%d")
    if 2007 <= date.year < 2011:
        return "2007-2010"
    elif 2011 <= date.year < 2016:
        return "2011-2015"
    elif 2016 <= date.year < 2021:
        return "2016-2020"
    else:
        return "2021-2024"


def parse_quarter(date_str):
    date_str = date_str[0: 10]
    date = datetime.strptime(date_str, "%Y-%m-%d")
    quarter = (date.month - 1) // 3 + 1
    return f"{date.year}-Q{quarter}"


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = word_tokenize(text)
    new_list = ["https", "com", "github", "using", "use", "add", "user", "githubusercontent", "would", "page",
                "const", "return", "new", "null", "true", "name", "string", "get", "id", "app", "like", "tried", "want",
                "one", "even", "make", "also", "much", "really", "could", "something", "time", "year", "way", "many",
                "still", "things", "us", "lot", "need", "see", "well", "know", "go", "used", "say", "point", "going",
                "without", "thing", "actually", "someone", "people", "think", "though", "probably", "seems", "sure",
                "may", "might", "take", "every"]
    stop_words = set(stopwords.words('english'))
    for word in new_list:
        stop_words.add(word)
    for i in range(10):
        stop_words.add(str(i))
    for i in string.ascii_lowercase:
        stop_words.add(i)
    return [word for word in tokens if word not in stop_words]


def get_hot_words(data, top_n=10):
    all_words = []
    for content in data['content']:
        all_words.extend(preprocess_text(content))
    counter = Counter(all_words)
    return counter.most_common(top_n)


def plot_wordcloud(word_freq, name, date):
    title = f"{name} {date} Hot Words"
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(dict(word_freq))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.savefig(f"result/{name}/{name}-{date}.png")
    plt.show()


def export_hot_word(hot_words, name, date):
    output_file = f"result/{name}/{name}-{date}.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"{name} hot words {date}:\n")
        for word, freq in hot_words:
            file.write(f"{word}: {freq}\n")


if __name__ == "__main__":
    file_names = ['github', 'stackoverflow', 'quora', 'hackernews']
    file_paths = [f"data/{filename}_text.txt" for filename in file_names]
    df = []
    for file_path in file_paths:
        data = pd.read_csv(file_path, sep='\t', header=None, names=["time", "content"])
        data['quarter'] = data['time'].apply(parse_quarter)
        data['year'] = data['time'].apply(parse_year)
        df.append(data)

    nltk.download('punkt_tab')
    nltk.download('stopwords')

    # Analyze overall data
    for i in range(4):
        name = file_names[i]
        hot_words = get_hot_words(df[i])
        export_hot_word(hot_words, name, "overall")
        plot_wordcloud(hot_words, name, "overall")

    # Analyze data by quarter - gitHub, stackoverflow
    for i in range(2):
        for quarter, group in df[i].groupby('quarter'):
            name = file_names[i]
            hot_words = get_hot_words(group)
            export_hot_word(hot_words, name, quarter)
            plot_wordcloud(hot_words, name, quarter)

    # Analyze data by year - hackernews
    for year, group in df[3].groupby('year'):
        name = "hackernews"
        hot_words = get_hot_words(group)
        export_hot_word(hot_words, name, year)
        plot_wordcloud(hot_words, name, year)
