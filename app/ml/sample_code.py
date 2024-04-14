# -*- coding: utf-8 -*-

# Recommendation System


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from textblob import TextBlob


# ___________Creating dic for getting business information______________

def create_category_dict(csv_file):
    df = pd.read_csv(csv_file)

    category_dict = {}

    for category in df["Category Title"].unique():
        category_df = df[df["Category Title"] == category]
        restaurant_list = [
            {
                "city": row["city"],
                "Business Name": row["business_names"],
                "Address": row["Address"],
                "Rating": row["Rating"],
                "Reviews": row["Comment"]
            }
            for _, row in category_df.iterrows()
        ]
        category_dict[category] = restaurant_list

    return category_dict
# _____________Applying sentiment analysis on the restaurant reviews_____________________


def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# _________Getting the top ranked restaurant on the basis of ratings and reviews_____________


def find_top_restaurant_in_category(category_dict, category):
    restaurants = category_dict.get(category, [])
    if not restaurants:
        return None, None

    for restaurant in restaurants:
        sentiment = analyze_sentiment(restaurant['Reviews'])
        if sentiment == 'positive':
            restaurant['Score'] = restaurant['Rating'] * 1.5
        elif sentiment == 'negative':
            restaurant['Score'] = restaurant['Rating'] * 0.5
        else:
            restaurant['Score'] = restaurant['Rating']

    top_restaurant = max(restaurants, key=lambda x: x['Score'])
    return top_restaurant, top_restaurant['city']


# ______________Getting the top 5 similar cities to one another_____________

def finding_similar_cities(csv_file, target_city, exclude_restaurant=None):
    df = pd.read_csv(csv_file)
    scaler = StandardScaler()

    if target_city not in df['city'].values:
        print(f"No demographic data found for {target_city}.")
        return []

    # _________Applying standard scaler to normalise the numerical features_______________

    numerical_features = [col for col in df.columns if col !=
                          'city' and df[col].dtype in [np.int64, np.float64]]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # __________Applying Tfidf on textual desciption of the cities_________________

    wiki_descriptions = df["wiki_description"].fillna(
        "")  # Filling NaN values with empty strings
    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    tfidf_features = tfidf_vectorizer.fit_transform(
        wiki_descriptions).toarray()

    # Combining the numerical and TF-IDF features
    combined_features = np.hstack((df[numerical_features], tfidf_features))

    # Applying PCA on combined features to reduced the dimentionality and get the best features.
    # removing noise and extra uncesasary features that deos not add up

    # ______________Applying PCA_____________________

    pca = PCA(n_components=5)
    data_pca = pca.fit_transform(combined_features)

    # KNN

    from sklearn.neighbors import NearestNeighbors

    # Using 6 neighbors to include the target city itself
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(data_pca)

    target_index = df.index[df['city'] == target_city].tolist()[0]
    i, indices = knn.kneighbors([data_pca[target_index]])
    top_5_similar_cities = indices[0][1:]  # Exclude the target city itself

    # Making sure that the top restaurant business does not exist in the recommended location
    if exclude_restaurant:
        top_restaurant_name = exclude_restaurant["Business Name"]
        city_names = df['city'].values
        for idx in top_5_similar_cities:
            similar_city_name = city_names[idx]
            similar_restaurants = category_dict.get(similar_city_name, [])
            for restaurant in similar_restaurants:
                if restaurant["Business Name"] == top_restaurant_name:
                    print(
                        f"The top-ranked restaurant '{top_restaurant_name}' is also found in the recommended city '{similar_city_name}'.")
                    top_5_similar_cities = np.delete(
                        top_5_similar_cities, np.where(top_5_similar_cities == idx))
                    break

    similar_cities = [df.iloc[i]['city'] for i in top_5_similar_cities]
    print(f"Similar cities: {similar_cities}")

    return similar_cities


def main():

    # ____path to get the datset for restaurant rating and reviews____
    category_dict = create_category_dict(
        "../static/data/csv/rating_reviews.csv")

    print("Available business types:")
    for business_type in sorted(category_dict.keys()):
        print(f"- {business_type}")

    chosen_category = input("Enter the name of the desired business type: ")
    print('----------------------------------')
    if chosen_category not in category_dict:
        print("Invalid business type. Please choose from the available options.")
        return

    top_restaurant, top_restaurant_city = find_top_restaurant_in_category(
        category_dict, chosen_category)
    if not top_restaurant:
        print(f"No top restaurant found for the category '{chosen_category}'.")
        return

    print(f"City: {top_restaurant['city']}\nBusiness Name: {top_restaurant['Business Name']}\nAddress: {top_restaurant['Address']}\nRating: {top_restaurant['Rating']}\nReviews: {top_restaurant['Reviews']}")
    # ____path to get the datset for cities demographics________

    similar_cities = finding_similar_cities(
        "../static/data/csv/clean_merged_iowa_demo_with_description.csv", top_restaurant_city)
    print("Top 5 similar cities:")
    for city in similar_cities:
        print(f"- {city}")


if __name__ == "__main__":
    main()
