import os
import pickle
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import re


def read_data(file_path):
    return pd.read_csv(file_path)

# ______________Cleanning numerical data________________


def clean_numerical_data(data):
    numerical_columns = [
        col for col in data.columns if data[col].dtype in [np.int64, np.float64]]
    data[numerical_columns] = data[numerical_columns].apply(
        lambda x: pd.to_numeric(x, errors='coerce'))
    return data

# ______________Function to save BERT embeddings to a folder_____________


def save_bert_embeddings(all_state_wiki_descriptions, all_state_bert_embeddings, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, 'wiki_descriptions.pkl'), 'wb') as f:
        pickle.dump(all_state_wiki_descriptions, f)
    with open(os.path.join(folder_path, 'bert_embeddings.pkl'), 'wb') as f:
        pickle.dump(all_state_bert_embeddings, f)

# _____________ Function to load BERT embeddings from a folder____________


def load_bert_embeddings(folder_path):
    with open(os.path.join(folder_path, 'wiki_descriptions.pkl'), 'rb') as f:
        all_state_wiki_descriptions = pickle.load(f)
    with open(os.path.join(folder_path, 'bert_embeddings.pkl'), 'rb') as f:
        all_state_bert_embeddings = pickle.load(f)
    return all_state_wiki_descriptions, all_state_bert_embeddings

# ______________ Get BERT embeddings for all_state_demo data_______________


def get_bert_embeddings(sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained(
        'bert-base-uncased', output_hidden_states=True)
    input_ids = []
    attention_masks = []
    for sentence in tqdm(sentences, desc="Extracting BERT embeddings"):
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)

    text_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    return text_embeddings

# ________________ Combinning numerical and textutal embeddings for all_state_demo data____________


def combine_features(data, bert_embeddings):
    scaler = StandardScaler()
    numerical_features = [col for col in data.columns if col !=
                          'City' and data[col].dtype in [np.int64, np.float64]]
    scaled_features = scaler.fit_transform(data[numerical_features])
    combined_features = np.hstack((scaled_features, bert_embeddings))
    return combined_features

# _________________ Applying PCA for dimensionality reduction_________________


def apply_pca(combined_features):
    pca = PCA(n_components=64)
    data_pca = pca.fit_transform(combined_features)
    return data_pca


def create_category_dict(data_frame):
    category_dict = {}
    for category in data_frame["Category Title"].unique():
        category_df = data_frame[data_frame["Category Title"] == category]
        restaurant_list = [
            {
                "state": row["State"],
                "city": row["City"],
                "Business Name": row["Business Name"],
                "Address": row["Address"],
                "Rating": row["Rating"],
                "Reviews": row["Combined_Comments"],
                "Population": pd.to_numeric(row["Population estimates base, April 1, 2020, (V2022)"].replace(',', ''), errors='coerce'),
                "High School Graduates": pd.to_numeric(row["High school graduate or higher, percent of persons age 25 years+, 2018-2022"].replace('%', ''), errors='coerce'),
                "Bachelor's Degree or Higher": pd.to_numeric(row["Bachelor's degree or higher, percent of persons age 25 years+, 2018-2022"].replace('%', ''), errors='coerce'),
                "Median Household Income": pd.to_numeric(re.sub(r'[\$,]', '', row["Median household income (in 2022 dollars), 2018-2022"]), errors='coerce'),
                "Per Capita Income": pd.to_numeric(re.sub(r'[\$,]', '', row["Per capita income in past 12 months (in 2022 dollars), 2018-2022"]), errors='coerce')
            }
            for _, row in category_df.iterrows()
        ]
        category_dict[category] = restaurant_list
    return category_dict

# ________________Applying Sentimental Analysis on Reviews_________________


def analyze_sentiment(comment):
    analysis = TextBlob(str(comment))
    return 'positive' if analysis.sentiment.polarity > 0 else ('negative' if analysis.sentiment.polarity < 0 else 'neutral')


def find_top_restaurants_in_category(category_dict, category):
    if category in category_dict:
        return category_dict[category]
    else:
        return []


def find_top_restaurants_info(category_dict, category, top_n):
    top_restaurants = find_top_restaurants_in_category(category_dict, category)
    return [
        {
            "Name": restaurant["Business Name"],
            "City": restaurant["city"],
            "State": restaurant["state"]
        }
        for restaurant in top_restaurants[:top_n]
    ]
# __________________Finding Similar cities in iowa for the top rated city founded in midwest and within iowa


def find_similar_cities_in_iowa(city_name, all_state_demo, iowa_indices, cos_sim_matrix, top_n):
    city_index = all_state_demo.index[all_state_demo['City'] == city_name].tolist()[
        0]
    top_similar_indices = np.argsort(cos_sim_matrix[city_index])[
        ::-1][:top_n + 1]
    similar_cities = []
    for idx in top_similar_indices:
        if idx != city_index:
            similar_city_name = all_state_demo.iloc[iowa_indices[idx]]['City']
            if similar_city_name != city_name:  # Exclude the top restaurant city itself
                similarity_score = cos_sim_matrix[city_index][idx]
                similar_cities.append((similar_city_name, similarity_score))
    return similar_cities[:top_n]


def main():
    # Read the data
    restaurants_common = read_data(
        "data/restaurants/all_state_rating_rev_common_cities.csv")
    all_state_demo = read_data(
        "data/demographics/all_state_demo_common_cities.csv")
    all_state_demo = clean_numerical_data(all_state_demo)

    # Load or get BERT embeddings
    bert_folder_path = "bert_embeddings"
    if os.path.exists(bert_folder_path):
        all_state_wiki_descriptions, all_state_bert_embeddings = load_bert_embeddings(
            bert_folder_path)
    else:
        # Get BERT embeddings for all_state_demo data
        all_state_wiki_descriptions = all_state_demo["wiki_description"].fillna(
            "")
        all_state_bert_embeddings = get_bert_embeddings(
            all_state_wiki_descriptions)
        save_bert_embeddings(all_state_wiki_descriptions,
                             all_state_bert_embeddings, bert_folder_path)

    # Filter data for Iowa
    iowa_demo = all_state_demo[all_state_demo["State"] == "Iowa"]

    # Get the indices of Iowa cities in the original all_state_demo dataframe
    iowa_indices = iowa_demo.index.tolist()

    # Get combined features
    all_state_combined_features = combine_features(
        all_state_demo, all_state_bert_embeddings)

    # Apply PCA for dimensionality reduction
    all_state_data_pca = apply_pca(all_state_combined_features)

    # Compute cosine similarity between all_state_data_pca and iowa_data_pca
    cos_sim_matrix = cosine_similarity(
        all_state_data_pca, all_state_data_pca[iowa_indices])

    # Create category dictionary
    category_dict = create_category_dict(restaurants_common)
    for category in sorted(category_dict.keys()):
        print("-", category)

    while True:

        print("-------------------------")
        business_type = input("Enter the business type (or 'quit' to exit): ")

        if business_type.lower() == 'quit':
            break

        # Finding top restaurants for the entered business type
        top_restaurants_info = find_top_restaurants_info(
            category_dict, business_type, 3)
        if top_restaurants_info:
            print(f"Top 3 restaurants for {business_type}:")
            for i, restaurant_info in enumerate(top_restaurants_info, 1):
                restaurant_name = restaurant_info['Name']
                restaurant_city = restaurant_info['City']
                restaurant_state = restaurant_info['State']
                print(
                    f"{i}. Restaurant: {restaurant_name} in {restaurant_city}, {restaurant_state}")
                similar_cities = find_similar_cities_in_iowa(
                    restaurant_city, all_state_demo, iowa_indices, cos_sim_matrix, 3)
                print(f"Top 3 similar cities in Iowa:")
                for j, (similar_city, similarity_score) in enumerate(similar_cities, 1):
                    print(
                        f"   {j}. {similar_city} (Similarity: {similarity_score})")
        else:
            print(
                f"No restaurants found for the business type: {business_type}")


if __name__ == "__main__":
    main()
