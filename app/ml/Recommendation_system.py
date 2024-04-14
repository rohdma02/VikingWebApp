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


class RestaurantRecommendationSystem:
    def __init__(self, restaurant_data_file, demographic_data_file):
        self.restaurants_common = pd.read_csv(restaurant_data_file)
        self.all_state_demo = pd.read_csv(demographic_data_file)
        self.categories = sorted(
            self.restaurants_common["Category Title"].unique())
        self.iowa_demo = self.all_state_demo[self.all_state_demo["State"] == "Iowa"]
        self.iowa_indices = self.iowa_demo.index.tolist()
        self.category_dict = self.create_category_dict()

    def create_category_dict(self):
        category_dict = {}
        for category in self.restaurants_common["Category Title"].unique():
            category_df = self.restaurants_common[self.restaurants_common["Category Title"] == category]
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

    def analyze_sentiment(self, comment):
        analysis = TextBlob(str(comment))
        return 'positive' if analysis.sentiment.polarity > 0 else ('negative' if analysis.sentiment.polarity < 0 else 'neutral')

    def find_top_restaurants_info(self, category, top_n):
        top_restaurants = self.find_top_restaurants_in_category(category)
        return [
            {
                "Name": restaurant["Business Name"],
                "City": restaurant["city"],
                "State": restaurant["state"]
            }
            for restaurant in top_restaurants[:top_n]
        ]

    def find_top_restaurants_in_category(self, category):
        if category in self.category_dict:
            return self.category_dict[category]
        else:
            return []

    def find_similar_cities_in_iowa(self, city_name, top_n):
        city_index = self.all_state_demo.index[self.all_state_demo['City'] == city_name].tolist()[
            0]
        cos_sim_matrix = self.compute_cosine_similarity()
        top_similar_indices = np.argsort(
            cos_sim_matrix[city_index])[::-1][:top_n]
        return [self.all_state_demo.iloc[self.iowa_indices[idx]]['City'] for idx in top_similar_indices]

    def compute_cosine_similarity(self):
        all_state_bert_embeddings = self.get_bert_embeddings(
            self.all_state_demo["wiki_description"].fillna(""))
        scaler = StandardScaler()
        numerical_features = [col for col in self.all_state_demo.columns if col !=
                              'City' and self.all_state_demo[col].dtype in [np.int64, np.float64]]
        all_state_scaled_features = scaler.fit_transform(
            self.all_state_demo[numerical_features])
        all_state_combined_features = np.hstack(
            (all_state_scaled_features, all_state_bert_embeddings))
        pca = PCA(n_components=64)
        all_state_data_pca = pca.fit_transform(all_state_combined_features)
        return cosine_similarity(all_state_data_pca, all_state_data_pca[self.iowa_indices])

    def get_bert_embeddings(self, sentences):
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

    def on_category_change(self, chosen_category, top_n):
        top_restaurants_info = self.find_top_restaurants_info(
            chosen_category, top_n)

        for i, restaurant_info in enumerate(top_restaurants_info, 1):
            restaurant_name = restaurant_info['Name']
            restaurant_city = restaurant_info['City']
            restaurant_state = restaurant_info['State']

            similar_cities = self.find_similar_cities_in_iowa(
                restaurant_city, top_n)

            print(
                f"{i}. Restaurant: {restaurant_name} in {restaurant_city}, {restaurant_state}")
            print(f"   Top {top_n} similar cities in Iowa:")
            for j, similar_city in enumerate(similar_cities, 1):
                print(f"   {j}. {similar_city}")
            print()


if __name__ == "__main__":
    recommendation_system = RestaurantRecommendationSystem(
        "../static/data/csv/all_state_rating_rev_common_cities.csv", "../static/data/csv/all_state_demo_common_cities.csv")

    print("Available business types:")
    for business_type in sorted(recommendation_system.categories):
        print(f"- {business_type}")

    chosen_category = input("Enter the name of the desired business type: ")
    top_n = int(input("Enter the number of top restaurants to show: "))

    print('----------------------------------')

    recommendation_system.on_category_change(chosen_category, top_n)
