from flask import Blueprint, render_template, request
from app.ml.recommendation import create_category_dict, find_top_restaurants_info, find_similar_cities_in_iowa, load_bert_embeddings, read_data, clean_numerical_data, get_bert_embeddings, combine_features, apply_pca, cosine_similarity, save_bert_embeddings
import os

main = Blueprint('main', __name__)

# Set the base directory path
base_dir = os.path.abspath(os.path.dirname(__file__))

# Read the data
restaurants_common = read_data(os.path.join(
    base_dir, "static/data/restaurants/all_state_rating_rev_common_cities.csv"))
all_state_demo = read_data(os.path.join(
    base_dir, "static/data/demographics/all_state_demo_common_cities.csv"))
all_state_demo = clean_numerical_data(all_state_demo)

# Get BERT embeddings for all_state_demo data
all_state_wiki_descriptions = all_state_demo["wiki_description"].fillna("")
bert_folder_path = os.path.join(base_dir, "static/data/bert_embeddings")
if os.path.exists(bert_folder_path):
    all_state_wiki_descriptions, all_state_bert_embeddings = load_bert_embeddings(
        bert_folder_path)
else:
    all_state_bert_embeddings = get_bert_embeddings(
        all_state_wiki_descriptions)
    save_bert_embeddings(all_state_wiki_descriptions,
                         all_state_bert_embeddings, bert_folder_path)

# Filter data for Iowa
iowa_demo = all_state_demo[all_state_demo["State"] == "Iowa"]
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
categories = sorted(category_dict.keys())


@main.route('/')
def index():
    return render_template('index.html', business_types=categories)


@main.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        chosen_category = request.form['category']
        top_n = int(request.form.get('top_n', 5))

        top_restaurants_info = find_top_restaurants_info(
            category_dict, chosen_category, top_n)

        if top_restaurants_info:
            similar_cities = []
            for restaurant_info in top_restaurants_info:
                restaurant_city = restaurant_info['City']
                similar_cities_data = find_similar_cities_in_iowa(
                    restaurant_city, all_state_demo, iowa_indices, cos_sim_matrix, 3)
                similar_cities.append(similar_cities_data)

            return render_template('recommendation.html', top_restaurants_info=top_restaurants_info, similar_cities=similar_cities, top_n=top_n)
        else:
            return render_template('recommendation.html', error_message=f"No restaurants found for the business type: {chosen_category}")

    return render_template('index.html', business_types=categories)


@main.route('/about')
def about():
    return render_template('about.html')


@main.route('/doc')
def doc():
    return render_template('doc.html')
