from flask import Blueprint, render_template, request
from app.ml.Recommendation_system import RestaurantRecommendationSystem

main = Blueprint('main', __name__)
recommendation_system = RestaurantRecommendationSystem(
    "app/static/data/csv/all_state_rating_rev_common_cities.csv",
    "app/static/data/csv/all_state_demo_common_cities.csv"
)


@main.route('/')
def index():
    business_types = recommendation_system.categories
    return render_template('index.html', business_types=business_types)


@main.route('/recommend', methods=['GET', 'POST'])
def recommend():
    business_types = recommendation_system.categories
    if request.method == 'POST':
        chosen_category = request.form['category']
        # Default to 5 if not provided
        top_n = int(request.form.get('top_n', 5))
        top_restaurants_info = recommendation_system.find_top_restaurants_info(
            chosen_category, top_n)
        return render_template('recommendation.html', top_restaurants_info=top_restaurants_info, top_n=top_n, recommendation_system=recommendation_system)
    return render_template('index.html', business_types=business_types)


@main.route('/about')
def about():
    return render_template('about.html')


@main.route('/doc')
def doc():
    return render_template('doc.html')
