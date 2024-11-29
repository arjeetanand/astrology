from flask import Flask, render_template, request
from api.swiss_ephemeris_api import (
    get_planetary_positions,
    get_zodiac_sign,
    cohere_generate_personality_insights,
    cohere_generate_life_event_predictions,
    cohere_analyze_past_events,
)


# Helper function to parse AI output
def parse_personality_insights(insights_str):
    """
    Converts plain text personality insights response into a dictionary.
    """
    insights = {}
    for line in insights_str.split("\n"):
        if ": " in line:
            planet, trait = line.split(": ", 1)
            insights[planet.strip()] = trait.strip()
    return insights


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get form inputs
        birth_date = request.form["birth_date"]
        birth_time = request.form["birth_time"]
        birth_place = request.form["birth_place"]

        # Calculate planetary positions
        positions = get_planetary_positions(birth_date, birth_time, birth_place)

        # Map positions to zodiac signs
        zodiac_positions = {
            planet: get_zodiac_sign(pos) for planet, pos in positions.items()
        }

        # Generate personality insights using AI and parse output
        personality_insights_str = cohere_generate_personality_insights(
            zodiac_positions
        )
        personality_insights = parse_personality_insights(personality_insights_str)

        # Generate life event predictions using AI
        life_event_predictions = cohere_generate_life_event_predictions(positions)

        # Example past date for analysis
        past_positions = get_planetary_positions("2020-03-27", "11:34", birth_place)
        past_insights = cohere_analyze_past_events(past_positions, positions)

        print("Personality Insights:", personality_insights)
        print("Life Event Predictions:", life_event_predictions)
        print("Past Analysis:", past_insights)

        # Render result template
        return render_template(
            "result.html",
            positions=positions,
            zodiac_positions=zodiac_positions,
            personality_insights=personality_insights,
            life_event_predictions=life_event_predictions,
            past_insights=past_insights,
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
