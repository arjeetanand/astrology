import cohere
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

import swisseph as swe
from datetime import datetime


def get_planetary_positions(birth_date, birth_time, birth_place="Default Location"):
    """
    Calculate planetary positions based on birth details.
    """
    # Ensure time has seconds
    if len(birth_time.split(":")) == 2:
        birth_time += ":00"

    # Convert strings to datetime
    birth_datetime = datetime.strptime(
        f"{birth_date} {birth_time}", "%Y-%m-%d %H:%M:%S"
    )

    # Calculate Julian day
    jd = swe.julday(birth_datetime.year, birth_datetime.month, birth_datetime.day)

    planets = [
        "Sun",
        "Moon",
        "Mercury",
        "Venus",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
        "Pluto",
    ]
    positions = {}

    # Fetch planetary positions
    for planet in planets:
        pos, _ = swe.calc_ut(jd, getattr(swe, planet.upper(), swe.SUN))
        positions[planet] = pos[0]  # Store longitude

    return positions


def get_zodiac_sign(longitude):
    """
    Map planetary longitude to a zodiac sign.

    Args:
        longitude (float): Planetary longitude in degrees.

    Returns:
        str: Corresponding zodiac sign.
    """
    zodiac_signs = [
        "Aries",
        "Taurus",
        "Gemini",
        "Cancer",
        "Leo",
        "Virgo",
        "Libra",
        "Scorpio",
        "Sagittarius",
        "Capricorn",
        "Aquarius",
        "Pisces",
    ]
    index = int(longitude // 30)  # Divide longitude by 30 to find the zodiac sign index
    return zodiac_signs[index]


def calculate_ascendant(birth_date, birth_time, birth_place_lat, birth_place_lon):
    """
    Calculate the Ascendant (Lagna) based on birth details.
    """
    # Ensure time has seconds
    if len(birth_time.split(":")) == 2:
        birth_time += ":00"

    # Convert strings to datetime
    birth_datetime = datetime.strptime(
        f"{birth_date} {birth_time}", "%Y-%m-%d %H:%M:%S"
    )

    # Calculate Julian day
    jd = swe.julday(
        birth_datetime.year,
        birth_datetime.month,
        birth_datetime.day,
        birth_datetime.hour + birth_datetime.minute / 60,
    )

    # Calculate sidereal time
    sidereal_time = swe.sidtime(jd)

    # Calculate Ascendant using geographic coordinates of the birth place
    asc, ascmc = swe.houses(jd, birth_place_lat, birth_place_lon, b"A")
    ascendant_sign = get_zodiac_sign(asc[0])

    return ascendant_sign


def cohere_generate_personality_insights(zodiac_positions):
    """
    Use Cohere to generate personality insights based on zodiac positions.
    """
    prompt = "Provide detailed personality traits based on the following planetary zodiac positions:\n"
    for planet, sign in zodiac_positions.items():
        prompt += f"{planet}: {sign}\n"

    prompt += "Output each planet and its traits in the format: Planet: Traits."

    response = co.generate(
        model="command-xlarge",
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
    )
    return response.generations[0].text.strip()


def cohere_generate_life_event_predictions(positions):
    """
    Use Cohere to generate life event predictions based on planetary positions.
    """
    prompt = (
        "Generate life event predictions based on the following planetary longitudes:\n"
    )
    for planet, position in positions.items():
        prompt += f"{planet}: {position} degrees\n"

    prompt += "Provide predictions for career, relationships, and personal growth."

    response = co.generate(
        model="command-xlarge",
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
    )
    return response.generations[0].text.strip()


def house_analysis(positions):
    """
    Analyze planetary positions in each of the 12 houses.
    """
    # Placeholder logic for calculating which planets are in which house.
    # A real implementation would require computing the house cusps based on the Ascendant.
    house_positions = {}
    for i in range(1, 13):
        house_positions[f"House {i}"] = []

    # Assuming planets are evenly distributed for this example.
    for index, (planet, pos) in enumerate(positions.items()):
        house_number = (index % 12) + 1
        house_positions[f"House {house_number}"].append(planet)

    return house_positions


def cohere_analyze_past_events(past_positions, natal_positions):
    """
    Use Cohere to analyze past celestial events compared to natal planetary positions.
    """
    prompt = "Analyze the following celestial events and their significance:\n"
    prompt += "Past positions:\n"
    for planet, position in past_positions.items():
        prompt += f"{planet}: {position} degrees\n"
    prompt += "Natal positions:\n"
    for planet, position in natal_positions.items():
        prompt += f"{planet}: {position} degrees\n"

    prompt += "Provide insights about significant life events and celestial alignments."

    response = co.generate(
        model="command-xlarge",
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
    )
    return response.generations[0].text.strip()
