from flask import Flask, request, render_template, send_from_directory, jsonify
import joblib
import pandas as pd
import os
import json

app = Flask(__name__)

# Load the trained model
model = joblib.load('speech_disorder_model.pkl')

# Load disorder mapping translations
def load_disorder_mapping(lang):
    with open(f'locales/{lang}.json', 'r', encoding='utf-8') as file:
        translations = json.load(file)
    return translations.get('disorder_mapping', {})

@app.route('/')
def home():
    return render_template('language.html')

@app.route('/select_language', methods=['GET'])
def select_language():
    lang = request.args.get('lang', 'en')  # Default to 'en' if no language is selected
    return render_template(f'{lang}/index.html', lang=lang)

@app.route('/predict', methods=['POST'])
def predict():
    lang = request.args.get("lang", "en")  # Get the selected language
    
    try:
        # Get data from the form
        age = float(request.form['age'])
        speech_rate = float(request.form['speech_rate'])
        pronunciation_difficulty = int(request.form['pronunciation_difficulty'])
        sound_substitution = int(request.form['sound_substitution'])
        word_repetition = int(request.form['word_repetition'])
        speech_pauses = int(request.form['speech_pauses'])
        nasal_speech = int(request.form['nasal_speech'])
        monotone_speech = int(request.form['monotone_speech'])
    except ValueError:
        return render_template(f'{lang}/index.html', result="Invalid input. Please enter correct values.")

    # Create a DataFrame for the new data
    new_data = pd.DataFrame({
        'age': [age],
        'speech_rate': [speech_rate],
        'pronunciation_difficulty': [pronunciation_difficulty],
        'sound_substitution': [sound_substitution],
        'word_repetition': [word_repetition],
        'speech_pauses': [speech_pauses],
        'nasal_speech': [nasal_speech],
        'monotone_speech': [monotone_speech]
    })

    # Predict the disorder
    prediction = model.predict(new_data)[0]

    # Load the translated disorder mappings
    disorder_mapping = load_disorder_mapping(lang)

    # Get the disorder name based on prediction
    result = disorder_mapping.get(str(prediction), "Unknown disorder")

    # Return the result to the page
    return render_template(f'{lang}/index.html', result=result)

@app.route('/locales/<lang>.json')
def locales(lang):
    return send_from_directory('locales', f'{lang}.json')

if __name__ == '__main__':
    app.run(debug=True)
