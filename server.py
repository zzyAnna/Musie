import time

import scipy
from flask import Flask, request, jsonify, send_file, render_template
from transformers import AutoProcessor, MusicgenForConditionalGeneration
# this is a comment 
app = Flask(__name__,static_folder="",template_folder="")

# Load the processor and model
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")


def generate_music(prompt):
    start = time.time()
    print(start, '---start---')

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs, max_new_tokens=512)
    sampling_rate = model.config.audio_encoder.sampling_rate
    
    filename = f"{prompt[:15].replace(' ', '_')}.wav"
    scipy.io.wavfile.write(filename, rate=sampling_rate, data=audio_values[0, 0].numpy())
    print(f'{time.time() - start}', '---end----')

    return filename

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/generate-music', methods=['POST'])
def generate_music_endpoint():
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        filename = generate_music(prompt)
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5020)
