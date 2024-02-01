from flask import Flask, render_template
from flask import Flask, request, send_from_directory
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

def chat(user_text):
    

    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return ("{}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))



app = Flask(__name__)

UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.add_url_rule(
    "/uploads/<name>", endpoint="download_file", build_only=True
)

@app.route("/")
def home():
    return render_template("home.html")
    
@app.route("/style_image")
def style_image():
    return render_template("style_image.html")

# @app.route("/content_summerizer")
# def content_summerizer():
#     return render_template("content_summerizer.html")

@app.route('/content_summerizer', methods=['GET', 'POST'])
def content_summerizer():
    from content_summerizer import run
    summary = ""
    if request.method == 'POST':
        title = request.form.get('title')
        length = int(request.form.get('length'))
        summary = run(title, length)
    return render_template('content_summerizer.html', summary=summary)

@app.route('/image_generator', methods=['GET', 'POST'])
def image_generator():
    from image_generator import generate
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        generate(prompt)
    return render_template('image_generator.html')

@app.route('/question_image', methods=['GET', 'POST'])
def question_image():
    from question_image import run
    answer = ""
    if request.method == 'POST':
        question = request.form.get('question')
        answer = run(question)
    return render_template('question_image.html', answer=answer)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload/<file_id>/<template>', methods=['GET', 'POST'])
def upload_file(file_id, template):
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Ensure the UPLOAD_FOLDER exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_id))
            # return redirect(url_for('download_file', name=filename))
        if template =='q':
            return render_template("question_image.html")

    return render_template("style_image.html")

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route('/style', methods=['GET', 'POST'])
def style():
    import model
    if request.method == 'POST':
        # Ensure the scripts/images folder exists
        os.makedirs('./scripts/images', exist_ok=True)
        # Call the execute function from model.py
        model.execute()
        return redirect(url_for('style_image'))
    return render_template("style_image.html")

@app.route('/styled_image')
def styled_image():
    # Assuming the execute function always creates an image with the name 'styled_image.jpg'
    return send_from_directory('./scripts/images', 'styled_image.jpg')
    

@app.route('/remove')
def remove():
    folder = './static/images/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return render_template("style_image.html")

@app.route('/chat_bot')
def chat_bot():
    return render_template('chat_bot.html')

@app.route('/get')
def get_bot_response():
    user_text = request.args.get('msg')
    response = chat(user_text)
    return str(response)

if __name__ == "__main__":
    app.run(debug=True)
