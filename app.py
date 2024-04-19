from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        # Process the form submission
        country = request.form['country']
        altitude = request.form['altitude']
        region = request.form['region']
        variety = request.form['variety']
        processing_method = request.form['processingMethod']
        color = request.form['color']
        
        # Here you can perform any processing with the form data, such as saving it to a database or performing predictions
        
        # For now, just print the form data
        print("Country of Origin:", country)
        print("Altitude:", altitude)
        print("Region:", region)
        print("Variety:", variety)
        print("Processing Method:", processing_method)
        print("Color:", color)
        
        return 'Form submitted successfully!'
    else:
        return render_template('submit.html')

if __name__ == '__main__':
    app.run(debug=True)
