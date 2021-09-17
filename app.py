from flask import Flask, render_template, url_for, request
import pickle
import joblib

filename = 'pickle.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('transform.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predit():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_perdiction = clf.predict(vect)
    return render_template('result.html', prediction = my_perdiction)

if __name__ == '__main__':
    app.run()