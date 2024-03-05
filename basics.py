from flask import Flask,render_template

app = Flask(__name__,template_folder='template')

@app.route('/index.html')

def home():
  return render_template('index.html')

if __name__ == 'main':
  app.run(debug=True)


