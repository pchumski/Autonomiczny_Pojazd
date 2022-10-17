from flask import Flask, redirect, url_for, render_template

app = Flask(__name__) # Tworzenie strony na serwerze developera

# strona bazowa Home
@app.route("/")
def home():
    return render_template("idn.html")

# Akcja po wpisaniu /<cos> 
@app.route("/<name>")
def user(name):
    return render_template("index.html", content= name) # przekazanie zmiennej na strone

# Administrator page
@app.route("/admin")
def admin():
    return redirect(url_for("user", name= "Admin!"))

# HTML template (wazne zeby nazwac folder templates)
if __name__ == "__main__":
    app.run()

