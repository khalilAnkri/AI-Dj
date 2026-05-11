from flask import Flask, redirect, url_for

app = Flask(__name__)


@app.route("/admin")
def hello_admin():
    return "Hello Admin"


@app.route("/guest/<guest>")
def hello_guest(guest):
    return f"Hello {guest} as Guest"


@app.route("/user/<name>")
def hello_user(name):
    if name == "admin":
        return redirect(url_for("hello_admin"))
    else:
        return redirect(url_for("hello_guest", guest=name))


@app.route("/")
def hello_world():
    # return 'Hello, World!'
    return "Hello from Flask!"


@app.route("/about")
def about():
    return "This is the about page"


@app.route("/hello/<username>")
def hello(username):
    return f"Hello, {username}!"


if __name__ == "__main__":
    app.run(debug=True)
