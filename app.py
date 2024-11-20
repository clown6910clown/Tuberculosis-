from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session to store history
model = load_model(r"C:\Users\Shrishti\Desktop\DPEDAL_Project\tb_cnn_model.h5")  # Update this with the path to your model

# Dummy user for simplicity (use a database for real applications)
users = {"user@example.com": "password123"}

@app.route("/")
def home():
    # Directly redirect to the login page
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if email in users and users[email] == password:
            session['logged_in'] = True
            session['user'] = email
            return redirect(url_for("predict_page"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    session.pop('user', None)
    return redirect(url_for("login"))

@app.route("/predict_page")
def predict_page():
    # Check if the user is logged in; otherwise redirect to login
    if 'logged_in' not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# Other routes remain the same...


@app.route("/predict", methods=["POST"])
def predict():
    if 'logged_in' not in session:
        return redirect(url_for("login"))
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        # Process the image and convert to RGB
        img = Image.open(file).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape((1, 224, 224, 3))

        # Make a prediction
        prediction = model.predict(img_array)
        probability = prediction[0][0]
        result = "TB Detected" if probability > 0.5 else "No TB Detected"

        # Add result to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            "filename": file.filename,
            "timestamp": timestamp,
            "result": result,
            "probability": f"{probability:.2%}"
        }
        session.setdefault('history', []).append(history_entry)

        # Return prediction result
        return jsonify({
            "result": result,
            "probability": f"{probability:.2%}"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/history")
def history():
    if 'logged_in' not in session:
        return redirect(url_for("login"))
    return jsonify(session.get('history', []))

@app.route("/feedback", methods=["POST"])
def feedback():
    if 'logged_in' not in session:
        return redirect(url_for("login"))
    
    feedback_text = request.form.get("feedback")
    if feedback_text:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session.setdefault('feedbacks', []).append({"timestamp": timestamp, "feedback": feedback_text})
        return jsonify({"message": "Feedback received"})
    return jsonify({"error": "No feedback provided"})

@app.route("/download_report/<int:report_id>")
def download_report(report_id):
    if 'logged_in' not in session:
        return redirect(url_for("login"))
    
    try:
        # Fetch the report details from the history
        report_data = session['history'][report_id]
        filename = f"Report_{report_data['filename']}_{report_id}.pdf"

        # Generate PDF report
        pdf_path = f"/tmp/{filename}"  # Ensure the /tmp directory exists or change this path

        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.drawString(100, 750, "TB Prediction Report")
        c.drawString(100, 720, f"Filename: {report_data['filename']}")
        c.drawString(100, 700, f"Prediction: {report_data['result']}")
        c.drawString(100, 680, f"Probability: {report_data['probability']}")
        c.drawString(100, 660, f"Date: {report_data['timestamp']}")
        c.save()

        # Send the PDF as an attachment
        return send_file(pdf_path, as_attachment=True, download_name=filename)

    except IndexError:
        return jsonify({"error": "Invalid report ID"}), 404
    except Exception as e:
        return jsonify({"error": f"Error generating report: {str(e)}"}), 500


@app.route("/download_feedbacks")
def download_feedbacks():
    if 'logged_in' not in session:
        return redirect(url_for("login"))
    
    # Generate feedback CSV
    feedback_df = pd.DataFrame(session.get('feedbacks', []))
    csv_path = "/tmp/feedbacks.csv"
    feedback_df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True, download_name="feedbacks.csv")

if __name__ == "__main__":
    # Create tmp directory if not exists for saving files
    os.makedirs("/tmp", exist_ok=True)
    app.run(debug=True)
