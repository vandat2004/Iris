from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load mô hình đã huấn luyện
model = joblib.load("knn_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form
            sl = float(request.form["sl"])
            sw = float(request.form["sw"])
            pl = float(request.form["pl"])
            pw = float(request.form["pw"])

            # Dự đoán
            sample = [[sl, sw, pl, pw]]
            pred_species = model.predict(sample)
            prediction = pred_species[0]
        except:
            prediction = "❌ Lỗi dữ liệu nhập vào!"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
