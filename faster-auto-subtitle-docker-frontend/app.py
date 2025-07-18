from flask import Flask, render_template, request, send_file, after_this_request
import requests
import os

app = Flask(__name__)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BACKEND_URL = "http://backend:8000/generate-subtitles/"  # Use `backend` if using Docker Compose networking

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model = request.form.get("model", "base")
        language = request.form.get("language", "auto")
        file = request.files["file"]

        if file:
            original_filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, original_filename)
            file.save(file_path)

            with open(file_path, "rb") as f:
                response = requests.post(
                    BACKEND_URL,
                    files={"file": (original_filename, f)},
                    data={"model": model, "language": language}
                )

            base_filename = os.path.splitext(original_filename)[0]
            srt_path = os.path.join(UPLOAD_FOLDER, f"{base_filename}.srt")

            with open(srt_path, "wb") as out:
                out.write(response.content)

            @after_this_request
            def cleanup(response):
                try:
                    os.remove(file_path)
                    os.remove(srt_path)
                except Exception as e:
                    app.logger.warning(f"Cleanup failed: {e}")
                return response

            return send_file(
                srt_path,
                as_attachment=True,
                download_name=f"{base_filename}.srt",
                mimetype="application/x-subrip"
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
