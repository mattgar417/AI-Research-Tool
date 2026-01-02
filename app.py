from flask import Flask, render_template, request, send_file, abort
import os, shutil, tempfile
from datetime import datetime
from tools import extract_data_from_pdf, extract_tables_from_pdf

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["OUTPUT_FOLDER"] = "static/outputs"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".pdf"):
            filename = os.path.splitext(file.filename)[0]
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(upload_path)

            # create output folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(app.config["OUTPUT_FOLDER"], f"{filename}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

            # Run processing pipeline
            tables = extract_tables_from_pdf(upload_path, output_dir)
            data = extract_data_from_pdf(upload_path, output_dir)

            # Prepare for template
            web_output_dir = os.path.relpath(output_dir, start="static").replace("\\", "/")
            dir_name = os.path.basename(output_dir)

            return render_template(
                "results.html",
                figures=data["figures"],
                summary=data["summary"],
                tables=tables,
                output_dir=output_dir,
                web_output_dir=web_output_dir,
                dir_name=dir_name,
            )
    return render_template("index.html")


@app.route("/download/<dir_name>")
def download(dir_name):
    """Serve all extracted files as a ZIP archive."""
    dir_path = os.path.join(app.config["OUTPUT_FOLDER"], dir_name)
    if not os.path.exists(dir_path):
        abort(404)

    tmpdir = tempfile.mkdtemp()
    archive_base = os.path.join(tmpdir, "archive")
    zip_path = shutil.make_archive(archive_base, "zip", dir_path)
    return send_file(zip_path, as_attachment=True, download_name=f"{dir_name}.zip")


if __name__ == "__main__":

    app.run(debug=True)
