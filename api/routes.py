from flask import Blueprint, jsonify, request, current_app, render_template, send_file, abort
import os, shutil, tempfile
from datetime import datetime
from db_connection import get_db
from mysql.connector import Error
from ml.tools import text_analysis, table_analysis, figure_analysis, compare_papers, entity_recognition

# Create a blueprint for routes
routes = Blueprint("routes", __name__)

# For uploading a file
@routes.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    if file.filename.endswith(".pdf"):
        cursor = get_db().cursor(dictionary=True)
        try:
            # Get file information
            filename = os.path.splitext(file.filename)[0]
            upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], file.filename)
            file.save(upload_path)

            # Check if file already is in db
            cursor.execute("SELECT * FROM File WHERE Name = %s and Path = %s", (filename, upload_path))
            existing_file = cursor.fetchone()

            if existing_file:
                return jsonify({"message": "Paper already added"}), 400

            # Add file path to db
            cursor.execute("INSERT INTO Paper (Name, Path) VALUES (%s, %s)", (filename, upload_path))
            paper_id = cursor.lastrowid

            # Create new output folder for paper being analyzed
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(current_app.config["OUTPUT_FOLDER"], f"{filename}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

            # Add folder path to db
            cursor.execute("INSERT INTO Results (PaperID, Path) VALUES (%s, %s)", (paper_id, output_dir))

            get_db().commit()
            return jsonify({"message": "New paper inserted!"}), 200
        except Error as e:
            current_app.logger.error(f'Database error in upload_file: {e}')
            return jsonify({"error": str(e)}), 500
        finally:
            cursor.close()
        
    else:
        return jsonify({"message": "Please upload a pdf"}), 400

        '''
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
        '''
    
# For figure analysis
@routes.route("/analyze/figures", methods=["POST"])
def analyze_figures(name, prompt):
    data = request.get_json()
    name = data.get("name")
    prompt = data.get("prompt")

    cursor = get_db().cursor(dictionary=True)
    try:
        cursor.execute("SELECT Path, PaperID FROM Paper WHERE Name = %s", (name))
        paper = cursor.fetchone()
        upload_path = paper["Path"]
        paperID = paper["PaperID"]

        cursor.execute("SELECT Path FROM Results WHERE PaperID = %s", (paperID))
        result = cursor.fetchone()
        output_dir = result["Path"]

        figure_summary = figure_analysis(upload_path, output_dir, prompt)

        cursor.execute("INSERT INTO FigureSummary (PaperID, Summary) VALUES (%s, %s)", (paperID, str(figure_summary)))
        get_db().commit()
        return jsonify({"message": "Figure data extracted!"}), 200
    except Error as e:
            current_app.logger.error(f'Database error in analyze_figures: {e}')
            return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()

# For table analysis
@routes.route("/analyze/tables", methods=["POST"])
def analyze_tables():
    data = request.get_json()
    name = data.get("name")
    prompt = data.get("prompt")

    cursor = get_db().cursor(dictionary=True)
    try:
        cursor.execute("SELECT Path, PaperID FROM Paper WHERE Name = %s", (name))
        paper = cursor.fetchone()
        upload_path = paper["Path"]
        paperID = paper["PaperID"]

        cursor.execute("SELECT Path FROM Results WHERE PaperID = %s", (paperID))
        result = cursor.fetchone()
        output_dir = result["Path"]

        table_summary = table_analysis(upload_path, output_dir, prompt)

        cursor.execute("INSERT INTO TableSummary (PaperID, Summary) VALUES (%s, %s)", (paperID, str(table_summary)))
        get_db().commit()
        return jsonify({"message": "Table data extracted!"}), 200
    except Error as e:
            current_app.logger.error(f'Database error in analyze_figures: {e}')
            return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()

# For text analysis
@routes.route("/analyze/text", methods=["POST"])
def analyze_text():
    data = request.get_json()
    name = data.get("name")
    prompt = data.get("prompt")

    cursor = get_db().cursor(dictionary=True)
    try:
        cursor.execute("SELECT Path, PaperID FROM Paper WHERE Name = %s", (name))
        paper = cursor.fetchone()
        upload_path = paper["Path"]
        paperID = paper["PaperID"]

        cursor.execute("SELECT Path FROM Results WHERE PaperID = %s", (paperID))
        result = cursor.fetchone()
        output_dir = result["Path"]

        text_summary  = text_analysis(upload_path, output_dir, prompt)

        cursor.execute("INSERT INTO TextSummary (PaperID, Summary) VALUES (%s, %s)", (paperID, str(text_summary)))
        get_db().commit()
        return jsonify({"message": "Text data extracted!"}), 200
    except Error as e:
            current_app.logger.error(f'Database error in analyze_text: {e}')
            return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()

# For named entity recognition
@routes.route("/analyze/entities", methods=["POST"])
def named_entity_recognition():
    data = request.get_json()
    name = data.get("name")

    cursor = get_db().cursor(dictionary=True)
    try:
        cursor.execute("SELECT Path, PaperID FROM Paper WHERE Name = %s", (name))
        paper = cursor.fetchone()
        upload_path = paper["Path"]
        paperID = paper["PaperID"]

        cursor.execute("SELECT Path FROM Results WHERE PaperID = %s", (paperID))
        result = cursor.fetchone()
        output_dir = result["Path"]

        results = entity_recognition(upload_path, output_dir)

        cursor.execute("INSERT INTO TextSummary (PaperID, Summary) VALUES (%s, %s)", (paperID, str(results)))
        get_db().commit()
        return jsonify({"message": "Text data extracted!"}), 200
    except Error as e:
            current_app.logger.error(f'Database error in analyze_text: {e}')
            return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()

# For paper comparison
@routes.route("/compare", methods=["POST"])
def compare_two_papers():
    data = request.get_json()
    name1 = data.get("name1")
    name2 = data.get("name2")

    cursor = get_db().cursor(dictionary=True)
    try:
        cursor.execute("SELECT Path, PaperID FROM Paper WHERE Name = %s", (name1))
        paper = cursor.fetchone()
        upload_path1 = paper["Path"]
        paperID1 = paper["PaperID"]

        cursor.execute("SELECT Path, PaperID FROM Paper WHERE Name = %s", (name2))
        paper = cursor.fetchone()
        upload_path2 = paper["Path"]
        paperID2 = paper["PaperID"]

        cursor.execute("SELECT Path FROM Results WHERE PaperID = %s", (paperID1))
        result = cursor.fetchone()
        output_dir = result["Path"]

        comparison_results = compare_papers(upload_path1, upload_path2, output_dir)

        cursor.execute("INSERT INTO TextSummary (PaperID, Summary) VALUES (%s, %s)", (paperID1, comparison_results))
        get_db().commit()
        return jsonify({"message": "Text data extracted!"}), 200
    except Error as e:
            current_app.logger.error(f'Database error in analyze_text: {e}')
            return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()

# Index
@routes.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# For downloading directory of extracted files
@routes.route("/download/<dir_name>")
def download(dir_name):
    """Serve all extracted files as a ZIP archive."""
    dir_path = os.path.join(current_app.config["OUTPUT_FOLDER"], dir_name)
    if not os.path.exists(dir_path):
        abort(404)

    tmpdir = tempfile.mkdtemp()
    archive_base = os.path.join(tmpdir, "archive")
    zip_path = shutil.make_archive(archive_base, "zip", dir_path)
    return send_file(zip_path, as_attachment=True, download_name=f"{dir_name}.zip")