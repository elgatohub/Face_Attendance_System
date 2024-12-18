# from flask import Flask, render_template, request, jsonify
# import subprocess

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('atten.html')

# @app.route('/run-script', methods=['POST'])
# def run_script():
#     try:
#         result = subprocess.run(['python', 'Recognize.py'], capture_output=True, text=True)
#         print(result)
#         return result.stdout
#     except Exception as e:
#         return str(e)

# @app.route('/run-script1', methods=['GET','POST'])
# def run_script1():
#     try:
#         result = subprocess.run(['python', 'Capture_Image.py'], capture_output=True, text=True)
#         print(result)
#         return result.stdout
#     except Exception as e:
#         return str(e)




# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, render_template, request
# import subprocess

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('atten.html')

# @app.route('/register')
# def register():
#     return render_template('atten2.html')

# @app.route('/run-script', methods=['POST'])
# def run_script():
#     try:
#         result = subprocess.run(['python', 'Recognize.py'], capture_output=True, text=True)
#         return result.stdout
#     except Exception as e:
#         return str(e)

# @app.route('/run-script1', methods=['POST'])
# def run_script1():
#     try:
#         data = request.get_json()
#         user_id = data['id']
#         user_name = data['name']
#         result = subprocess.run(['python', 'Capture_Image.py'], capture_output=True, text=True)
#         return result.stdout
#     except Exception as e:
#         return str(e)

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('atten.html')

@app.route('/register')
def register():
    return render_template('atten2.html')

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        result = subprocess.run(['python', 'Recognize.py'], capture_output=True, text=True)
        # return result.stdout
        return result.stdout
    except Exception as e:
        return str(e)

@app.route('/run-script1', methods=['POST'])
def run_script1():
    try:
        data = request.get_json()
        user_id = data['Id']
        user_name = data['name']
        print(f"Running Capture_Image.py with ID: {user_id}, Name: {user_name}")  # Debug print
        result = subprocess.run(['python', 'Capture_Image.py', user_id, user_name], capture_output=True, text=True)
        print(result.stdout)  # Debug print
        return result.stdout
    except Exception as e:
        print(f"Error: {str(e)}")  # Debug print
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)





# from flask import Flask, render_template, request, jsonify
# import subprocess
# import json

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('atten.html')

# @app.route('/register')
# def register():
#     return render_template('atten2.html')

# @app.route('/run-script', methods=['POST'])
# def run_script():
#     try:
#         result = subprocess.run(['python', 'Recognize.py'], capture_output=True, text=True)
#         return result.stdout
#     except Exception as e:
#         return str(e)

# @app.route('/run-script1', methods=['POST'])
# def run_script1():
#     try:
#         data = request.get_json()
#         user_id = data.get('id')
#         user_name = data.get('name')
#         result = subprocess.run(['python', 'Capture_Image.py', user_id, user_name], capture_output=True, text=True)
#         print(result)
#         return result.stdout
#     except Exception as e:
#         return str(e)

# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, render_template, redirect, url_for, request, jsonify
# import subprocess

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('atten.html')

# @app.route('/register')
# def register():
#     return render_template('atten2.html')

# @app.route('/run-script', methods=['POST'])
# def run_script():
#     try:
#         result = subprocess.run(['python3', 'Recognize.py'], capture_output=True, text=True)
#         output = result.stdout
#         html_output = f"<div class='output-box'>{output}</div>"
#         return html_output
#     except Exception as e:
#         return str(e)

# @app.route('/run-script1', methods=['POST'])
# def run_script1():
#     try:
#         result = subprocess.run(['python3', 'Capture_Image.py'], capture_output=True, text=True)
#         output = result.stdout.replace("\n", "<br>")
#         html_output = f"<div class='output-box'>{output}</div>"
#         return html_output
#     except Exception as e:
#         return str(e)

# @app.route('/home')
# def home():
#     return redirect(url_for('index'))

# if __name__ == '__main__':
#     app.run(debug=True)



