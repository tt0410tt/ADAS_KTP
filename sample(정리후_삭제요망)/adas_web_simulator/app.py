from flask import Flask
from python.routes.main_routes import main_bp
from python.routes.menu_routes import menu1_bp, menu2_bp, menu3_bp
from python.feature.calculator import calculate

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(main_bp)
app.register_blueprint(menu1_bp)
app.register_blueprint(menu2_bp)
app.register_blueprint(menu3_bp)

@app.route('/calculate', methods=['POST'])
def calculate_route():
    from flask import request, jsonify
    data = request.json
    num1 = data.get('num1', 0)
    num2 = data.get('num2', 0)
    operation = data.get('operation')

    result = calculate(num1, num2, operation)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
