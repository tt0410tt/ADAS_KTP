from flask import Blueprint, render_template

menu1_bp = Blueprint('menu1', __name__, url_prefix='/menu1')
menu2_bp = Blueprint('menu2', __name__, url_prefix='/menu2')
menu3_bp = Blueprint('menu3', __name__, url_prefix='/menu3')

@menu1_bp.route('/camera-view')
def camera_view():
    return render_template('menu1/camera-view.html')

@menu2_bp.route('/setting')
def setting():
    return render_template('menu2/setting.html')
    
@menu3_bp.route('/sub1/board')
def board():
    return render_template('menu3/sub1/board.html')
