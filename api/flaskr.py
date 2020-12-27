from flask import Flask
from flask import request
from flask import make_response
from flask import jsonify
from numpy.core.fromnumeric import reshape
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load
import numpy as np

# https://gitlab.com/patkennedy79/flask_user_management_example/-/tree/master


def create_app():
    app = Flask(__name__)
    app.logger.setLevel('INFO')

    @app.route('/')
    def hello_world():
        app.logger.info('asdfghjk')
        return 'Hello, World!'

    @app.route('/gdp-per-capita', methods=['POST'])
    def predict_gpd_per_capita():

        req = request.json
        app.logger.info(req)

        if not req:
            return make_response(
                ({'message': 'unable to parse request'},
                 400,
                 {'Content-Type': 'application/json'})
            )

        # 2 mandatory params plus 5 discovered
        if req and len(req) < 7:
            rsp = make_response(
                jsonify({'message': 'params are missing'}),
                409)

            rsp.headers['Content-Type'] = 'application/json'
            return rsp

        # TODO: make only one time loaded request
        file_name = 'filename5.joblib'
        model = load(file_name)

        # prepare parameters

        prediction_data = prepare_prediction_data(req)
        predicted_result = model.predict(prediction_data)

        return {'gdpPerCapita': predicted_result[0]}

    def prepare_prediction_data(jsonData):
        '''
        PCPI	Inflation, average consumer prices
        PCPIE	Inflation, end of period consumer prices
        LUR     Unemployment rate
        LE      Employment
        LP      Population
        '''
        return [[
            jsonData['PCPI'],
            jsonData['PCPIE'],
            jsonData['LUR'],
            jsonData['LE'],
            jsonData['LP']]]

    return app
