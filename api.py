from flask import Flask, jsonify
from flask_restx import Resource, Api, reqparse, fields
from model import BaseModel
import os
import pickle


app = Flask(__name__)
api = Api(app, title='ML models', description='Heart Attack Analysis & Prediction Dataset')
upload_parser = api.parser()
models = {}

upload_parser.add_argument('model_name', required=True, location='args')
upload_parser.add_argument('model_params', required=False, location='args')

delete = api.model(
    'Delete', {
        'model_id': fields.String(required=True, title='Model id')
    })

train = api.model(
    'Train', {
        'model_id': fields.String(required=True, title='Model id')
    })

predict_sample = api.model(
    'Predict sample', {
        'model_id': fields.String(required=True, title='Model id')
    }
)


@api.route('/model/add')
class Add(Resource):
    @api.expect(upload_parser)
    @api.doc(
        responses={
            200: "Success",
            500: "Select other model (logreg or svc)"
        })
    def post(self):

        args = upload_parser.parse_args()
        model_name = args['model_name']
        clf = BaseModel(model_name=model_name, model_params={})
        model_id = clf.model_id
        pickle.dump(clf, open(f'{model_id}.pkl', 'wb'))
        models[str(model_id)] = model_name
        msg = f"Model {model_name} added, id: {model_id}"
        return msg


@api.route("/model/list")
class List(Resource):
    @api.doc(responses={200: "Success"})
    def get(self):
        if len(models) == 0:
            return 'No models added'
        else:
            return models


@api.route("/model/delete")
class Delete(Resource):
    @api.expect(delete)
    @api.doc(
        responses={
            200: "Success",
            500: "No such model"
        })
    def delete(self):
        model_id = api.payload["model_id"]
        if model_id in models.keys():
            models.pop(model_id)
            os.remove(f'{model_id}.pkl')
            msg = f'Model {model_id} deleted'
        else:
            msg = 'No such model'
        return msg


@api.route("/model/train")
class Train(Resource):
    @api.expect(train)
    @api.doc(
        responses={
            200: "Success",
            500: "No such model"
        })
    def post(self):
        model_id = api.payload["model_id"]
        if model_id in models.keys():
            model_id = api.payload["model_id"]
            pickled_model = pickle.load(open(f'{model_id}.pkl', 'rb'))
            pickled_model.fit()
            pickle.dump(pickled_model, open(f'{model_id}.pkl', 'wb'))
            msg = f'Model {model_id} trained'
        else:
            msg = 'No such model'
        return msg


@api.route("/model/predict")
class Predict(Resource):
    @api.expect(predict_sample)
    @api.doc(
        responses={
            200: "Success",
            500: "No such model"
        })
    def post(self):
        model_id = api.payload["model_id"]
        if model_id in models.keys():
            model_id = api.payload["model_id"]
            pickled_model = pickle.load(open(f'{model_id}.pkl', 'rb'))
            return jsonify(pickled_model.predict().tolist())
        else:
            return 'No such model'


if __name__ == "__main__":
    app.run(debug=True)
