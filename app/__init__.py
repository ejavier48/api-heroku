from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from uuid import uuid4
import logging
import json
from app.agent import Agent, ProcessPolicy

app = Flask(__name__, static_folder='./')
app.secret_key =  str(uuid4())

cors = CORS(app, resources = { r"/*" : { "origins" : "*" } })

policyFile = "app/policy.json"
proPolicy = ProcessPolicy()
Q = proPolicy.getPolicy(policyFile)

@app.route('/')
def index():
    return 'Index Page'

@app.route("/api/agent", methods = ['POST'])
@cross_origin(origin = "*", headers=['Content-Type', 'Authorization'])
def get_simulation_data():
    uid = uuid4()
    logger = logging.getLogger('get_simulation_data' + str(uid))

    try:
        data = request.get_json()
        agent = Agent(data['data'], Q)
        response = agent.run()
        return jsonify(response)
    except Exception as e:
        logger.warning('Id:{} Error: {}'.format(str(uid), str(e)))
        raise('Id:{} Error: {}'.format(str(uid), str(e)))
