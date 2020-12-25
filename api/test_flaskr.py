import os
import tempfile
from flask import json

import pytest

from flaskr import create_app

# to run the tests
# python -m pytest

# to activate virtual environment
# . api/venv/Scripts/activate


@pytest.fixture
def client():
    with create_app().test_client() as client:
        # with create_app().app_context():
        #     flaskr.init_db()
        yield client

    # os.close(db_fd)
    # os.unlink(flaskr.app.config['DATABASE'])


def test_root_response_hello_world(client):
    """Should return Hello world"""

    rv = client.get('/')
    assert b'Hello, World!' in rv.data


def test_gdp_per_capita_too_little_params(client):
    """
    then parameters less than 7 return 409 conflict
    """
    rqs = {"grossNationalSavings": 10,
           "continent": "Europe"}

    rv = client.post(
        '/gdp-per-capita',
        json=rqs,
        headers={'Content-type':'application/json'})

    assert 409 == rv.status_code

def test_gdp_per_capita_unable_to_parse_request(client):
    """
    request is unparsable should return bad request
    """
    rqs = {"grossNationalSavings": 10,
           "continent": "Europe"}

    rv = client.post(
        '/gdp-per-capita',
        data=rqs,
        headers={'Content-type':'application/text'})

    assert 400 == rv.status_code

def test_gdp_per_capita_retuns_ok(client):
    """
    successfull request returns predicted GDP
    """
    rqs = {"grossNationalSavings": 10,
           "continent": "Europe",
           "param1":1,
           "param2":1,
           "param3":1,
           "param4":1,
           "param5":1
           }
# https://stackoverflow.com/a/28840457
    rv = client.post(
        '/gdp-per-capita',
        json=rqs,
        headers={'Content-type':'application/json'})

    assert 200 == rv.status_code
    assert rv.json['gdpPerCapita']
