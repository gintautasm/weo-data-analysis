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
        yield client

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

    params from germany
    2008
    NGDPRPPPPC 48,641.279 
    91.392
    91.200
    7.383
    37.644
    80.764
    predicted 50970.0028
    """
    rqs = {"grossNationalSavings": 10,
           "continent": "Europe",
           "PCPI":91.392,
           "PCPIE":91.200,
           "LUR":7.383,
           "LE":37.644,
           "LP":80.764
           }
    # https://stackoverflow.com/a/28840457
    rv = client.post(
        '/gdp-per-capita',
        json=rqs,
        headers={'Content-type':'application/json'})

    assert 200 == rv.status_code
    assert rv.json['gdpPerCapita']
