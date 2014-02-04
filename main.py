
import pandas as pd
import numpy as np
import json
df = pd.DataFrame([
    {'a':0,  'b':5},
    {'a':8,  'b':9},
    {'a':7,  'b':12},
    {'a':13, 'b':2, 'c':.9},
    {'a':34,  'b':28},
])


df = pd.DataFrame({'a':np.arange(10000), 'b':np.random.random(10000)*5000})



from flask import request
from flask import Flask
app = Flask(__name__)
app.debug = True
from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper


def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator


@app.route('/columns')
@crossdomain(origin='*')
def columns():
    return json.dumps({'columns':df.columns.tolist()})

@app.route('/values/<column_name>')
@crossdomain(origin='*')
def values(column_name):
    return json.dumps({column_name:df[column_name].tolist()})

@app.route('/values_index/<column_name>')
@crossdomain(origin='*')
def values_index(column_name):
    return json.dumps({column_name:df[column_name].tolist(), 
                       'index':df.index.tolist()})


class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # or map(int, obj)
        return json.JSONEncoder.default(self, obj)

@app.route('/index')
@crossdomain(origin='*')
def index():
    return json.dumps({'index': np.array(df.index.tolist()).tolist() })




@app.route('/slice/<column_name>')
@crossdomain(origin='*')
def slice(column_name):
    start_raw, end_raw = [request.args['start'], request.args['end']]
    if df.index.dtype == np.int:
        start, end = [int(start_raw), int(end_raw)]
    else:
        start, end = [start_raw, end_raw]

    # I might eventually apply some type conversion here, based on the
    # type of the index, for now though, I'll leave it

    ts = df[column_name]
    return json.dumps(ts[start:end].tolist())


if __name__ == '__main__':
    app.run()
