from flask import make_response, request, current_app, Flask
app = Flask(__name__)
app.debug = True
from functools import update_wrapper
from datetime import timedelta
import json

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

millifactor = 10 ** 6.
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Series):
            if np.isnan(obj).any():
                l = obj.tolist()
                for offset, val in enumerate(l):
                    if np.isnan(val):
                        l[offset] = None
                return l
            else:
                return obj.tolist()
        elif isinstance(obj, np.ndarray):
            if obj.dtype.kind == 'M':
                return obj.astype('datetime64[ms]').astype('int').tolist() 
            return obj.tolist()
        elif isinstance(obj, np.number):
            if isinstance(obj, np.integer):
                return int(obj)
            else:
                return float(obj)
        elif isinstance(obj, pd.tslib.Timestamp):
            #return obj.value / millifactor
            return obj.value / millifactor
        else:
            return super(NumpyJSONEncoder, self).default(obj)

def serialize_json(obj, encoder=NumpyJSONEncoder, **kwargs):
    return json.dumps(obj, cls=encoder, **kwargs)


@app.route('/columns')
@crossdomain(origin='*')
def columns():
    return json.dumps({'columns':df.columns.tolist()})

@app.route('/values/<column_name>')
@crossdomain(origin='*')
def values(column_name):
    return serialize_json({column_name:df[column_name]})

@app.route('/values_index/<column_name>')
@crossdomain(origin='*')
def values_index(column_name):
    return json.dumps({column_name:df[column_name].tolist(), 
                       'index':df.index.tolist()})
@app.route('/index')
@crossdomain(origin='*')
def index():
    return serialize_json({'index': np.array(df.index.tolist()).tolist() })


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


import pandas as pd
import numpy as np


import argparse
if __name__ == '__main__':




    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-f", "--hf5file", dest="hf5_filename")
    parser.add_argument("-k", "--hf5_key", dest="hf5_key")
    parser.add_argument("-p", "--port", dest="port", default=5000, type=int)
    args = parser.parse_args()
    if args.hf5_filename:
        df = pd.read_hdf(args.hf5_filename, args.hf5_key)
    else:
        df = pd.DataFrame([
            {'a':0,  'b':5},
            {'a':8,  'b':9},
            {'a':7,  'b':12},
            {'a':13, 'b':2, 'c':.9},
            {'a':34,  'b':28},
        ])
        df = pd.DataFrame({'a':np.arange(10000), 'b':np.random.random(10000)*5000})

    app.run(port=args.port)


