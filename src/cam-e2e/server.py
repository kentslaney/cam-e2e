import numpy as np
import flask, json, argparse, pathlib, cv2, functools, inspect, io
from werkzeug.utils import secure_filename

project_path = pathlib.Path(__file__).parents[2]
jobs_path = project_path / "jobs"
app = flask.Flask(__name__, static_folder=project_path / "src" / "flask-static")
jobs_list = lambda: [i.name for i in jobs_path.iterdir()]

@app.route("/jobs")
def jobs():
    return json.dumps(jobs_list())

def sanitize_job(arg):
    def decorator(f):
        sig = inspect.signature(f)
        @functools.wraps(f)
        def wrapper(*a, **kw):
            bound = sig.bind_partial(*a, **kw)
            if secure_filename(bound.arguments[arg]) != bound.arguments[arg]:
                flask.abort(400)
            if bound.arguments[arg] not in jobs_list():
                flask.abort(404)
            return f(*bound.args, **bound.kwargs)
        return wrapper
    return decorator

@app.route("/jobs/<job_name>/")
@sanitize_job("job_name")
def job(job_name):
    return app.send_static_file("job.html")

def encode_img(uint8s):
    success, buffer = cv2.imencode(".png", uint8s)
    if not success:
        flask.abort(500)
    buffer = io.BytesIO(buffer)
    return flask.Response(buffer, mimetype='image/png')

@app.route("/jobs/<job_name>/input.png")
@sanitize_job("job_name")
def input_image(job_name):
    return encode_img(np.load(jobs_path / job_name / "image_uint8.npy"))

def requires(**args):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*a, **kw):
            if not all(i in flask.request.args for i in args):
                flask.abort(400)
            try:
                extra = {k: v(flask.request.args[k]) for k, v in args.items()}
            except ValueError:
                flask.abort(400)
            return f(*a, **{**kw, **extra})
        return wrapper
    return decorator

def bounded(lo, hi):
    assert type(lo) == type(hi)
    cast = type(lo)
    def caller(x):
        x = cast(x)
        if not lo <= x <= hi:
            raise ValueError
        return x
    return caller

embedding_size = (224, 224) # rows, cols

@app.route("/jobs/<job_name>/knn")
@sanitize_job("job_name")
@requires(x=bounded(0., 1.), y=bounded(0., 1.))
def knn_mask(job_name, x, y):
    x, y = int(x * embedding_size[1]), int(y * embedding_size[0])
    i = y * embedding_size[1] + x
    knn = np.load(jobs_path / job_name / "flat_knn.npy")[i]
    knn = np.stack((knn // embedding_size[1], knn % embedding_size[1]), 0)
    out = np.zeros(embedding_size + (4,), dtype=np.uint8)
    out[*knn] = [0, 0, 255, 255]
    out[y, x] = [0, 255, 0, 255]
    return encode_img(out)

@app.route("/jobs/<job_name>/l2")
@sanitize_job("job_name")
@requires(
        x0=bounded(0., 1.), y0=bounded(0., 1.),
        x1=bounded(0., 1.), y1=bounded(0., 1.),
        t=bounded(0., 1.),  c=bounded(0., 1.), i=bounded(0, 1))
def l2(job_name, x0, y0, x1, y1, t, c, i):
    x0, y0 = int(x0 * embedding_size[1]), int(y0 * embedding_size[0])
    x1, y1 = int(x1 * embedding_size[1]), int(y1 * embedding_size[0])
    if i == 0:
        out = np.zeros(embedding_size + (4,), dtype=np.uint8)
        out[..., 2] = 255
    else:
        out = np.load(jobs_path / job_name / "image_uint8.npy")
        if out.shape != embedding_size:
            out = cv2.resize(out, embedding_size[::-1])
        out = np.dstack((out, np.zeros(out.shape[:-1] + (1,))))
    embeddings = np.load(jobs_path / job_name / "flat_embeddings.npy")
    embeddings = embeddings.reshape(*embedding_size, -1)
    if c != 0:
        im = np.load(jobs_path / job_name / "image_uint8.npy")
        if im.shape != embedding_size:
            im = cv2.resize(im, embedding_size[::-1])
        embeddings = np.dstack((embeddings * (1 - c), im / 255 * c))
    if t != 0:
        pos = np.stack(np.meshgrid(
            np.linspace(0, 1, embedding_size[1]),
            np.linspace(0, 1, embedding_size[0])), -1)
        embeddings = np.dstack((embeddings * (1 - t), pos * t))
    embeddings = np.sum((embeddings - embeddings[y0, x0]) ** 2, -1)
    out[embeddings < embeddings[y1, x1], 3] = 255
    out[y0, x0] = [0, 255, 0, 255]
    out[y1, x1] = [0, 255, 0, 255]
    return encode_img(out)

if __name__ == "__main__":
    app.run(debug=True)
