.PHONY: venv

server: .venv
	.venv/bin/python src/cam-e2e/server.py

venv:
	@if ! which python >/dev/null; then exit 1; fi
	if [ ! -d .venv ]; then python -m venv .venv; fi
	if [ ! -d .venv/lib/*/site-packages/numpy ]; \
		then .venv/bin/pip install -U numpy; fi
	if [ ! -d .venv/lib/*/site-packages/sklearn ]; \
		then .venv/bin/pip install -U scikit-learn; fi
	if [ ! -d .venv/lib/*/site-packages/flask ]; \
		then .venv/bin/pip install -U flask; fi
	if [ ! -d .venv/lib/*/site-packages/cv2 ]; \
		then .venv/bin/pip install -U opencv-python; fi

.venv: venv
