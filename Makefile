ininstall_env:
	pip3.7 install -r modules/requirements.txt

install_dev_env:
	pip3.7 install -r dev/requirements.txt

build_package:
	python3.7 setup.py sdist bdist -b ~/temp/bdistwheel_$$ bdist_wheel --bdist-dir ~/temp/bdistwheel_$$
	
install:
	python3.7 setup.py install

clean:
	rm -rf build *.egg-info modules/*.egg-info
	
py_venv:
	python3.7 -m venv venv/
	# source ./venv/bin/activate
