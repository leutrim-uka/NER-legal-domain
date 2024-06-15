# set up the project and install all necessary dependencies
setup:
	echo "setting up virtual environment"
	pip3 install virtualenv
	echo "create virtual environment"
	virtualenv venv

install:
	echo "installing all dependencies"
	pip3 install -r ./requirements.txt

