create the requirements.txt using
conda list | grep 'pypi' | awk '{print $1 "==" $2}' > requirements.txt

create the Dockerfile
