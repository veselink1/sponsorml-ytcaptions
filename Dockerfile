FROM public.ecr.aws/lambda/python:3.9
RUN mkdir ${LAMBDA_TASK_ROOT}/app

COPY . ${LAMBDA_TASK_ROOT}/app
# COPY __init__.py lambda_function.py requirements.txt data_loader.py sequence_labelling.py preprocessing.py dataset_scripts ./
# COPY model /opt/ml/model

RUN python3.9 -m pip install -r ${LAMBDA_TASK_ROOT}/app/requirements.txt

CMD ["app.lambda_function.lambda_handler"]
