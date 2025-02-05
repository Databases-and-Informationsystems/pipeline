FROM pytorch/pytorch


WORKDIR /app

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y git && apt-get clean

ADD ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
ADD . /app

EXPOSE 8080

CMD ["flask", "run", "--host=0.0.0.0"]