FROM python:3.9-slim-buster
WORKDIR /chatbot

RUN mkdir -p /usr/share/man/man1 /usr/share/man/man2 && \
    apt-get update &&\
    apt-get install -y --no-install-recommends openjdk-11-jre && \
    apt-get install ca-certificates-java -y && \
    apt-get clean && \
    update-ca-certificates -f;
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /chatbot

EXPOSE 5000
CMD ["python", "app.py"]