FROM --platform=${BUILDPLATFORM} python:3.9

WORKDIR /app

# Add the trainer user.
RUN \
    groupadd -g 1001 trainer \
    && useradd --no-log-init --create-home --shell /bin/bash --gid 1001 --uid 1001 trainer

RUN \
    apt-get update \
    && apt-get install -y \
      htop \
      sudo \
      tmux \
      vim \
    && apt-get clean

RUN usermod -aG sudo trainer \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Cache PostgreSQL build dependencies.
RUN \
    apt-get update \
    && apt-get install -y \
      bison \
      build-essential \
      flex \
      libreadline-dev \
      libssl-dev \
      libxml2-dev \
      libxml2-utils \
      libxslt-dev \
      pg-bsd-indent \
      xsltproc \
      zlib1g-dev \
    && apt-get clean

RUN chown -R trainer:trainer /app

USER trainer

COPY --chown=trainer:trainer requirements.txt /app
# Install Python packages.
RUN --mount=type=cache,mode=0755,target=/home/trainer/.cache/pip \
    pip3 install -r requirements.txt
ENV PATH="${PATH}:/home/trainer/.local/bin"

COPY --chown=trainer:trainer . /app

COPY docker-entrypoint.sh /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]