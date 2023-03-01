FROM --platform=${BUILDPLATFORM} python:3.9

WORKDIR /app

# Add the nyoom user.
RUN \
    groupadd -g 1001 nyoom \
    && useradd --no-log-init --create-home --shell /bin/bash --gid 1001 --uid 1001 nyoom

RUN \
    apt-get update \
    && apt-get install -y \
      htop \
      sudo \
      tmux \
      vim \
    && apt-get clean

RUN usermod -aG sudo nyoom \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN \
    apt-get update \
    && apt-get install -y \
      graphviz \
      graphviz-dev \
    && apt-get clean

RUN chown -R nyoom:nyoom /app

USER nyoom

COPY --chown=nyoom:nyoom requirements.txt /app
# Install Python packages.
RUN --mount=type=cache,mode=0755,target=/home/nyoom/.cache/pip \
    pip3 install -r requirements.txt
ENV PATH="${PATH}:/home/nyoom/.local/bin"

COPY --chown=nyoom:nyoom . /nyoom_default

COPY docker-entrypoint.sh /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]